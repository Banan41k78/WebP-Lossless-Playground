"""
Тесты для webp_codec.header.

Проверяем:
- сборку/разбор 5-байтного VP8L-заголовка (40 бит);
- граничные размеры (1 и 16384);
- корректность RIFF-обёртки и padding;
- round-trip build ↔ parse;
- главное: Pillow/libwebp распознаёт наш заголовок.
"""
import io
import random

import pytest
from PIL import Image

from webp_codec.header import (
    VP8L_HEADER_SIZE,
    VP8L_SIGNATURE,
    VP8L_VERSION,
    build_riff,
    build_vp8l_header,
    build_webp,
    parse_riff,
    parse_vp8l_header,
    parse_webp,
)


# ------------------------------------------------------------------
# Константы
# ------------------------------------------------------------------
class TestConstants:
    def test_signature(self):
        assert VP8L_SIGNATURE == 0x2F

    def test_version(self):
        assert VP8L_VERSION == 0

    def test_header_size(self):
        assert VP8L_HEADER_SIZE == 5


# ------------------------------------------------------------------
# build_vp8l_header: конкретные байты
# ------------------------------------------------------------------
class TestBuildVP8LHeader:
    def test_minimal_1x1(self):
        """Минимальный валидный заголовок: всё в нулях, кроме сигнатуры."""
        assert build_vp8l_header(1, 1, False) == b"\x2f\x00\x00\x00\x00"

    def test_known_value_100x50_alpha(self):
        """width=100, height=50, alpha=True → 2F 63 40 0C 10."""
        assert build_vp8l_header(100, 50, True) == b"\x2f\x63\x40\x0c\x10"

    def test_max_size(self):
        """16384×16384 — все 14 бит = 1, alpha=False."""
        data = build_vp8l_header(16384, 16384, False)
        assert len(data) == VP8L_HEADER_SIZE
        # сигнатура на месте
        assert data[0] == VP8L_SIGNATURE
        # версия = 0 → старшие 3 бита последнего байта = 0
        assert data[4] & 0b11100000 == 0

    def test_length_always_5(self):
        for w, h, a in [(1, 1, False), (100, 50, True), (16384, 16384, True)]:
            assert len(build_vp8l_header(w, h, a)) == VP8L_HEADER_SIZE

    def test_alpha_bit(self):
        """has_alpha=True выставляет бит 36 = бит 4 байта 4 → 0x10."""
        no_alpha = build_vp8l_header(1, 1, False)
        with_alpha = build_vp8l_header(1, 1, True)
        assert no_alpha[4] == 0x00
        assert with_alpha[4] == 0x10


# ------------------------------------------------------------------
# parse_vp8l_header
# ------------------------------------------------------------------
class TestParseVP8LHeader:
    def test_roundtrip_specific(self):
        for w, h, a in [(1, 1, False), (100, 50, True),
                        (16384, 16384, False), (256, 128, True)]:
            data = build_vp8l_header(w, h, a)
            assert parse_vp8l_header(data) == (w, h, a)

    def test_roundtrip_random(self):
        rnd = random.Random(42)
        for _ in range(20):
            w = rnd.randint(1, 16384)
            h = rnd.randint(1, 16384)
            a = rnd.choice([True, False])
            data = build_vp8l_header(w, h, a)
            assert parse_vp8l_header(data) == (w, h, a)

    def test_bad_signature_raises(self):
        with pytest.raises(ValueError):
            parse_vp8l_header(b"\x00\x00\x00\x00\x00")

    def test_bad_version_raises(self):
        """Версия != 0 → ValueError. Version — биты 5..7 байта 4."""
        good = build_vp8l_header(1, 1, False)     # b"\x2f\x00\x00\x00\x00"
        bad = good[:4] + b"\x20"                    # version=1 (бит 5)
        with pytest.raises(ValueError):
            parse_vp8l_header(bad)

    def test_extra_bytes_ignored(self):
        """Если передали больше 5 байт — читаются только первые 5."""
        data = build_vp8l_header(10, 20, True) + b"\xde\xad\xbe\xef"
        assert parse_vp8l_header(data) == (10, 20, True)


# ------------------------------------------------------------------
# build_riff / parse_riff
# ------------------------------------------------------------------
class TestRIFF:
    def test_build_starts_with_riff(self):
        data = build_riff(b"")
        assert data[:4] == b"RIFF"

    def test_build_contains_webp(self):
        data = build_riff(b"")
        assert data[8:12] == b"WEBP"

    def test_build_contains_vp8l_fourcc(self):
        data = build_riff(b"")
        assert data[12:16] == b"VP8L"

    def test_size_field(self):
        """Поле размера = file_size − 8."""
        data = build_riff(b"\x00" * 10)
        size = int.from_bytes(data[4:8], "little")
        assert size == len(data) - 8

    def test_chunk_size_field(self):
        """Поле размера чанка = длина vp8l_data (без padding)."""
        payload = b"\x00" * 7
        data = build_riff(payload)
        chunk_size = int.from_bytes(data[16:20], "little")
        assert chunk_size == 7

    def test_padding_even(self):
        """Чётный payload → padding нет, длина файла чётная."""
        data = build_riff(b"\x00" * 10)
        assert len(data) % 2 == 0
        # 12 (RIFF+size+WEBP) + 8 (VP8L+chunk_size) + 10 = 30
        assert len(data) == 30

    def test_padding_odd(self):
        """Нечётный payload → 1 байт padding, длина всё равно чётная."""
        data = build_riff(b"\x00" * 7)
        assert len(data) % 2 == 0
        # 12 + 8 + 7 + 1 = 28
        assert len(data) == 28
        # последний байт — padding 0x00
        assert data[-1] == 0x00

    def test_parse_roundtrip(self):
        for payload in [b"", b"\x00", b"\x01\x02\x03", b"\xff" * 100]:
            assert parse_riff(build_riff(payload)) == payload

    def test_parse_rejects_non_riff(self):
        with pytest.raises(ValueError):
            parse_riff(b"NOPE" + b"\x00" * 20)

    def test_parse_rejects_non_webp(self):
        bad = b"RIFF" + b"\x00" * 4 + b"NOPE" + b"\x00" * 20
        with pytest.raises(ValueError):
            parse_riff(bad)

    def test_parse_rejects_non_vp8l(self):
        bad = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8 " + b"\x00" * 20
        with pytest.raises(ValueError):
            parse_riff(bad)

    def test_exact_riff_bytes(self):
        """Побайтовая проверка для минимального случая."""
        data = build_riff(b"\x2f\x00\x00\x00\x00")
        expected = (
            b"RIFF" + b"\x12\x00\x00\x00" +       # file_size − 8 = 18
            b"WEBP" +
            b"VP8L" + b"\x05\x00\x00\x00" +       # chunk_size = 5
            b"\x2f\x00\x00\x00\x00" +
            b"\x00"                                # padding
        )
        assert data == expected


# ------------------------------------------------------------------
# build_webp / parse_webp
# ------------------------------------------------------------------
class TestWebP:
    def test_build_webp_roundtrip(self):
        data = build_webp(100, 50, True, b"PAYLOAD")
        w, h, a, payload = parse_webp(data)
        assert (w, h, a, payload) == (100, 50, True, b"PAYLOAD")

    def test_build_webp_empty_payload(self):
        data = build_webp(1, 1, False, b"")
        w, h, a, payload = parse_webp(data)
        assert (w, h, a, payload) == (1, 1, False, b"")


# ------------------------------------------------------------------
# Главный тест: Pillow / libwebp распознаёт наш заголовок
# ------------------------------------------------------------------
class TestPillowCompat:
    def test_pillow_reads_minimal_header(self):
        """Pillow должен распознать формат и вернуть правильный размер."""
        data = build_webp(1, 1, False, b"")
        img = Image.open(io.BytesIO(data))
        assert img.format == "WEBP"
        assert img.size == (1, 1)

    def test_pillow_reads_various_sizes(self):
        for w, h in [(1, 1), (100, 50), (256, 128), (1000, 800)]:
            data = build_webp(w, h, False, b"")
            img = Image.open(io.BytesIO(data))
            assert img.size == (w, h)

    def test_pillow_reads_alpha(self):
        data = build_webp(10, 20, True, b"")
        img = Image.open(io.BytesIO(data))
        assert img.size == (10, 20)