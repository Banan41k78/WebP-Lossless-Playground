"""Тесты для webp_codec.lossless — round-trip, Pillow-совместимость, трансформации."""
import io

import numpy as np
from PIL import Image

from webp_codec.lossless import encode_lossless, decode_lossless


# ==================================================================
# Round-trip через свой кодек
# ==================================================================
class TestRoundtrip:
    def test_1x1(self):
        px = np.array([[[100, 50, 200, 255]]], dtype=np.uint8)
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)

    def test_2x2(self):
        px = np.array([
            [[10, 20, 30, 255], [40, 50, 60, 255]],
            [[70, 80, 90, 255], [100, 110, 120, 255]],
        ], dtype=np.uint8)
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)

    def test_random_8x8(self):
        rng = np.random.default_rng(42)
        px = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)

    def test_random_16x16(self):
        rng = np.random.default_rng(7)
        px = rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8)
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)

    def test_solid_color(self):
        """Все пиксели одинаковые — все каналы 1-символьные."""
        px = np.full((4, 4, 4), 42, dtype=np.uint8)
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)

    def test_gradient(self):
        """Плавный градиент — много повторов, но пока без LZ77."""
        px = np.zeros((8, 8, 4), dtype=np.uint8)
        for x in range(8):
            px[:, x] = [x * 30, x * 30, x * 30, 255]
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)

    def test_with_alpha(self):
        """Канал alpha с разными значениями."""
        px = np.array([
            [[10, 20, 30, 0], [40, 50, 60, 128]],
            [[70, 80, 90, 200], [100, 110, 120, 255]],
        ], dtype=np.uint8)
        assert np.array_equal(decode_lossless(encode_lossless(px)), px)


# ==================================================================
# Совместимость с libwebp (Pillow)
# ==================================================================
class TestPillowCompat:
    def test_pillow_reads_header(self):
        """Image.open должен распознать формат и вернуть правильный размер."""
        px = np.array([[[100, 50, 200, 255]]], dtype=np.uint8)
        data = encode_lossless(px)
        img = Image.open(io.BytesIO(data))
        assert img.format == "WEBP"
        assert img.size == (1, 1)

    def test_pillow_reads_various_sizes(self):
        for h, w in [(1, 1), (2, 3), (8, 8), (16, 16)]:
            rng = np.random.default_rng(h * w)
            px = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
            data = encode_lossless(px)
            img = Image.open(io.BytesIO(data))
            assert img.size == (w, h)

    def test_pillow_decodes_pixels(self):
        """Pillow декодирует НАШИ пиксели — это бит-совместимость с libwebp."""
        px = np.array([
            [[100, 50, 200, 255], [10, 20, 30, 40]],
            [[1, 2, 3, 4],       [250, 251, 252, 253]],
        ], dtype=np.uint8)
        data = encode_lossless(px)
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img)
        assert np.array_equal(arr, px), (
            f"libwebp декодировал не то:\n"
            f"  ожидалось: {px.tolist()}\n"
            f"  получено:  {arr.tolist()}"
        )


# ==================================================================
# Subtract green transform
# ==================================================================
class TestSubtractGreen:
    def test_roundtrip_with_subtract_green(self):
        rng = np.random.default_rng(11)
        px = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        data = encode_lossless(px, use_subtract_green=True)
        assert np.array_equal(decode_lossless(data), px)

    def test_subtract_green_reduces_size(self):
        """На градиенте subtract-green должен давать меньший файл."""
        # Плавный градиент: R и G сильно коррелируют
        h, w = 16, 16
        px = np.zeros((h, w, 4), dtype=np.uint8)
        for x in range(w):
            for y in range(h):
                v = (x * 10 + y * 5) & 0xFF
                px[y, x] = [v, v, v, 255]
        size_off = len(encode_lossless(px, use_subtract_green=False))
        size_on = len(encode_lossless(px, use_subtract_green=True))
        assert size_on < size_off, (
            f"subtract-green должен уменьшить размер: {size_off} → {size_on}"
        )


# ==================================================================
# LZ77
# ==================================================================
class TestLZ77:
    def test_roundtrip_with_lz77(self):
        """Повторяющиеся пиксели → LZ77 сжимает, round-trip работает."""
        px = np.zeros((8, 8, 4), dtype=np.uint8)
        px[:] = [100, 150, 200, 255]      # все пиксели одинаковые
        data = encode_lossless(px, use_lz77=True)
        assert np.array_equal(decode_lossless(data), px)

    def test_lz77_reduces_size_on_repeats(self):
        """Однотонная картинка: с LZ77 файл МЕНЬШЕ, чем без."""
        px = np.full((16, 16, 4), 42, dtype=np.uint8)
        size_with = len(encode_lossless(px, use_lz77=True))
        size_without = len(encode_lossless(px, use_lz77=False))
        assert size_with < size_without, (
            f"LZ77 должен уменьшить размер: {size_without} → {size_with}"
        )

    def test_pillow_with_lz77(self):
        """libwebp декодирует наш LZ77-поток — момент истины."""
        px = np.zeros((4, 4, 4), dtype=np.uint8)
        px[0, 0] = [1, 2, 3, 4]
        px[1, 1] = [5, 6, 7, 8]
        # остальные — нули: LZ77 найдёт длинные повторы нулей
        data = encode_lossless(px, use_lz77=True)
        arr = np.array(Image.open(io.BytesIO(data)).convert("RGBA"))
        assert np.array_equal(arr, px)