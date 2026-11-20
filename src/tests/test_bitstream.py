"""
Тесты для BitWriter / BitReader (LSB-first, как в VP8L).

Проверяем:
- порядок битов внутри байта (LSB-first);
- склейку значений через границу байта;
- n=0 как no-op;
- маску старших бит;
- round-trip на случайных данных;
- EOFError при чтении за пределами буфера;
- bits_left / len.
"""
import random

import pytest

from webp_codec.bitstream import BitReader, BitWriter


# ------------------------------------------------------------------
# BitWriter: базовая упаковка
# ------------------------------------------------------------------
class TestBitWriter:
    def test_example_from_docstring(self):
        """write_bit(1), write_bit(0), write_bit(1) → байт 0b0000_0101."""
        w = BitWriter()
        w.write_bit(1)
        w.write_bit(0)
        w.write_bit(1)
        assert w.finish() == b"\x05"

    def test_write_bits_across_byte_boundary(self):
        """3 бита + 2 бита = 5 бит → один байт 0b0001_1101."""
        w = BitWriter()
        w.write_bits(0b101, 3)
        w.write_bits(0b11, 2)
        assert w.finish() == bytes([0b0001_1101])   # 0x1D

    def test_write_full_byte(self):
        w = BitWriter()
        w.write_bits(0xFF, 8)
        assert w.finish() == b"\xff"

    def test_write_zero_bits_is_noop(self):
        """n=0 не должен ничего писать и не должен падать."""
        w = BitWriter()
        w.write_bits(123, 0)
        assert w.finish() == b""

    def test_write_zero_bits_does_not_disturb_state(self):
        """n=0 не сдвигает счётчик, если уже что-то записано."""
        w = BitWriter()
        w.write_bits(0b1, 1)
        w.write_bits(0xFFFF, 0)      # no-op
        w.write_bits(0b0, 1)
        assert w.finish() == b"\x01"  # биты 1, 0 → 0b01

    def test_write_bits_masks_high_bits(self):
        """Старшие биты value за пределами n отрезаются маской."""
        w = BitWriter()
        w.write_bits(0xFFFF, 3)   # должны записаться только младшие 3 бита = 0b111
        assert w.finish() == b"\x07"

    def test_write_multi_byte_value(self):
        """32-битное значение в один write_bits → ровно 4 байта LSB-first."""
        w = BitWriter()
        w.write_bits(0x12345678, 32)
        assert w.finish() == bytes([0x78, 0x56, 0x34, 0x12])

    def test_write_bits_large_n_overflow_byte(self):
        """7 бит + 32 бита = 39 бит, должно вытолкнуть 4 байта + хвост."""
        w = BitWriter()
        w.write_bits(0b1111111, 7)             # 7 бит
        w.write_bits(0xFFFFFFFF, 32)           # ещё 32 бита → всего 39
        data = w.finish()
        assert len(data) == 5                  # ceil(39/8) = 5 байт
        # проверим, что обратно читается всё, что писали
        r = BitReader(data)
        assert r.read_bits(7) == 0b1111111
        assert r.read_bits(32) == 0xFFFFFFFF

    def test_negative_n_raises(self):
        w = BitWriter()
        with pytest.raises(ValueError):
            w.write_bits(0, -1)

    def test_too_large_n_raises(self):
        w = BitWriter()
        with pytest.raises(ValueError):
            w.write_bits(0, 33)

    def test_finish_is_idempotent(self):
        """Повторный finish() не должен дублировать хвост."""
        w = BitWriter()
        w.write_bit(1)
        first = w.finish()
        second = w.finish()
        assert first == second == b"\x01"

    def test_len_before_finish(self):
        """len(w) считает биты без добивки."""
        w = BitWriter()
        assert len(w) == 0
        w.write_bits(0b101, 3)
        assert len(w) == 3
        w.write_bits(0b11, 2)
        assert len(w) == 5


# ------------------------------------------------------------------
# BitReader: базовое чтение
# ------------------------------------------------------------------
class TestBitReader:
    def test_read_single_bit(self):
        r = BitReader(b"\x05")    # 0b0000_0101
        assert r.read_bit() == 1
        assert r.read_bit() == 0
        assert r.read_bit() == 1

    def test_read_multi_bit(self):
        """0b0001_1101: читаем 3 бита → 0b101, потом 2 бита → 0b11."""
        r = BitReader(bytes([0b0001_1101]))
        assert r.read_bits(3) == 0b101
        assert r.read_bits(2) == 0b11

    def test_read_full_byte(self):
        r = BitReader(b"\xff")
        assert r.read_bits(8) == 0xFF

    def test_read_zero_bits_returns_zero(self):
        r = BitReader(b"\xff")
        assert r.read_bits(0) == 0
        # состояние не изменилось — можно прочитать тот же байт
        assert r.read_bits(8) == 0xFF

    def test_read_across_byte_boundary(self):
        """4 + 4 бита из двух байт."""
        r = BitReader(bytes([0xAB, 0xCD]))
        assert r.read_bits(4) == 0xB
        assert r.read_bits(4) == 0xA
        assert r.read_bits(4) == 0xD
        assert r.read_bits(4) == 0xC

    def test_read_32_bits(self):
        r = BitReader(bytes([0x78, 0x56, 0x34, 0x12]))
        assert r.read_bits(32) == 0x12345678

    def test_read_zero_bits_does_not_advance(self):
        r = BitReader(b"\x00\x00")
        before = r.bits_left()
        r.read_bits(0)
        assert r.bits_left() == before


# ------------------------------------------------------------------
# EOF / границы буфера
# ------------------------------------------------------------------
class TestBitReaderBounds:
    def test_eof_after_reading_all(self):
        r = BitReader(b"\x00")
        r.read_bits(8)
        with pytest.raises(EOFError):
            r.read_bits(1)

    def test_eof_partial(self):
        """3 бита остаётся, просим 5 — EOFError."""
        r = BitReader(b"\xff")
        r.read_bits(5)
        with pytest.raises(EOFError):
            r.read_bits(5)

    def test_empty_buffer_eof(self):
        r = BitReader(b"")
        with pytest.raises(EOFError):
            r.read_bits(1)

    def test_read_zero_bits_from_empty_ok(self):
        r = BitReader(b"")
        assert r.read_bits(0) == 0

    def test_bits_left(self):
        r = BitReader(b"\xff\xff")
        assert r.bits_left() == 16
        r.read_bits(5)
        assert r.bits_left() == 11
        r.read_bits(11)
        assert r.bits_left() == 0

    def test_negative_n_raises(self):
        r = BitReader(b"\xff")
        with pytest.raises(ValueError):
            r.read_bits(-1)

    def test_too_large_n_raises(self):
        r = BitReader(b"\xff" * 8)
        with pytest.raises(ValueError):
            r.read_bits(33)


# ------------------------------------------------------------------
# Round-trip: write → read
# ------------------------------------------------------------------
class TestRoundTrip:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32])
    def test_single_value_each_width(self, n):
        """Записать одно значение ширины n и прочитать обратно."""
        value = (1 << n) - 1 if n <= 32 else 0
        w = BitWriter()
        w.write_bits(value, n)
        r = BitReader(w.finish())
        assert r.read_bits(n) == value

    def test_random_sequence(self):
        """1000 случайных (value, n) — записать, прочитать, сравнить."""
        rnd = random.Random(42)
        pairs = []
        for _ in range(1000):
            n = rnd.randint(1, 32)
            v = rnd.randrange(1 << n)
            pairs.append((v, n))

        w = BitWriter()
        for v, n in pairs:
            w.write_bits(v, n)
        data = w.finish()

        r = BitReader(data)
        got = [(r.read_bits(n), n) for _, n in pairs]
        assert got == pairs

    def test_all_bit_positions(self):
        """Записать 0b101_1010... по одному биту, прочитать по одному."""
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        w = BitWriter()
        for b in bits:
            w.write_bit(b)
        r = BitReader(w.finish())
        got = [r.read_bit() for _ in bits]
        assert got == bits

    def test_empty(self):
        """Пустой writer → пустой reader."""
        w = BitWriter()
        assert w.finish() == b""
        assert BitReader(b"").bits_left() == 0

    def test_len_matches_bits_left_after_roundtrip(self):
        """len(w) == исходное bits_left у reader'а."""
        w = BitWriter()
        w.write_bits(0b10110, 5)
        w.write_bits(0b1111, 4)
        n_total = len(w)
        data = w.finish()
        r = BitReader(data)
        assert r.bits_left() >= n_total
        # после чтения всех записанных бит — остаётся только padding
        for _ in range(n_total):
            r.read_bit()
        assert r.bits_left() == len(data) * 8 - n_total