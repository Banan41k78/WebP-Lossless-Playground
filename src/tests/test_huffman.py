"""Тесты для webp_codec.huffman."""
import numpy as np
import pytest

from webp_codec.bitstream import BitReader, BitWriter
from webp_codec.huffman import (
    build_code_lengths,
    build_canonical_codes,
    compress_code_lengths,
    decompress_code_lengths,
    HuffmanDecoder,
    write_simple_code, read_simple_code,
    write_normal_code, read_normal_code,
    write_huffman_code, read_huffman_code,
)


# ==================================================================
# build_code_lengths
# ==================================================================
class TestBuildCodeLengths:
    def test_empty(self):
        assert build_code_lengths([]) == []

    def test_all_zero(self):
        assert build_code_lengths([0, 0, 0]) == [0, 0, 0]

    def test_single_symbol(self):
        assert build_code_lengths([42, 0, 0]) == [1, 0, 0]

    def test_two_symbols(self):
        assert build_code_lengths([5, 3]) == [1, 1]

    def test_known_distribution(self):
        """[5, 3, 2, 1] → длины [1, 2, 3, 3] (Kraft = 1)."""
        lengths = build_code_lengths([5, 3, 2, 1])
        assert sorted(lengths) == [1, 2, 3, 3]

    def test_kraft_inequality(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            freqs = rng.integers(1, 100, size=20).tolist()
            lengths = build_code_lengths(freqs)
            kraft = sum(2 ** -l for l in lengths if l > 0)
            assert kraft <= 1.0 + 1e-9


# ==================================================================
# build_canonical_codes
# ==================================================================
class TestCanonicalCodes:
    def test_known_lengths(self):
        """[1, 2, 3, 3] → LSB-first коды (0,1), (1,2), (3,3), (7,3)."""
        codes = build_canonical_codes([1, 2, 3, 3])
        assert codes[0] == (0, 1)
        assert codes[1] == (1, 2)   # MSB 10 → LSB 01
        assert codes[2] == (3, 3)   # MSB 110 → LSB 011
        assert codes[3] == (7, 3)   # MSB 111 → LSB 111

    def test_prefix_free(self):
        """Ни один код не является префиксом другого (MSB-first)."""
        codes = build_canonical_codes([1, 2, 3, 3])
        msb = {}
        for sym, (rev, length) in codes.items():
            # обратно переворачиваем, чтобы получить MSB-first
            val = 0
            for i in range(length):
                val = (val << 1) | ((rev >> i) & 1)
            msb[sym] = (val, length)
        items = list(msb.items())
        for i, (s1, (c1, l1)) in enumerate(items):
            for s2, (c2, l2) in items[i+1:]:
                bin1 = format(c1, f"0{l1}b")
                bin2 = format(c2, f"0{l2}b")
                assert not bin2.startswith(bin1), f"{bin1} — префикс {bin2}"
                assert not bin1.startswith(bin2), f"{bin2} — префикс {bin1}"

    def test_empty(self):
        assert build_canonical_codes([]) == {}


# ==================================================================
# HuffmanDecoder: round-trip
# ==================================================================
class TestHuffmanRoundtrip:
    def test_encode_decode(self):
        rng = np.random.default_rng(42)
        freqs = rng.integers(1, 100, size=20).tolist()
        lengths = build_code_lengths(freqs)
        codes = build_canonical_codes(lengths)
        decoder = HuffmanDecoder(lengths)

        symbols = rng.integers(0, 20, size=200).tolist()

        w = BitWriter()
        for s in symbols:
            code, length = codes[s]
            w.write_bits(code, length)
        data = w.finish()

        r = BitReader(data)
        got = [decoder.decode(r) for _ in symbols]
        assert got == symbols


# ==================================================================
# Code-length codes
# ==================================================================
class TestCodeLengthCodes:
    def test_roundtrip_simple(self):
        lengths = [1, 1, 2, 2, 2, 2]
        cl = compress_code_lengths(lengths)
        assert decompress_code_lengths(cl) == lengths

    def test_roundtrip_zeros(self):
        lengths = [0, 0, 0, 0, 0, 1, 2, 3]
        cl = compress_code_lengths(lengths)
        assert decompress_code_lengths(cl) == lengths

    def test_roundtrip_long_zero_run(self):
        lengths = [5] + [0] * 50 + [3]
        cl = compress_code_lengths(lengths)
        assert decompress_code_lengths(cl) == lengths

    def test_roundtrip_repeated_nonzero(self):
        lengths = [4, 4, 4, 4, 4, 4, 4]
        cl = compress_code_lengths(lengths)
        assert decompress_code_lengths(cl) == lengths

    def test_random_roundtrip(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            lengths = rng.integers(0, 16, size=50).tolist()
            cl = compress_code_lengths(lengths)
            assert decompress_code_lengths(cl) == lengths


# ==================================================================
# Simple / Normal code: round-trip через BitWriter/BitReader
# ==================================================================
class TestSimpleCode:
    def test_one_symbol(self):
        w = BitWriter()
        write_simple_code(w, [5])
        r = BitReader(w.finish())
        assert r.read_bit() == 1  # маркер
        assert read_simple_code(r) == [5]

    def test_two_symbols(self):
        w = BitWriter()
        write_simple_code(w, [3, 200])
        r = BitReader(w.finish())
        assert r.read_bit() == 1
        assert read_simple_code(r) == [3, 200]

    def test_first_small(self):
        """Первый символ 0 или 1 кодируется 1 битом."""
        w = BitWriter()
        write_simple_code(w, [1, 100])
        r = BitReader(w.finish())
        assert r.read_bit() == 1
        assert read_simple_code(r) == [1, 100]


class TestNormalCode:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        # 20 символов, случайные длины 0..5
        lengths = rng.integers(0, 6, size=20).tolist()
        if max(lengths) == 0:
            lengths[0] = 1  # хотя бы один ненулевой

        w = BitWriter()
        write_normal_code(w, lengths)
        data = w.finish()

        r = BitReader(data)
        assert r.read_bit() == 0  # маркер normal
        got = read_normal_code(r, len(lengths))
        assert got == lengths


class TestHuffmanCode:
    def test_simple_roundtrip(self):
        lengths = [1, 0, 0, 1]
        w = BitWriter()
        write_huffman_code(w, lengths)
        r = BitReader(w.finish())
        assert read_huffman_code(r, 4) == lengths

    def test_normal_roundtrip(self):
        lengths = [3, 3, 2, 2, 2]
        w = BitWriter()
        write_huffman_code(w, lengths)
        r = BitReader(w.finish())
        assert read_huffman_code(r, 5) == lengths

    def test_random_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            lengths = rng.integers(0, 6, size=30).tolist()
            if sum(1 for l in lengths if l > 0) == 0:
                lengths[0] = 1
            w = BitWriter()
            write_huffman_code(w, lengths)
            r = BitReader(w.finish())
            assert read_huffman_code(r, len(lengths)) == lengths