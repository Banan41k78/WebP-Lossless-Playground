"""Тесты для webp_codec.lz77."""
import numpy as np
import pytest

from webp_codec.lz77 import lz77_encode, lz77_decode, token_stats


# ==================================================================
# Базовые случаи
# ==================================================================
class TestLZ77Basics:
    def test_empty(self):
        assert lz77_encode(np.array([], dtype=np.uint32)) == []
        assert lz77_decode([]).tolist() == []

    def test_single_pixel(self):
        tokens = lz77_encode(np.array([42], dtype=np.uint32))
        assert tokens == [(0, 42)]
        assert lz77_decode(tokens).tolist() == [42]

    def test_no_repeats(self):
        """Все пиксели разные → все литералы."""
        pixels = np.array([10, 20, 30, 40, 50], dtype=np.uint32)
        tokens = lz77_encode(pixels)
        assert all(t[0] == 0 for t in tokens)
        assert len(tokens) == 5
        assert np.array_equal(lz77_decode(tokens), pixels)

    def test_two_same_no_match(self):
        """2 одинаковых пикселя < min_length=3 → два литерала."""
        pixels = np.array([7, 7], dtype=np.uint32)
        tokens = lz77_encode(pixels, min_length=3)
        assert tokens == [(0, 7), (0, 7)]

    def test_two_same_match_min2(self):
        """2 одинаковых при min_length=2: макс. длина ссылки = 1 < 2 → два литерала."""
        pixels = np.array([7, 7], dtype=np.uint32)
        tokens = lz77_encode(pixels, min_length=2)
        assert tokens == [(0, 7), (0, 7)]
        assert np.array_equal(lz77_decode(tokens), pixels)


# ==================================================================
# Паттерны
# ==================================================================
class TestLZ77Patterns:
    def test_repeated_pattern(self):
        """A B A B A B A B → A B + ссылка(d=2, len=6)."""
        pixels = np.array([10, 20, 10, 20, 10, 20, 10, 20], dtype=np.uint32)
        tokens = lz77_encode(pixels, min_length=3)
        assert tokens[0] == (0, 10)
        assert tokens[1] == (0, 20)
        assert tokens[2][0] == 1
        assert tokens[2][1] == 2   # distance
        assert tokens[2][2] == 6   # length
        assert np.array_equal(lz77_decode(tokens), pixels)

    def test_run_of_identical(self):
        """100 одинаковых → 1 литерал + 1 ссылка(d=1, len=99)."""
        pixels = np.full(100, 42, dtype=np.uint32)
        tokens = lz77_encode(pixels)
        assert len(tokens) == 2
        assert tokens[0] == (0, 42)
        assert tokens[1] == (1, 1, 99)
        assert np.array_equal(lz77_decode(tokens), pixels)

    def test_two_blocks(self):
        """A B C D A B C D → 4 литерала + ссылка(d=4, len=4)."""
        pixels = np.array([10, 20, 30, 40, 10, 20, 30, 40], dtype=np.uint32)
        tokens = lz77_encode(pixels, min_length=3)
        assert tokens[:4] == [(0, 10), (0, 20), (0, 30), (0, 40)]
        assert tokens[4] == (1, 4, 4)
        assert np.array_equal(lz77_decode(tokens), pixels)

    def test_long_run_capped(self):
        """10000 одинаковых, max_length=4096 → несколько ссылок, каждая ≤ 4096."""
        pixels = np.full(10000, 7, dtype=np.uint32)
        tokens = lz77_encode(pixels, max_length=4096)
        for t in tokens:
            if t[0] == 1:
                assert t[2] <= 4096
        assert np.array_equal(lz77_decode(tokens), pixels)


# ==================================================================
# Round-trip на случайных данных
# ==================================================================
class TestLZ77Roundtrip:
    def test_random_small_alphabet(self):
        """Маленький алфавит → много повторов → round-trip должен работать."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 5, size=500, dtype=np.uint32)
        tokens = lz77_encode(pixels)
        assert np.array_equal(lz77_decode(tokens), pixels)

    def test_random_large_alphabet(self):
        """Большой алфавит → почти все литералы, но round-trip тоже."""
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 2**32, size=200, dtype=np.uint32)
        tokens = lz77_encode(pixels)
        assert np.array_equal(lz77_decode(tokens), pixels)

    def test_realistic_pattern(self):
        """Имитация «неба»: плавный градиент с редкими повторами."""
        pixels = np.tile(np.arange(100, dtype=np.uint32), 50)
        tokens = lz77_encode(pixels)
        assert np.array_equal(lz77_decode(tokens), pixels)
        # должно сжаться
        assert len(tokens) < len(pixels)


# ==================================================================
# Ошибки
# ==================================================================
class TestLZ77Errors:
    def test_invalid_distance(self):
        with pytest.raises(ValueError):
            lz77_decode([(1, 5, 3)])   # нет истории на 5 назад

    def test_unknown_token_type(self):
        with pytest.raises(ValueError):
            lz77_decode([(7, 1, 2)])


# ==================================================================
# Статистика
# ==================================================================
class TestTokenStats:
    def test_empty(self):
        s = token_stats([])
        assert s["total_tokens"] == 0
        assert s["total_pixels"] == 0

    def test_mixed(self):
        tokens = [(0, 10), (0, 20), (1, 2, 6)]
        s = token_stats(tokens)
        assert s["literals"] == 2
        assert s["references"] == 1
        assert s["literal_pixels"] == 2
        assert s["reference_pixels"] == 6
        assert s["total_pixels"] == 8
        assert s["compression_by_count"] == pytest.approx(8 / 3)