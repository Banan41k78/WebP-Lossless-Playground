"""Тесты для webp_codec.color_cache."""
import pytest

from webp_codec.color_cache import ColorCache, HASH_MUL


# ==================================================================
# Базовые операции
# ==================================================================
class TestColorCacheBasics:
    def test_bits_zero_disabled(self):
        """bits=0 → кэш отключён, lookup всегда None."""
        c = ColorCache(0)
        assert c.size == 1
        c.insert(0xDEADBEEF)
        assert c.lookup(0xDEADBEEF) is None

    def test_insert_lookup_hit(self):
        c = ColorCache(6)
        pixel = 0xFF001122
        c.insert(pixel)
        assert c.lookup(pixel) is not None
        assert c.lookup(pixel) == c.hash(pixel)

    def test_lookup_miss(self):
        c = ColorCache(6)
        c.insert(0x11111111)
        assert c.lookup(0x22222222) is None

    def test_get_after_insert(self):
        c = ColorCache(6)
        pixel = 0xABCDEF01
        c.insert(pixel)
        idx = c.lookup(pixel)
        assert c.get(idx) == pixel

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            ColorCache(-1)
        with pytest.raises(ValueError):
            ColorCache(11)


# ==================================================================
# Перезапись (коллизии)
# ==================================================================
class TestColorCacheOverwrite:
    def test_overwrite_same_slot(self):
        """Два разных цвета с одинаковым хешем → в кэше последний."""
        c = ColorCache(1)   # size=2, хешей мало — легко поймать коллизию
        found = False
        for a in range(0, 0x100000):
            ha = c.hash(a)
            for b in range(a + 1, a + 100):
                if c.hash(b) == ha and a != b:
                    found = True
                    break
            if found:
                break
        assert found, "не нашли коллизию — увеличьте диапазон"

        c.insert(a)
        assert c.lookup(a) == ha
        c.insert(b)
        assert c.lookup(b) == ha
        # a вытеснен
        assert c.lookup(a) is None or c.hash(a) != ha

    def test_lru_style(self):
        """Многократные вставки одного цвета — он всегда в кэше."""
        c = ColorCache(8)
        pixel = 0xCAFEBABE
        for _ in range(100):
            c.insert(pixel)
        assert c.lookup(pixel) is not None


# ==================================================================
# Хеш-функция
# ==================================================================
class TestColorCacheHash:
    def test_known_hash_constant(self):
        """Константа хеша должна быть 0x1e35a7bd."""
        assert HASH_MUL == 0x1E35A7BD

    def test_hash_range(self):
        """Хеш всегда в [0, 2^bits)."""
        for bits in (1, 4, 6, 8, 10):
            c = ColorCache(bits)
            for pixel in (0, 1, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF):
                h = c.hash(pixel)
                assert 0 <= h < c.size

    def test_hash_deterministic(self):
        c = ColorCache(6)
        assert c.hash(0x12345678) == c.hash(0x12345678)


# ==================================================================
# Round-trip через encode/decode
# ==================================================================
class TestColorCacheRoundtrip:
    def test_full_roundtrip(self):
        """
        Имитация encode/decode над последовательностью пикселей:
        каждый пиксель либо кодируется индексом кэша, либо литералом
        (после чего попадает в кэш).
        """
        pixels = [10, 20, 10, 30, 20, 10, 20, 30, 10, 20, 30, 10]
        cache = ColorCache(4)

        # --- encode ---
        tokens = []
        for p in pixels:
            idx = cache.lookup(p)
            if idx is not None:
                tokens.append(("cache", idx))
            else:
                tokens.append(("lit", p))
                cache.insert(p)

        # --- decode ---
        cache_d = ColorCache(4)
        decoded = []
        for kind, val in tokens:
            if kind == "cache":
                decoded.append(cache_d.get(val))
            else:
                decoded.append(val)
                cache_d.insert(val)

        assert decoded == pixels

    def test_all_unique_no_cache_hits(self):
        """Если все пиксели разные — все литералы, кэш ничего не даёт."""
        pixels = list(range(50))
        cache = ColorCache(4)
        hits = 0
        for p in pixels:
            if cache.lookup(p) is not None:
                hits += 1
            cache.insert(p)
        # с 16 ячейками и 50 уникальными цветами возможны редкие коллизии,
        # но их мало — проверим, что большинство промахов
        assert hits < 5