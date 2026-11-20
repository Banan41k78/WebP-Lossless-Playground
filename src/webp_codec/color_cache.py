"""
Color cache для lossless WebP (VP8L). !!! ОПЦИОНАЛЬНО

Хеш-кэш последних 2^bits уникальных пикселей (uint32 в порядке GRBA).
Если пиксель уже встречался недавно — кодируется индексом кэша
(0 .. 2^bits-1), а не полным RGBA. Индекс идёт в Green-поток
как символ 256+i (i — индекс кэша).

Хеш-функция — из спеки VP8L:
    h = (pixel * 0x1e35a7bd) >> (32 - bits)

Замена: при вставке нового пикселя в занятую ячейку старое значение
просто перезаписывается (это нормально — кэш «недавних» цветов).
"""
from __future__ import annotations

# Константа из спеки. 
# !!! менять нельзя
HASH_MUL = 0x1E35A7BD


class ColorCache:
    """Хеш-кэш последних 2^bits пикселей (uint32)."""

    __slots__ = ("bits", "size", "_cache", "_shift")

    def __init__(self, bits: int) -> None:
        if not 0 <= bits <= 10:
            raise ValueError(f"cache_bits должен быть 0..10, получено {bits}")
        self.bits = bits
        self.size = 1 << bits
        self._cache = [0] * self.size
        self._shift = 32 - bits if bits > 0 else 32

    def hash(self, pixel: int) -> int:
        """Хеш пикселя → индекс в кэше."""
        return ((int(pixel) * HASH_MUL) & 0xFFFFFFFF) >> self._shift

    def insert(self, pixel: int) -> None:
        """Добавить пиксель в кэш (перезаписывает ячейку)."""
        if self.bits == 0:
            return
        self._cache[self.hash(pixel)] = int(pixel)

    def lookup(self, pixel: int) -> int | None:
        """Вернуть индекс в кэше, если пиксель там есть, иначе None."""
        if self.bits == 0:
            return None
        h = self.hash(pixel)
        return h if self._cache[h] == int(pixel) else None

    def get(self, index: int) -> int:
        """Вернуть пиксель по индексу кэша (для декодера)."""
        return self._cache[index]