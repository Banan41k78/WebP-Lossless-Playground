"""
Битовый поток для VP8L.

Соглашение VP8L: биты пишутся и читаются LSB-first (младший бит первым),
внутри байта — от младшего к старшему.

Пример: записать значение 0b101 (5) в 3 бита → байт будет 0b101,
а не 0b10100000 (как было бы в MSB-first).
"""
from __future__ import annotations


class BitWriter:
    """Пишет биты в байтовый буфер LSB-first."""

    __slots__ = ("_buf", "_acc", "_nbits")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0        # аккумулятор битов
        self._nbits = 0      # сколько бит в аккумуляторе

    def write_bits(self, value: int, n: int) -> None:
        """Записать младшие n бит value."""
        if n < 0 or n > 32:
            raise ValueError(f"n должно быть в [0, 32], получено {n}")
        if n == 0:
            return
        value &= (1 << n) - 1
        self._acc |= value << self._nbits
        self._nbits += n
        while self._nbits >= 8:
            self._buf.append(self._acc & 0xFF)
            self._acc >>= 8
            self._nbits -= 8

    def write_bit(self, bit: int) -> None:
        self.write_bits(bit, 1)

    def finish(self) -> bytes:
        """Дописать нулями до границы байта и вернуть байты."""
        if self._nbits > 0:
            self._buf.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._buf)

    def __len__(self) -> int:
        """Текущий размер в битах (без добивки)."""
        return len(self._buf) * 8 + self._nbits


class BitReader:
    """Читает биты из байтового буфера LSB-first."""

    __slots__ = ("_buf", "_pos", "_acc", "_nbits")

    def __init__(self, data: bytes) -> None:
        self._buf = data
        self._pos = 0        # индекс следующего байта
        self._acc = 0
        self._nbits = 0

    def _fill(self, need: int = 32) -> None:
        while self._nbits < need and self._pos < len(self._buf):
            self._acc |= self._buf[self._pos] << self._nbits
            self._pos += 1
            self._nbits += 8

    def read_bits(self, n: int) -> int:
        """Прочитать n бит (LSB-first)."""
        if n < 0 or n > 32:
            raise ValueError(f"n должно быть в [0, 32], получено {n}")
        if n == 0:
            return 0
        if self._nbits < n:
            self._fill(n)
            if self._nbits < n:
                raise EOFError(
                    f"нужно {n} бит, осталось {self._nbits} "
                    f"(позиция {self._pos}/{len(self._buf)})"
                )
        value = self._acc & ((1 << n) - 1)
        self._acc >>= n
        self._nbits -= n
        return value

    def read_bit(self) -> int:
        return self.read_bits(1)

    def bits_left(self) -> int:
        """Сколько бит осталось до конца буфера."""
        return (len(self._buf) - self._pos) * 8 + self._nbits