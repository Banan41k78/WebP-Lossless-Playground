"""
webp_codec — реализация lossless WebP (VP8L) с нуля.

Конвейер кодирования (порядок шагов как в спецификации VP8L):

    пиксели → [трансформации] → [color cache] → [LZ77] → [префиксное кодирование] → битстрим
                    ↑                                    ↑
              transforms.py                         huffman.py

Конвейер декодирования — обратный порядок ↑.

Модули:
-------------------------------------------------------------------------------
bitstream.py    BitWriter / BitReader (LSB-first, как в VP8L).
|               Фундамент: без него не работает ничего остального.

huffman.py      Канонический Хаффман + code-length codes (19 символов).
|               Префиксное кодирование литералов, длин и distance-кодов.

lz77.py         Поиск backward-ссылок (distance + length).
|               В VP8L distance кодируется через префиксные distance-коды.

transforms.py   4 трансформации изображения:
|               1) subtract green    (R -= G, B -= G)
|               2) color transform   (линейное преобразование по коэффициентам)
|               3) color indexing    (палитра ≤ 256 цветов + индексы)
|               4) predictor         (14 режимов, кодируется остаток)

color_cache.py  Хеш-кэш последних (1 << cache_bits) цветов.
|               Устраняет локальные повторы без LZ77-ссылок.

header.py       RIFF-контейнер + VP8L chunk + 5-байтный заголовок
|               (сигнатура 0x2F, ширина−1, высота−1, alpha-флаг, version).

lossless.py     Публичный API: encode_lossless() / decode_lossless().
|               Собирает всё вместе в правильном порядке.
"""
from .lossless import decode_lossless, encode_lossless

__all__ = ["encode_lossless", "decode_lossless"]