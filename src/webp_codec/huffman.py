"""
Канонический Хаффман для lossless WebP (VP8L).

Реализовано:
- построение оптимальных длин кодов из частот;
- канонические коды с переворотом битов для LSB-first записи;
- simple code (1–2 символа) и normal code (3+ символов);
- code-length codes (19-символьный алфавит: 0–15, 16, 17, 18).
"""
from __future__ import annotations

import heapq
from collections import Counter


MAX_CODE_LENGTH = 15
NUM_CODE_LENGTH_CODES = 19
CODE_LENGTH_ORDER = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


# Построение длин кодов из частот
def build_code_lengths(frequencies, max_length: int = MAX_CODE_LENGTH):
    """Оптимальные длины Хаффман-кодов из частот символов.

    Для символов с частотой 0 длина 0 (не используются).
    Если ненулевой символ один — длина 1.
    """
    n = len(frequencies)
    lengths = [0] * n
    nonzero = [(f, i) for i, f in enumerate(frequencies) if f > 0]

    if not nonzero:
        return lengths
    if len(nonzero) == 1:
        lengths[nonzero[0][1]] = 1
        return lengths

    counter = 0
    heap = []
    for f, i in nonzero:
        heapq.heappush(heap, (f, counter, i))
        counter += 1

    nodes = []
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        idx = len(nodes)
        nodes.append((n1, n2))
        heapq.heappush(heap, (f1 + f2, counter, ~idx))
        counter += 1

    root = heap[0][2]

    def traverse(node, depth):
        if isinstance(node, int) and node >= 0:
            lengths[node] = max(1, depth)
        else:
            left, right = nodes[~node]
            traverse(left, depth + 1)
            traverse(right, depth + 1)

    traverse(root, 0)

    if max(lengths) > max_length:
        used = [i for i, l in enumerate(lengths) if l > 0]
        bits = 1
        while (1 << bits) < len(used):
            bits += 1
        for i in used:
            lengths[i] = bits

    return lengths


# Канонические коды
def _reverse_bits(value: int, n: int) -> int:
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (value & 1)
        value >>= 1
    return rev


def build_canonical_codes(lengths):
    """Из длин кодов → {символ: (перевёрнутый_код, длина)} для LSB-first."""
    if not lengths:
        return {}
    max_len = max(lengths)
    if max_len == 0:
        return {}

    bl_count = [0] * (max_len + 1)
    for l in lengths:
        if l > 0:
            bl_count[l] += 1

    next_code = [0] * (max_len + 1)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    codes = {}
    for sym, l in enumerate(lengths):
        if l > 0:
            c = next_code[l]
            next_code[l] += 1
            codes[sym] = (_reverse_bits(c, l), l)
    return codes


def huffman_encode_symbol(writer, codes, symbol: int) -> None:
    if symbol not in codes:
        raise ValueError(f"символ {symbol} отсутствует в кодах")
    code, length = codes[symbol]
    writer.write_bits(code, length)


class HuffmanDecoder:
    """Декодер по каноническим длинам (посимвольное чтение бит за битом)."""

    __slots__ = ("_table", "_max_len")

    def __init__(self, lengths):
        self._max_len = max(lengths) if lengths else 0
        self._table = {}
        for sym, (code, length) in build_canonical_codes(lengths).items():
            self._table[(code, length)] = sym

    def decode(self, reader) -> int:
        code = 0
        for length in range(1, self._max_len + 1):
            code |= reader.read_bit() << (length - 1)
            if (code, length) in self._table:
                return self._table[(code, length)]
        raise ValueError("некорректный Хаффман-код")



# Code-length codes (19-символьный алфавит)
def compress_code_lengths(lengths):
    out = []
    i, n = 0, len(lengths)
    prev_nonzero = 8

    while i < n:
        v = lengths[i]
        j = i + 1
        while j < n and lengths[j] == v:
            j += 1
        run = j - i
        i = j

        if v == 0:
            while run >= 11:
                chunk = min(run, 138)
                out.append((18, chunk - 11))
                run -= chunk
            while run >= 3:
                chunk = min(run, 10)
                out.append((17, chunk - 3))
                run -= chunk
            while run > 0:
                out.append((0, 0))
                run -= 1
        else:
            if v != prev_nonzero:
                out.append((v, 0))
                prev_nonzero = v
                run -= 1
            while run >= 3:
                chunk = min(run, 6)
                out.append((16, chunk - 3))
                run -= chunk
            while run > 0:
                out.append((v, 0))
                run -= 1

    return out


def decompress_code_lengths(cl_symbols):
    out = []
    prev_nonzero = 8
    for sym, extra in cl_symbols:
        if sym < 16:
            out.append(sym)
            if sym > 0:
                prev_nonzero = sym
        elif sym == 16:
            out.extend([prev_nonzero] * (3 + extra))
        elif sym == 17:
            out.extend([0] * (3 + extra))
        elif sym == 18:
            out.extend([0] * (11 + extra))
        else:
            raise ValueError(f"недопустимый code-length символ: {sym}")
    return out



# Simple code (1–2 символа, только символы < 256)
def write_simple_code(writer, symbols):
    """Записать simple code для 1 или 2 уникальных символов (< 256)."""
    if len(symbols) not in (1, 2):
        raise ValueError("simple code: 1 или 2 символа")
    if any(s >= 256 for s in symbols):
        raise ValueError("simple code: символы должны быть < 256")
    writer.write_bits(1, 1)                     # маркер simple
    writer.write_bits(len(symbols) - 1, 1)
    first = symbols[0]
    if first < 2:
        writer.write_bits(0, 1)                 # is_first_8bits = 0
        writer.write_bits(first, 1)
    else:
        writer.write_bits(1, 1)                 # is_first_8bits = 1
        writer.write_bits(first, 8)
    if len(symbols) == 2:
        writer.write_bits(symbols[1], 8)


def read_simple_code(reader):
    num = reader.read_bit() + 1
    is_8 = reader.read_bit()
    first = reader.read_bits(8) if is_8 else reader.read_bit()
    symbols = [first]
    if num == 2:
        symbols.append(reader.read_bits(8))
    return symbols


# Normal code (3+ символов или символы >= 256)
def write_normal_code(writer, lengths):
    """Записать normal code: длины через code-length codes."""
    cl_symbols = compress_code_lengths(lengths)

    freq = Counter(sym for sym, _ in cl_symbols)
    cl_freq = [freq.get(i, 0) for i in range(NUM_CODE_LENGTH_CODES)]
    # Длины code-length кодов пишутся 3 битами → максимум 7.
    cl_lengths = build_code_lengths(cl_freq, max_length=7)
    cl_codes = build_canonical_codes(cl_lengths)

    writer.write_bits(0, 1)                     # маркер normal
    writer.write_bits(NUM_CODE_LENGTH_CODES - 4, 4)
    for i in range(NUM_CODE_LENGTH_CODES):
        writer.write_bits(cl_lengths[CODE_LENGTH_ORDER[i]], 3)

    # Обязательный бит: use_max_symbol=0 (max_symbol = размер алфавита).
    writer.write_bits(0, 1)

    for sym, extra in cl_symbols:
        code, length = cl_codes[sym]
        writer.write_bits(code, length)
        if sym == 16:
            writer.write_bits(extra, 2)
        elif sym == 17:
            writer.write_bits(extra, 3)
        elif sym == 18:
            writer.write_bits(extra, 7)


def read_normal_code(reader, num_symbols: int):
    """Прочитать normal code (после маркера) → массив длин."""
    num_cl = reader.read_bits(4) + 4
    cl_lengths = [0] * NUM_CODE_LENGTH_CODES
    for i in range(num_cl):
        cl_lengths[CODE_LENGTH_ORDER[i]] = reader.read_bits(3)

    use_max_symbol = reader.read_bit()
    if use_max_symbol:
        length_nbits = 2 + 2 * reader.read_bits(3)
        max_symbol = 2 + reader.read_bits(length_nbits)
        if max_symbol > num_symbols:
            raise ValueError(f"max_symbol {max_symbol} > алфавит {num_symbols}")
    else:
        max_symbol = num_symbols

    cl_decoder = HuffmanDecoder(cl_lengths)

    cl_symbols = []
    total = 0
    while total < max_symbol:
        sym = cl_decoder.decode(reader)
        if sym < 16:
            cl_symbols.append((sym, 0))
            total += 1
        elif sym == 16:
            extra = reader.read_bits(2)
            cl_symbols.append((16, extra))
            total += 3 + extra
        elif sym == 17:
            extra = reader.read_bits(3)
            cl_symbols.append((17, extra))
            total += 3 + extra
        elif sym == 18:
            extra = reader.read_bits(7)
            cl_symbols.append((18, extra))
            total += 11 + extra

    lengths = decompress_code_lengths(cl_symbols)
    if len(lengths) < num_symbols:
        lengths.extend([0] * (num_symbols - len(lengths)))
    return lengths[:num_symbols]


# Публичные точки входа: выбор simple/normal

def write_huffman_code(writer, lengths, *, force_normal=False):
    """Записать Хаффман-код: simple (≤2 и все символы < 256), иначе normal."""
    used = [s for s, l in enumerate(lengths) if l > 0]
    if len(used) == 0:
        raise ValueError("нечего кодировать: все длины 0")
    if not force_normal and len(used) <= 2 and all(s < 256 for s in used):
        write_simple_code(writer, used)
    else:
        write_normal_code(writer, lengths)


def read_huffman_code(reader, num_symbols: int):
    """Прочитать Хаффман-код → массив длин (длина num_symbols)."""
    is_simple = reader.read_bit()
    if is_simple:
        symbols = read_simple_code(reader)
        lengths = [0] * num_symbols
        for s in symbols:
            lengths[s] = 1
        return lengths
    return read_normal_code(reader, num_symbols)