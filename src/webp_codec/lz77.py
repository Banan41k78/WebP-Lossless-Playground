"""
LZ77 для lossless WebP (VP8L).

Работает над ПЛОСКИМ массивом пикселей (N,) uint32 (в порядке GRBA,
как в битстриме VP8L). Возвращает поток токенов:

    (0, pixel)              — литерал
    (1, distance, length)   — обратная ссылка

distance — в пикселях (не байтах), 1 <= distance <= window_size.
length   — в пикселях, min_length <= length <= max_length.

Реализация — классический LZ77 с hash-chain (как в zlib/deflate):
для каждого 32-битного пикселя считается хеш, в бакете хранится
цепочка последних позиций с этим хешем. На каждом шаге проверяются
до chain_limit кандидатов.

VP8L-специфика (префиксные коды для length и distance) — НЕ здесь.
Это задача lossless.py: lz77.py хранит сырые distance/length.
"""
from __future__ import annotations

import numpy as np

# Параметры по умолчанию (можно переопределить через аргументы)
DEFAULT_MIN_LENGTH = 3
DEFAULT_MAX_LENGTH = 4096
DEFAULT_WINDOW = 1 << 16          # 65536 пикселей
DEFAULT_CHAIN_LIMIT = 64

# Хеш-функция (из zlib: 0x1e35a7bd), усечённая до 16 бит
HASH_BITS = 16
HASH_SIZE = 1 << HASH_BITS
HASH_MUL = 0x1E35A7BD
HASH_MASK = HASH_SIZE - 1


# Encoding
def lz77_encode(pixels: np.ndarray,
                min_length: int = DEFAULT_MIN_LENGTH,
                max_length: int = DEFAULT_MAX_LENGTH,
                window_size: int = DEFAULT_WINDOW,
                chain_limit: int = DEFAULT_CHAIN_LIMIT) -> list[tuple]:
    """LZ77 over pixels (N,) uint32.

    Возвращает список токенов:
        (0, pixel)              — литерал
        (1, distance, length)   — обратная ссылка
    """
    n = len(pixels)
    if n == 0:
        return []

    head = [-1] * HASH_SIZE          # head[h] — последняя позиция с хешем h
    prev = [-1] * n                  # prev[i] — предыдущая позиция с тем же хешем

    def phash(idx: int) -> int:
        return ((int(pixels[idx]) * HASH_MUL) >> (32 - HASH_BITS)) & HASH_MASK

    def insert(idx: int) -> None:
        h = phash(idx)
        prev[idx] = head[h]
        head[h] = idx

    def find(pos: int) -> tuple[int, int]:
        """Найти лучшее совпадение для pixels[pos:]. Вернуть (dist, length)."""
        if pos + min_length > n:
            return 0, 0
        candidate = head[phash(pos)]
        best_len = 0
        best_dist = 0
        max_possible = min(max_length, n - pos)
        chain = 0
        while candidate >= 0 and chain < chain_limit:
            dist = pos - candidate
            if dist > window_size:
                break
            length = 0
            while (length < max_possible
                   and pixels[candidate + length] == pixels[pos + length]):
                length += 1
            if length > best_len:
                best_len = length
                best_dist = dist
                if best_len >= max_possible:
                    break
            candidate = prev[candidate]
            chain += 1
        return best_dist, best_len

    tokens: list[tuple] = []
    pos = 0
    while pos < n:
        dist, length = find(pos)
        if length >= min_length:
            tokens.append((1, dist, length))
            for i in range(pos, pos + length):
                insert(i)
            pos += length
        else:
            tokens.append((0, int(pixels[pos])))
            insert(pos)
            pos += 1
    return tokens


# Decoding
def lz77_decode(tokens: list[tuple]) -> np.ndarray:
    """Восстановить (N,) uint32 из токенов."""
    out: list[int] = []
    for t in tokens:
        if t[0] == 0:
            out.append(int(t[1]))
        elif t[0] == 1:
            _, dist, length = t
            start = len(out) - dist
            if start < 0:
                raise ValueError(
                    f"distance={dist} больше доступной истории ({len(out)})"
                )
            for i in range(length):
                out.append(out[start + i])
        else:
            raise ValueError(f"неизвестный тип токена: {t[0]!r}")
    return np.array(out, dtype=np.uint32)


# Статистика
def token_stats(tokens: list[tuple]) -> dict:
    """Сводка по потоку токенов."""
    n_lit = sum(1 for t in tokens if t[0] == 0)
    n_ref = sum(1 for t in tokens if t[0] == 1)
    lit_pixels = n_lit
    ref_pixels = sum(t[2] for t in tokens if t[0] == 1)
    return {
        "total_tokens": len(tokens),
        "literals": n_lit,
        "references": n_ref,
        "literal_pixels": lit_pixels,
        "reference_pixels": ref_pixels,
        "total_pixels": lit_pixels + ref_pixels,
        "compression_by_count": (
            (lit_pixels + ref_pixels) / len(tokens) if tokens else 1.0
        ),
    }