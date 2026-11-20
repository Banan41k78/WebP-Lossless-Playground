"""
Оркестратор lossless WebP (VP8L).  [ВРЕМЕННАЯ ОТЛАДОЧНАЯ ВЕРСИЯ]

Исправления по сравнению с исходной версией:
  * distance_to_code() теперь строго обратна code_to_distance():
        for code, (dx, dy) in enumerate(DISTANCE_MAP, start=1):
            if dy * width + dx == dist: return code
        return dist + 120
    Старая версия использовала (dx == dist%width, dy == dist//width) и
    могла давать расстояние, не совпадающее с тем, что вернёт декодер
    (особенно опасно для узких изображений, width=1..4).
  * encode_lossless() получил флаг debug (по умолчанию True).
    Печатает токены, distance-коды, round-trip-проверку и self-decode.

Формат маркеров трансформаций в битстриме (VP8L spec):
    while read_bit() == 1:
        2 бита type (0=predictor, 1=cross-color,
                     2=subtract-green, 3=color-indexing)
        данные трансформации (для subtract-green данных нет)

После цикла трансформаций идут:
    1 бит use_color_cache; если 1 — 4 бита color_cache_bits;
    1 бит meta-huffman is_used (0 = одна группа);
    5 Хаффманов (green, red, blue, alpha, distance);
    закодированные данные.

Distance в битстриме кодируется через distance_code:
    - код 1..120 -> 2D-смещение (xi, yi) из DISTANCE_MAP,
      distance = yi * width + xi;
    - код > 120 -> distance = code - 120.
"""
from __future__ import annotations

import numpy as np

from .bitstream import BitReader, BitWriter
from .header import (
    VP8L_HEADER_SIZE,
    build_riff,
    build_vp8l_header,
    parse_riff,
    parse_vp8l_header,
)
from .huffman import (
    HuffmanDecoder,
    build_canonical_codes,
    build_code_lengths,
    read_huffman_code,
    write_huffman_code,
)
from .transforms import subtract_green_forward, subtract_green_inverse
from .lz77 import lz77_encode

# Порядок каналов в битстриме VP8L: G, R, B, A
_CHANNEL_SPECS = [
    (1, 280),   # Green: 256 литералов + 24 length-кода
    (0, 256),   # Red
    (2, 256),   # Blue
    (3, 256),   # Alpha
]
_DISTANCE_ALPHABET = 40
_TYPE_PREDICTOR = 0
_TYPE_CROSS_COLOR = 1
_TYPE_SUBTRACT_GREEN = 2
_TYPE_COLOR_INDEXING = 3

# Префиксные коды VP8L для длины (символы 256-279 в Green-алфавите)
LENGTH_CODES = [
    # symbols 0-3: lengths 1-4, 0 extra bits
    (1, 0), (2, 0), (3, 0), (4, 0),
    # symbols 4-5: lengths 5-8, 1 extra bit
    (5, 1), (7, 1),
    # symbols 6-7: lengths 9-16, 2 extra bits
    (9, 2), (13, 2),
    # symbols 8-9: lengths 17-32, 3 extra bits
    (17, 3), (25, 3),
    # symbols 10-11: lengths 33-64, 4 extra bits
    (33, 4), (49, 4),
    # symbols 12-13: lengths 65-128, 5 extra bits
    (65, 5), (97, 5),
    # symbols 14-15: lengths 129-256, 6 extra bits
    (129, 6), (193, 6),
    # symbols 16-17: lengths 257-512, 7 extra bits
    (257, 7), (385, 7),
    # symbols 18-19: lengths 513-1024, 8 extra bits
    (513, 8), (769, 8),
    # symbols 20-21: lengths 1025-2048, 9 extra bits
    (1025, 9), (1537, 9),
    # symbols 22-23: lengths 2049-4096, 10 extra bits
    (2049, 10), (3073, 10),
]

# Префиксные коды VP8L для distance_code (40 символов)
DISTANCE_CODES = [
    (1, 0), (2, 0), (3, 0), (4, 0),
    (5, 1), (7, 1),
    (9, 2), (13, 2),
    (17, 3), (25, 3),
    (33, 4), (49, 4),
    (65, 5), (97, 5),
    (129, 6), (193, 6),
    (257, 7), (385, 7),
    (513, 8), (769, 8),
    (1025, 9), (1537, 9),
    (2049, 10), (3073, 10),
    (4097, 11), (6145, 11),
    (8193, 12), (12289, 12),
    (16385, 13), (24577, 13),
    (32769, 14), (49153, 14),
    (65537, 15), (98305, 15),
    (131073, 16), (196609, 16),
    (262145, 17), (393217, 17),
    (524289, 18), (786433, 18),
]

# 2D-смещения для distance_code 1..120 (VP8L spec).
DISTANCE_MAP = [
    (0, 1), (1, 0), (1, 1), (-1, 1), (0, 2), (2, 0), (1, 2), (-1, 2),
    (2, 1), (-2, 1), (2, 2), (-2, 2), (0, 3), (3, 0), (1, 3), (-1, 3),
    (3, 1), (-3, 1), (2, 3), (-2, 3), (3, 2), (-3, 2), (0, 4), (4, 0),
    (1, 4), (-1, 4), (4, 1), (-4, 1), (3, 3), (-3, 3), (2, 4), (-2, 4),
    (4, 2), (-4, 2), (0, 5), (3, 4), (-3, 4), (4, 3), (-4, 3), (5, 0),
    (1, 5), (-1, 5), (5, 1), (-5, 1), (2, 5), (-2, 5), (5, 2), (-5, 2),
    (4, 4), (-4, 4), (3, 5), (-3, 5), (5, 3), (-5, 3), (0, 6), (6, 0),
    (1, 6), (-1, 6), (6, 1), (-6, 1), (2, 6), (-2, 6), (6, 2), (-6, 2),
    (4, 5), (-4, 5), (5, 4), (-5, 4), (3, 6), (-3, 6), (6, 3), (-6, 3),
    (0, 7), (7, 0), (1, 7), (-1, 7), (5, 5), (-5, 5), (7, 1), (-7, 1),
    (4, 6), (-4, 6), (6, 4), (-6, 4), (2, 7), (-2, 7), (7, 2), (-7, 2),
    (3, 7), (-3, 7), (7, 3), (-7, 3), (5, 6), (-5, 6), (6, 5), (-6, 5),
    (8, 0), (4, 7), (-4, 7), (7, 4), (-7, 4), (8, 1), (8, 2), (6, 6),
    (-6, 6), (8, 3), (5, 7), (-5, 7), (7, 5), (-7, 5), (8, 4), (6, 7),
    (-6, 7), (7, 6), (-7, 6), (8, 5), (7, 7), (-7, 7), (8, 6), (8, 7),
]
assert len(DISTANCE_MAP) == 120

MAX_LENGTH = 4096     
MAX_DISTANCE = 1048576


# ==================================================================
# Helpers (длины / distance-коды)
# ==================================================================
def length_to_code(length: int) -> tuple[int, int, int]:
    """length -> (code_index 0..23, extra_value, extra_bits)."""
    for i, (base, extra) in enumerate(LENGTH_CODES):
        if length < base + (1 << extra):
            return i, length - base, extra
    raise ValueError(f"length {length} вне диапазона (макс. {MAX_LENGTH})")


def distance_code_to_symbol(distance_code: int) -> tuple[int, int, int]:
    """distance_code -> (code_index 0..39, extra_value, extra_bits)."""
    for i, (base, extra) in enumerate(DISTANCE_CODES):
        if distance_code < base + (1 << extra):
            return i, distance_code - base, extra
    raise ValueError(
        f"distance_code {distance_code} вне диапазона (макс. {MAX_DISTANCE})"
    )


def code_to_length(code_idx: int, extra_value: int) -> int:
    base, _ = LENGTH_CODES[code_idx]
    return base + extra_value


def symbol_to_distance_code(code_idx: int, extra_value: int) -> int:
    base, _ = DISTANCE_CODES[code_idx]
    return base + extra_value


def distance_to_code(dist: int, width: int) -> int:
    """Сырое расстояние (в пикселях) -> distance_code для битстрима VP8L.

    ВАЖНО: выбираем код, для которого code_to_distance(code, width) == dist.
    Это строгая обратная операция. Старая версия использовала
    (dx == dist % width, dy == dist // width) и для узких изображений
    (width <= 4) могла не найти совпадения вовсе или вернуть код,
    который декодируется в другое расстояние.
    """
    for code, (dx, dy) in enumerate(DISTANCE_MAP, start=1):
        if dy * width + dx == dist:
            return code
    # Fallback: "прямой" distance_code > 120.
    return dist + 120


def code_to_distance(distance_code: int, width: int) -> int:
    """distance_code из битстрима -> сырое расстояние в пикселях."""
    if distance_code <= 120:
        xi, yi = DISTANCE_MAP[distance_code - 1]
        return yi * width + xi
    return distance_code - 120


# ==================================================================
# Debug helpers
# ==================================================================
def _fmt_word(word: int) -> str:
    g = (word      ) & 0xFF
    r = (word >> 8 ) & 0xFF
    b = (word >> 16) & 0xFF
    a = (word >> 24) & 0xFF
    return f"G={g:02X} R={r:02X} B={b:02X} A={a:02X}"


def _fmt_used(syms, limit: int = 20) -> str:
    used = sorted(set(syms))
    if len(used) <= limit:
        return f"{used}"
    return f"{used[:limit]}...(+{len(used) - limit} more)"


# ==================================================================
# Encoding
# ==================================================================
def encode_lossless(pixels: np.ndarray, *,
                    use_subtract_green: bool = True,
                    use_lz77: bool = True,
                    debug: bool = True) -> bytes:
    """(H, W, 4) uint8 RGBA -> полный .webp-файл (bytes).

    ВНИМАНИЕ: debug=True по умолчанию — это временная отладочная версия.
    Перед релизом поменяй на False.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"ожидался (H, W, 4), получено {pixels.shape}")

    h, w, _ = pixels.shape
    has_alpha = bool(np.any(pixels[..., 3] != 255))
    header = build_vp8l_header(w, h, has_alpha)

    if debug:
        print(f"\n[ENC] image {h}x{w}  sg={use_subtract_green}  "
              f"lz={use_lz77}  has_alpha={has_alpha}")

    work = subtract_green_forward(pixels) if use_subtract_green else pixels

    words = (
        work[..., 1].astype(np.uint32)
        | (work[..., 0].astype(np.uint32) << 8)
        | (work[..., 2].astype(np.uint32) << 16)
        | (work[..., 3].astype(np.uint32) << 24)
    ).flatten()

    if use_lz77:
        raw_tokens = lz77_encode(words, min_length=3,
                                 max_length=MAX_LENGTH,
                                 window_size=min(MAX_DISTANCE, len(words)))
    else:
        raw_tokens = [(0, int(wv)) for wv in words]

    if debug:
        n_lit = sum(1 for t in raw_tokens if t[0] == 0)
        n_copy = len(raw_tokens) - n_lit
        print(f"[ENC] tokens: {len(raw_tokens)} "
              f"(lit={n_lit}, copy={n_copy})")
        show = min(len(raw_tokens), 40)
        pos = 0
        for i, tok in enumerate(raw_tokens[:show]):
            if tok[0] == 0:
                print(f"[ENC]   [{i:02d}] @{pos:2d} LIT   {_fmt_word(tok[1])}")
                pos += 1
            else:
                _, d, l = tok
                dc = distance_to_code(d, w)
                back = code_to_distance(dc, w)
                st = "OK" if back == d else f"!!! back={back}"
                print(f"[ENC]   [{i:02d}] @{pos:2d} COPY  dist={d:2d} "
                      f"len={l:2d} -> code={dc:3d} -> back={back:2d} [{st}]")
                pos += l
        if len(raw_tokens) > show:
            print(f"[ENC]   ... ({len(raw_tokens) - show} more tokens)")

    # --------- Сборка символьных потоков ----------
    green_syms, red_syms, blue_syms, alpha_syms, dist_syms = [], [], [], [], []
    extra_stream = []

    for tok in raw_tokens:
        if tok[0] == 0:
            word = tok[1]
            green_syms.append(word & 0xFF)
            red_syms.append((word >> 8) & 0xFF)
            blue_syms.append((word >> 16) & 0xFF)
            alpha_syms.append((word >> 24) & 0xFF)
        else:
            _, dist, length = tok
            # Защита: distance не может быть 0 и не может превышать историю
            if dist <= 0:
                raise ValueError(f"битый токен: dist={dist}")
            l_idx, l_extra_val, l_extra_bits = length_to_code(length)
            dc = distance_to_code(dist, w)
            d_idx, d_extra_val, d_extra_bits = distance_code_to_symbol(dc)
            green_syms.append(256 + l_idx)
            dist_syms.append(d_idx)
            extra_stream.append((l_extra_val, l_extra_bits))
            extra_stream.append((d_extra_val, d_extra_bits))

    if debug:
        # Тщательная проверка distance round-trip
        bad = 0
        for tok in raw_tokens:
            if tok[0] == 1:
                _, d, _ = tok
                dc = distance_to_code(d, w)
                if code_to_distance(dc, w) != d:
                    bad += 1
        print(f"[ENC] used G symbols: {_fmt_used(green_syms)}")
        print(f"[ENC] used D symbols: {_fmt_used(dist_syms)}")
        if bad:
            print(f"[ENC] !!! {bad} DISTANCE ROUND-TRIP MISMATCHES !!!")
        else:
            print("[ENC] distance round-trip: OK (все токены)")

    def _freqs(syms, alphabet):
        if syms:
            c = np.bincount(syms, minlength=alphabet)
            return [int(x) for x in c]
        return [0] * alphabet

    green_freqs = _freqs(green_syms, 280)
    red_freqs = _freqs(red_syms, 256)
    blue_freqs = _freqs(blue_syms, 256)
    alpha_freqs = _freqs(alpha_syms, 256)
    dist_freqs = _freqs(dist_syms, 40)

    for freqs in (red_freqs, blue_freqs, alpha_freqs):
        if sum(freqs) == 0:
            freqs[0] = 1
    if sum(dist_freqs) == 0:
        dist_freqs[0] = 1

    green_lengths = build_code_lengths(green_freqs)
    red_lengths = build_code_lengths(red_freqs)
    blue_lengths = build_code_lengths(blue_freqs)
    alpha_lengths = build_code_lengths(alpha_freqs)
    dist_lengths = build_code_lengths(dist_freqs)

    green_codes = build_canonical_codes(green_lengths)
    red_codes = build_canonical_codes(red_lengths)
    blue_codes = build_canonical_codes(blue_lengths)
    alpha_codes = build_canonical_codes(alpha_lengths)
    dist_codes = build_canonical_codes(dist_lengths)

    writer = BitWriter()

    # --- маркеры трансформаций ---
    if use_subtract_green:
        writer.write_bits(1, 1)
        writer.write_bits(_TYPE_SUBTRACT_GREEN, 2)
    writer.write_bits(0, 1)

    # --- color cache (выключен) ---
    writer.write_bits(0, 1)

    # --- meta-huffman (одна группа) ---
    writer.write_bits(0, 1)

    # 5 Хаффманов
    write_huffman_code(writer, green_lengths, force_normal=True)
    write_huffman_code(writer, red_lengths, force_normal=True)
    write_huffman_code(writer, blue_lengths, force_normal=True)
    write_huffman_code(writer, alpha_lengths, force_normal=True)
    write_huffman_code(writer, dist_lengths, force_normal=True)

    # --- данные ---
    extra_iter = iter(extra_stream)
    for tok in raw_tokens:
        if tok[0] == 0:
            word = tok[1]
            for sym, codes in (
                (word & 0xFF, green_codes),
                ((word >> 8) & 0xFF, red_codes),
                ((word >> 16) & 0xFF, blue_codes),
                ((word >> 24) & 0xFF, alpha_codes),
            ):
                code, length_bits = codes[sym]
                writer.write_bits(code, length_bits)
        else:
            _, dist, length = tok
            l_idx, _, _ = length_to_code(length)
            dc = distance_to_code(dist, w)
            d_idx, _, _ = distance_code_to_symbol(dc)
            code, length_bits = green_codes[256 + l_idx]
            writer.write_bits(code, length_bits)
            l_val, l_bits = next(extra_iter)
            writer.write_bits(l_val, l_bits)
            code, length_bits = dist_codes[d_idx]
            writer.write_bits(code, length_bits)
            d_val, d_bits = next(extra_iter)
            writer.write_bits(d_val, d_bits)

    body = writer.finish()
    result = build_riff(header + body)

    if debug:
        print(f"[ENC] body: {len(body)} bytes, total VP8L stream: "
              f"{len(result) - 12} bytes")
        try:
            decoded = decode_lossless(result)
            if np.array_equal(decoded, pixels):
                print("[ENC] self-roundtrip: OK (наш декодер)")
            else:
                print("[ENC] self-roundtrip: PIXEL MISMATCH (наш декодер)")
        except Exception as e:  # noqa: BLE001
            print(f"[ENC] self-roundtrip: ERROR {type(e).__name__}: {e}")

    return result


# ==================================================================
# Decoding
# ==================================================================
def decode_lossless(data: bytes) -> np.ndarray:
    """Полный .webp-файл -> (H, W, 4) uint8 RGBA."""
    vp8l_data = parse_riff(data)
    w, h, _ = parse_vp8l_header(vp8l_data[:VP8L_HEADER_SIZE])

    reader = BitReader(vp8l_data[VP8L_HEADER_SIZE:])

    # --- маркеры трансформаций ---
    apply_subtract_green = False
    while reader.read_bit():
        t = reader.read_bits(2)
        if t == _TYPE_SUBTRACT_GREEN:
            apply_subtract_green = True
        elif t == _TYPE_PREDICTOR:
            raise NotImplementedError("predictor transform не поддерживается")
        elif t == _TYPE_CROSS_COLOR:
            raise NotImplementedError("cross-color transform не поддерживается")
        elif t == _TYPE_COLOR_INDEXING:
            raise NotImplementedError("color indexing transform не поддерживается")
        else:
            raise ValueError(f"неизвестный тип трансформации: {t}")

    # --- color cache ---
    if reader.read_bit():
        color_cache_bits = reader.read_bits(4)
        raise NotImplementedError(
            f"color cache не поддерживается (bits={color_cache_bits})"
        )

    # --- meta-huffman ---
    if reader.read_bit():
        raise NotImplementedError("meta-huffman не поддерживается")

    # 5 Хаффманов
    green_lengths = read_huffman_code(reader, 280)
    red_lengths = read_huffman_code(reader, 256)
    blue_lengths = read_huffman_code(reader, 256)
    alpha_lengths = read_huffman_code(reader, 256)
    dist_lengths = read_huffman_code(reader, 40)

    green_codes = build_canonical_codes(green_lengths)
    red_codes = build_canonical_codes(red_lengths)
    blue_codes = build_canonical_codes(blue_lengths)
    alpha_codes = build_canonical_codes(alpha_lengths)
    dist_codes = build_canonical_codes(dist_lengths)

    green_inv = {v[0]: (s, v[1]) for s, v in green_codes.items()}
    red_inv = {v[0]: (s, v[1]) for s, v in red_codes.items()}
    blue_inv = {v[0]: (s, v[1]) for s, v in blue_codes.items()}
    alpha_inv = {v[0]: (s, v[1]) for s, v in alpha_codes.items()}
    dist_inv = {v[0]: (s, v[1]) for s, v in dist_codes.items()}

    def _read_sym(inv, max_len):
        code = 0
        for length in range(1, max_len + 1):
            code |= reader.read_bit() << (length - 1)
            if code in inv:
                sym, sym_len = inv[code]
                if sym_len == length:
                    return sym
        raise ValueError("некорректный Хаффман-код")

    total_pixels = h * w
    decoded_words: list[int] = []

    while len(decoded_words) < total_pixels:
        g = _read_sym(green_inv, max(green_lengths))
        if g < 256:
            r = _read_sym(red_inv, max(red_lengths))
            b = _read_sym(blue_inv, max(blue_lengths))
            a = _read_sym(alpha_inv, max(alpha_lengths))
            decoded_words.append(g | (r << 8) | (b << 16) | (a << 24))
        else:
            l_idx = g - 256
            l_base, l_extra_bits = LENGTH_CODES[l_idx]
            l_extra = reader.read_bits(l_extra_bits) if l_extra_bits else 0
            length = l_base + l_extra

            d_idx = _read_sym(dist_inv, max(dist_lengths))
            d_base, d_extra_bits = DISTANCE_CODES[d_idx]
            d_extra = reader.read_bits(d_extra_bits) if d_extra_bits else 0
            dc = d_base + d_extra
            dist = code_to_distance(dc, w)

            start = len(decoded_words) - dist
            if start < 0:
                raise ValueError(
                    f"distance {dist} > history {len(decoded_words)}"
                )
            for i in range(length):
                decoded_words.append(decoded_words[start + i])

    decoded_words = decoded_words[:total_pixels]

    words = np.array(decoded_words, dtype=np.uint32).reshape(h, w)
    pixels = np.empty((h, w, 4), dtype=np.uint8)
    pixels[..., 0] = (words >> 8) & 0xFF
    pixels[..., 1] = (words >> 0) & 0xFF
    pixels[..., 2] = (words >> 16) & 0xFF
    pixels[..., 3] = (words >> 24) & 0xFF

    if apply_subtract_green:
        pixels = subtract_green_inverse(pixels)
    return pixels
