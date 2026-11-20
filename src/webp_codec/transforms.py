"""
Трансформации изображения для lossless WebP (VP8L).

Каждая трансформация — обратимое преобразование массива (H, W, 4) uint8 RGBA.
Forward применяется при кодировании, inverse — при декодировании.
"""
import numpy as np

# Индексы каналов в numpy-массиве (H, W, 4), порядок Pillow: RGBA
R, G, B, A = 0, 1, 2, 3


# ==================================================================
# Subtract green
# ==================================================================
def subtract_green_forward(pixels: np.ndarray) -> np.ndarray:
    """(H, W, 4) uint8 → (H, W, 4) uint8, R-=G, B-=G mod 256."""
    out = pixels.copy()
    out[..., 0] = (out[..., 0] - out[..., 1]) & 0xFF
    out[..., 2] = (out[..., 2] - out[..., 1]) & 0xFF
    return out


def subtract_green_inverse(pixels: np.ndarray) -> np.ndarray:
    """Обратное преобразование: R += G, B += G mod 256."""
    out = pixels.copy()
    out[..., 0] = (out[..., 0] + out[..., 1]) & 0xFF
    out[..., 2] = (out[..., 2] + out[..., 1]) & 0xFF
    return out


# ==================================================================
# Cross-color
# ==================================================================
def cross_color_forward(pixels, green_to_red, green_to_blue, red_to_blue, block_size):
    out = pixels.copy()
    g = out[..., 1].astype(np.int32)

    # new_R = (R + ((g2r * G) >> 5)) & 0xFF
    new_r = (out[..., 0].astype(np.int32) + ((green_to_red * g) >> 5)) & 0xFF
    out[..., 0] = new_r

    # new_B = (B + ((g2b * G + r2b * new_R) >> 5)) & 0xFF
    new_b = (out[..., 2].astype(np.int32) + ((green_to_blue * g + red_to_blue * new_r) >> 5)) & 0xFF
    out[..., 2] = new_b
    return out


def cross_color_inverse(pixels, green_to_red, green_to_blue, red_to_blue, block_size):
    out = pixels.copy()
    g = out[..., 1].astype(np.int32)

    # B = (new_B - ((g2b * G + r2b * new_R) >> 5)) & 0xFF
    new_r = out[..., 0].astype(np.int32)
    new_b = (out[..., 2].astype(np.int32) - ((green_to_blue * g + red_to_blue * new_r) >> 5)) & 0xFF
    out[..., 2] = new_b

    # R = (new_R - ((g2r * G) >> 5)) & 0xFF
    out[..., 0] = (new_r - ((green_to_red * g) >> 5)) & 0xFF
    return out


# ==================================================================
# Color indexing
# ==================================================================
def color_indexing_forward(pixels) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 4) RGBA → (indices (H, W), palette (N, 4) RGBA)."""
    h, w, _ = pixels.shape
    flat = pixels.reshape(-1, 4).astype(np.uint32)
    words = flat[:, 0] | (flat[:, 1] << 8) | (flat[:, 2] << 16) | (flat[:, 3] << 24)
    uniq, inverse = np.unique(words, return_inverse=True)

    if len(uniq) > 256:
        raise ValueError(f"слишком много цветов для палитры: {len(uniq)} > 256")

    palette = np.empty((len(uniq), 4), dtype=np.uint8)
    palette[:, 0] = (uniq      ) & 0xFF
    palette[:, 1] = (uniq >> 8 ) & 0xFF
    palette[:, 2] = (uniq >> 16) & 0xFF
    palette[:, 3] = (uniq >> 24) & 0xFF

    indices = inverse.reshape(h, w).astype(np.uint8)
    return indices, palette


def color_indexing_inverse(indices, palette) -> np.ndarray:
    """(indices (H, W), palette (N, 4) RGBA) → (H, W, 4) RGBA."""
    return palette[indices]


# ==================================================================
# Predictor: 14 режимов
# ==================================================================
def _avg2(a, b):
    """Среднее без переполнения: (a + b) >> 1 по каждому каналу."""
    return tuple((int(a[i]) + int(b[i])) >> 1 for i in range(4))


def _clamp255(v: int) -> int:
    """Ограничить до [0, 255] (Clip255 из libwebp)."""
    return 0 if v < 0 else 255 if v > 255 else v


def _c_div2(x: int) -> int:
    """Деление на 2 с усечением к нулю (C-семантика, не floor)."""
    return x >> 1 if x >= 0 else -((-x) >> 1)


def _select(L, T, TL):
    """Select: вернуть L, если L ближе к TL, иначе T (сумма |.| по каналам)."""
    pa_minus_pb = 0
    for c in range(4):
        pbc = int(T[c]) - int(TL[c])
        pca = int(L[c]) - int(TL[c])
        pa_minus_pb += abs(pbc) - abs(pca)
    return L if pa_minus_pb <= 0 else T


def _clamp_add_subtract_full(L, T, TL):
    return tuple(_clamp255(int(L[c]) + int(T[c]) - int(TL[c])) for c in range(4))


def _clamp_add_subtract_half(L, T, TL):
    out = []
    for c in range(4):
        ave = (int(L[c]) + int(T[c])) >> 1
        out.append(_clamp255(ave + _c_div2(ave - int(TL[c]))))
    return tuple(out)


def predict_pixel(x: int, y: int, mode: int,
                  reconstructed: np.ndarray,
                  width: int, height: int) -> tuple[int, int, int, int]:
    """Предсказать цвет пикселя (x, y) по уже восстановленным соседям.

    Режимы (по спецификации VP8L):
        0  — black
        1  — L  (левый)
        2  — T  (верхний)
        3  — TR (верхний правый; на правой границе = TL)
        4  — TL (верхний левый)
        5  — Average2(Average2(L, TR), T)
        6  — Average2(L, TL)
        7  — Average2(L, T)
        8  — Average2(TL, T)
        9  — Average2(T, TR)
        10 — Average2(Average2(L, TL), Average2(T, TR))
        11 — Select(L, T, TL)
        12 — ClampAddSubtractFull(L, T, TL)
        13 — ClampAddSubtractHalf(L, T, TL)

    Отсутствующие соседи (за границей) заменяются на (0, 0, 0, 0).
    """
    ZERO = (0, 0, 0, 0)

    if mode == 0:                       # black
        return ZERO

    # --- L (левый): отсутствует на левой границе ---
    if x > 0:
        L = tuple(int(v) for v in reconstructed[y, x - 1])
    else:
        L = ZERO

    # --- T, TL, TR ---
    if y == 0:
        # верхняя граница: нет верхних соседей
        T = TL = TR = ZERO
    else:
        T = tuple(int(v) for v in reconstructed[y - 1, x])
        if x > 0:
            TL = tuple(int(v) for v in reconstructed[y - 1, x - 1])
        else:
            TL = ZERO
        if x < width - 1:
            TR = tuple(int(v) for v in reconstructed[y - 1, x + 1])
        else:
            # правая граница: по спеке VP8L TR = TL
            TR = TL

    # --- режимы 1..13 ---
    if mode == 1:
        return L
    if mode == 2:
        return T
    if mode == 3:
        return TR
    if mode == 4:
        return TL
    if mode == 5:
        return _avg2(_avg2(L, TR), T)
    if mode == 6:
        return _avg2(L, TL)
    if mode == 7:
        return _avg2(L, T)
    if mode == 8:
        return _avg2(TL, T)
    if mode == 9:
        return _avg2(T, TR)
    if mode == 10:
        return _avg2(_avg2(L, TL), _avg2(T, TR))
    if mode == 11:
        return _select(L, T, TL)
    if mode == 12:
        return _clamp_add_subtract_full(L, T, TL)
    if mode == 13:
        return _clamp_add_subtract_half(L, T, TL)

    raise ValueError(f"predict_pixel: неизвестный режим {mode}")


def _mode_for(modes, x: int, y: int, block_size: int | None) -> int:
    """Взять режим для пикселя (x, y): int → один на всё, array → поблочно."""
    if isinstance(modes, (int, np.integer)):
        return int(modes)
    if block_size is None or block_size <= 0:
        raise ValueError("block_size обязателен, если modes — массив")
    return int(modes[y // block_size, x // block_size])


def predictor_forward(pixels: np.ndarray, modes,
                     block_size: int | None = None) -> np.ndarray:
    """Вычесть предсказание по каждому блоку (растровый обход).

    Предсказание считается по ИСХОДНЫМ пикселям — это то, что увидит
    декодер после восстановления (forward и inverse симметричны).
    """
    h, w, _ = pixels.shape
    out = np.empty_like(pixels)
    for y in range(h):
        for x in range(w):
            mode = _mode_for(modes, x, y, block_size)
            pred = predict_pixel(x, y, mode, pixels, w, h)
            for c in range(4):
                out[y, x, c] = (int(pixels[y, x, c]) - pred[c]) & 0xFF
    return out


def predictor_inverse(differences: np.ndarray, modes,
                     block_size: int | None = None) -> np.ndarray:
    """Прибавить предсказание, восстанавливая пиксели в растровом порядке.

    Предсказание считается по УЖЕ ВОССТАНОВЛЕННЫМ пикселям (out) —
    ровно так же, как их видел энкодер на этапе forward.
    """
    h, w, _ = differences.shape
    out = np.zeros_like(differences)
    for y in range(h):
        for x in range(w):
            mode = _mode_for(modes, x, y, block_size)
            pred = predict_pixel(x, y, mode, out, w, h)
            for c in range(4):
                out[y, x, c] = (int(differences[y, x, c]) + pred[c]) & 0xFF
    return out