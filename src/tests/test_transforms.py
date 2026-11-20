"""Тесты для webp_codec.transforms."""
import numpy as np
import pytest

from webp_codec.transforms import (
    # subtract green
    subtract_green_forward,
    subtract_green_inverse,
    # cross-color
    cross_color_forward,
    cross_color_inverse,
    # color indexing
    color_indexing_forward,
    color_indexing_inverse,
    # predictor
    predict_pixel,
    predictor_forward,
    predictor_inverse,
)
# ==================================================================
# Predictor
# ==================================================================
class TestPredictor:
    def test_mode_black_is_identity(self):
        """Режим 0: pred = 0, поэтому residual = pixel."""
        px = np.array([[[10, 20, 30, 255], [40, 50, 60, 255]]], dtype=np.uint8)
        res = predictor_forward(px, modes=0)
        assert np.array_equal(res, px)
        assert np.array_equal(predictor_inverse(res, modes=0), px)

    def test_mode_L_roundtrip(self):
        rng = np.random.default_rng(42)
        px = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        res = predictor_forward(px, modes=1)
        assert np.array_equal(predictor_inverse(res, modes=1), px)

    def test_mode_T_roundtrip(self):
        rng = np.random.default_rng(7)
        px = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        res = predictor_forward(px, modes=2)
        assert np.array_equal(predictor_inverse(res, modes=2), px)

    def test_L_small_residuals_on_gradient(self):
        """На горизонтальном градиенте шаг 10 → все residual = 10."""
        h, w = 4, 8
        px = np.zeros((h, w, 4), dtype=np.uint8)
        for x in range(w):
            px[:, x] = [x * 10, x * 10, x * 10, 255]
        res = predictor_forward(px, modes=1)
        # после 0-го столбца: pred = левый = x*10, residual = 10
        assert (res[:, 1:, 0] == 10).all()
        # 0-й столбец: pred = 0, residual = pixel
        assert np.array_equal(res[:, 0, 0], px[:, 0, 0])

    def test_predict_pixel_L_uses_left(self):
        px = np.zeros((3, 3, 4), dtype=np.uint8)
        px[1, 0] = [100, 200, 50, 255]     # y=1, x=0
        # (x=1, y=1): левый сосед — px[1, 0] = [100, 200, 50, 255]
        assert predict_pixel(1, 1, 1, px, 3, 3) == (100, 200, 50, 255)
        # (x=2, y=1): левый сосед — px[1, 1] = нули
        assert predict_pixel(2, 1, 1, px, 3, 3) == (0, 0, 0, 0)
        # (x=0, y=1): левая граница → (0, 0, 0, 0)
        assert predict_pixel(0, 1, 1, px, 3, 3) == (0, 0, 0, 0)
        # (x=1, y=0): левый сосед — px[0, 0], туда положим цвет
        px[0, 0] = [10, 20, 30, 40]
        assert predict_pixel(1, 0, 1, px, 3, 3) == (10, 20, 30, 40)

    def test_predict_pixel_T_uses_top(self):
        px = np.zeros((3, 3, 4), dtype=np.uint8)
        px[0, 1] = [100, 200, 50, 255]
        assert predict_pixel(1, 1, 2, px, 3, 3) == (100, 200, 50, 255)
        # верхняя граница (y=0) → (0,0,0,0)
        assert predict_pixel(2, 0, 2, px, 3, 3) == (0, 0, 0, 0)

    def test_unknown_mode_raises(self):
        px = np.zeros((2, 2, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            predict_pixel(0, 0, mode=99, reconstructed=px, width=2, height=2)

    def test_block_modes_roundtrip(self):
        """modes как массив (поблочно) тоже должен работать."""
        rng = np.random.default_rng(0)
        px = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        modes = np.array([[1, 2], [2, 1]], dtype=np.uint8)   # 2x2 блоков по 4 пикселя
        res = predictor_forward(px, modes, block_size=4)
        assert np.array_equal(predictor_inverse(res, modes, block_size=4), px)