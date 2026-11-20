"""
Streamlit-приложение для WebP Lossless Codec.

Режим: полный цикл (encode → decode) с проверкой и статистикой.
Слева — панель управления с ползунками и переключателями.

Запуск:
    streamlit run app.py
"""
import io

import numpy as np
import streamlit as st
from PIL import Image

from webp_codec.lossless import encode_lossless, decode_lossless
from webp_codec.lz77 import lz77_encode, token_stats
from webp_codec.transforms import subtract_green_forward


# ---------------------------------------------------------------------------
# Конфигурация страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WebP Lossless Codec",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Утилиты (только для отображения, не для кодека)
# ---------------------------------------------------------------------------
def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def png_bytes(rgba: np.ndarray) -> bytes:
    """Сериализовать (H, W, 4) uint8 RGBA в PNG."""
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def pixels_to_words(pixels: np.ndarray, use_sg: bool) -> np.ndarray:
    """(H, W, 4) RGBA -> (N,) uint32 в порядке GRBA (как в битстриме VP8L)."""
    work = subtract_green_forward(pixels) if use_sg else pixels
    return (
        work[..., 1].astype(np.uint32)
        | (work[..., 0].astype(np.uint32) << 8)
        | (work[..., 2].astype(np.uint32) << 16)
        | (work[..., 3].astype(np.uint32) << 24)
    ).flatten()


def get_lz_stats(
    pixels: np.ndarray,
    use_sg: bool,
    use_lz: bool,
    min_length: int,
    max_length: int,
    window_size: int,
    chain_limit: int,
) -> dict:
    """Статистика по токенам LZ77 (или псевдо-статистика, если LZ77 off)."""
    n_px = pixels.shape[0] * pixels.shape[1]
    if not use_lz:
        return {
            "total_tokens": n_px,
            "literals": n_px,
            "references": 0,
            "literal_pixels": n_px,
            "reference_pixels": 0,
            "total_pixels": n_px,
            "compression_by_count": 1.0,
        }
    words = pixels_to_words(pixels, use_sg)
    tokens = lz77_encode(
        words,
        min_length=min_length,
        max_length=max_length,
        window_size=min(window_size, len(words)),
        chain_limit=chain_limit,
    )
    return token_stats(tokens)


# ---------------------------------------------------------------------------
# Боковая панель с ползунками
# ---------------------------------------------------------------------------
st.sidebar.title("Управление")

st.sidebar.subheader("Трансформации")
use_sg = st.sidebar.toggle(
    "Subtract green",
    value=True,
    help="R -= G, B -= G. Обычно улучшает сжатие на цветных фото.",
)
use_lz = st.sidebar.toggle(
    "LZ77",
    value=True,
    help="Поиск повторяющихся пикселей (backward-ссылки).",
)

st.sidebar.divider()
st.sidebar.subheader("Параметры LZ77")
st.sidebar.caption(
    "Влияют на предпросмотр статистики ниже. "
    "Реальный кодер (`encode_lossless`) использует фиксированные "
    "min_length=3, max_length=MAX_LENGTH, window_size=MAX_DISTANCE, "
    "chain_limit=64."
)

min_length = st.sidebar.slider(
    "min_length",
    min_value=3,
    max_value=16,
    value=3,
    step=1,
    disabled=not use_lz,
    help="Минимальная длина ссылки. Меньше — лучше сжатие, но медленнее.",
)
max_length = st.sidebar.slider(
    "max_length",
    min_value=16,
    max_value=512,
    value=128,
    step=16,
    disabled=not use_lz,
    help="Максимальная длина одной backward-ссылки.",
)
window_size = st.sidebar.slider(
    "window_size",
    min_value=1024,
    max_value=65536,
    value=16384,
    step=1024,
    disabled=not use_lz,
    help="Размер окна поиска (в пикселях).",
)
chain_limit = st.sidebar.slider(
    "chain_limit",
    min_value=16,
    max_value=256,
    value=64,
    step=16,
    disabled=not use_lz,
    help="Сколько кандидатов проверять в цепочке хеша.",
)

st.sidebar.divider()
st.sidebar.subheader("Отладка")
show_debug = st.sidebar.toggle(
    "Печатать токены в stdout",
    value=False,
    help="Вывод `encode_lossless(debug=True)` в консоль Streamlit.",
)

st.sidebar.divider()
st.sidebar.caption(
    "Lossless WebP (VP8L) — собственная реализация. "
    "Поддерживается: subtract green + LZ77. "
    "Без predictor / cross-color / color-indexing / color-cache."
)


# ---------------------------------------------------------------------------
# Заголовок
# ---------------------------------------------------------------------------
st.title("WebP Lossless Codec")
st.caption(
    "Собственная реализация lossless WebP на Python + NumPy — "
    "полный цикл encode → decode"
)


# ---------------------------------------------------------------------------
# Загрузка и обработка
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Загрузите изображение",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
)

if uploaded is None:
    st.info("⬆️ Загрузите изображение, чтобы начать")
    st.stop()

try:
    img = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"Не удалось открыть изображение: {e}")
    st.stop()

pixels = np.array(img, dtype=np.uint8)

with st.spinner("Кодирование..."):
    encoded = encode_lossless(
        pixels,
        use_subtract_green=use_sg,
        use_lz77=use_lz,
        debug=show_debug,
    )

with st.spinner("Декодирование..."):
    try:
        decoded = decode_lossless(encoded)
    except Exception as e:
        st.error(f"Ошибка декодирования: {type(e).__name__}: {e}")
        st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Оригинал")
    st.image(img, use_container_width=True)
with col2:
    st.subheader("После encode → decode")
    st.image(decoded, use_container_width=True)

if np.array_equal(decoded, pixels):
    st.success("Roundtrip: пиксели полностью совпадают")
else:
    diff = np.abs(decoded.astype(int) - pixels.astype(int))
    st.warning(
        f"Roundtrip: есть расхождения. "
        f"Макс. разница: {int(diff.max())}, сумма: {int(diff.sum())}"
    )

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ Скачать .webp",
        data=encoded,
        file_name=uploaded.name.rsplit(".", 1)[0] + ".webp",
        mime="image/webp",
    )
with dl2:
    st.download_button(
        "⬇️ Скачать PNG (roundtrip)",
        data=png_bytes(decoded),
        file_name=uploaded.name.rsplit(".", 1)[0] + "_decoded.png",
        mime="image/png",
    )


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📈 Статистика")

h, w = pixels.shape[:2]
raw_size = w * h * 4
enc_size = len(encoded)
ratio = raw_size / enc_size if enc_size else 0.0
saving = (1 - enc_size / raw_size) * 100 if raw_size else 0.0

st.markdown("#### Основные метрики")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Размер", f"{w} × {h}")
c2.metric("Raw RGBA", fmt_bytes(raw_size))
c3.metric("Закодировано", fmt_bytes(enc_size))
c4.metric("Сжатие", f"{ratio:.2f}×", delta=f"{saving:.1f}%")

st.markdown("#### LZ77-статистика")
st.caption(
    f"Параметры: min_length={min_length}, max_length={max_length}, "
    f"window_size={window_size}, chain_limit={chain_limit}"
)
lz = get_lz_stats(
    pixels, use_sg, use_lz,
    min_length=min_length,
    max_length=max_length,
    window_size=window_size,
    chain_limit=chain_limit,
)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Токенов", f"{lz['total_tokens']:,}")
d2.metric("Литералов", f"{lz['literals']:,}")
d3.metric("Ссылок", f"{lz['references']:,}")
d4.metric(
    "Пикселей (лит. + ссыл.)",
    f"{lz['literal_pixels']:,} + {lz['reference_pixels']:,}",
)
st.caption(
    f"Коэффициент по токенам: **{lz['compression_by_count']:.2f}×** · "
    f"всего пикселей: **{lz['total_pixels']:,}** · "
    f"subtract_green: **{'ON' if use_sg else 'OFF'}** · "
    f"LZ77: **{'ON' if use_lz else 'OFF'}**"
)

if use_sg:
    with st.expander("Промежуточное преобразование (subtract green)"):
        st.image(
            subtract_green_forward(pixels),
            use_container_width=True,
        )
        st.caption(
            "Это то, что реально кодируется Хаффманом "
            "перед обратной трансформацией при декодировании."
        )
