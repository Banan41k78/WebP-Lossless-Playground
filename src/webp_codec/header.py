"""Заголовки и RIFF-контейнер для lossless WebP (VP8L)."""

from .bitstream import BitReader, BitWriter

# Константы для дальнейшего использования
VP8L_SIGNATURE = 0x2F
VP8L_VERSION = 0
VP8L_HEADER_SIZE = 5   # байт


def build_vp8l_header(width: int, height: int, has_alpha: bool) -> bytes:
    """Собрать 5-байтный VP8L-заголовок (40 бит, LSB-first)."""
    w = BitWriter()
    w.write_bits(VP8L_SIGNATURE, 8)
    w.write_bits(width  - 1, 14)
    w.write_bits(height - 1, 14)
    w.write_bits(1 if has_alpha else 0, 1)
    w.write_bits(VP8L_VERSION, 3)
    return w.finish()


def parse_vp8l_header(data: bytes) -> tuple[int, int, bool]:
    """Прочитать (width, height, has_alpha) из 5 байт заголовка."""
    r = BitReader(data[:VP8L_HEADER_SIZE])
    if r.read_bits(8) != VP8L_SIGNATURE:
        raise ValueError("неверная сигнатура VP8L")
    width  = r.read_bits(14) + 1
    height = r.read_bits(14) + 1
    alpha  = bool(r.read_bits(1))
    version = r.read_bits(3)
    if version != VP8L_VERSION:
        raise ValueError(f"неподдерживаемая версия VP8L: {version}")
    return width, height, alpha

def build_riff(vp8l_data: bytes) -> bytes:
    """Обернуть VP8L-данные в RIFF-чанк."""
    chunk = b"VP8L" + len(vp8l_data).to_bytes(4, "little") + vp8l_data
    if len(vp8l_data) % 2:            # RIFF требует чётной длины чанка
        chunk += b"\x00"
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def parse_riff(data: bytes) -> bytes:
    """Проверить RIFF-обёртку и вернуть содержимое VP8L-чанка."""
    if data[:4] != b"RIFF":
        raise ValueError("не RIFF-файл")
    if data[8:12] != b"WEBP":
        raise ValueError("не WEBP-контейнер")
    if data[12:16] != b"VP8L":
        raise ValueError(f"не VP8L-чанк: {data[12:16]!r}")
    chunk_size = int.from_bytes(data[16:20], "little")
    return data[20 : 20 + chunk_size]


def build_webp(width: int, height: int, has_alpha: bool,
               vp8l_payload: bytes) -> bytes:
    """Собрать полный .webp: заголовок + payload, обёрнутые в RIFF."""
    header = build_vp8l_header(width, height, has_alpha)
    return build_riff(header + vp8l_payload)


def parse_webp(data: bytes) -> tuple[int, int, bool, bytes]:
    """Разобрать .webp → (width, height, has_alpha, vp8l_payload)."""
    vp8l = parse_riff(data)
    width, height, alpha = parse_vp8l_header(vp8l[:VP8L_HEADER_SIZE])
    return width, height, alpha, vp8l[VP8L_HEADER_SIZE:]