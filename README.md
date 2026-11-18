# WebP Interface

Interactive Streamlit app for exploring a from-scratch implementation of the
**lossless WebP (VP8L)** codec. Term project for *Information Theory and Coding*.

The app lets you upload an image and walk through the full lossless compression
pipeline step by step: bit-level I/O, RIFF container, transforms (subtract green,
cross-color, color indexing, predictor), LZ77 and canonical Huffman. Every stage
is inspectable, and the produced bitstream is verified against `libwebp`
(via Pillow).

## Install

```bash
pip install -r requirements.txt
```
## Run

```bash
streamlit run src/app.py
```

## Project structure
```
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── app.py                # Streamlit entry point
    ├── webp_codec/
    │   ├── __init__.py       # package overview
    │   ├── bitstream.py      # BitWriter / BitReader, LSB-first
    │   ├── header.py         # RIFF wrapper + 40-bit VP8L header
    │   ├── transforms.py     # subtract green, cross-color, color indexing, predictor
    │   ├── huffman.py        # canonical Huffman + code-length codes
    │   ├── lz77.py           # backward references + distance codes
    │   ├── color_cache.py    # hash cache of recent colors
    │   └── lossless.py       # encode/decode orchestration
    └── tests/ # Tests folder
```
## References
- [VP8L bitstream specification](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)
- [RFC 9649 — WebP Image Format](https://www.rfc-editor.org/rfc/rfc9649)
