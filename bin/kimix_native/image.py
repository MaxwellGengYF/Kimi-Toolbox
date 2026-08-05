"""kimix_native.image — image header sniffing kernels.

Native implementations live in ``runtime_py.image`` (compiled kernels, GIL
released). The pure-Python ``_compat`` functions below mirror
`kimi_cli/utils/image_compress.py` exactly.
"""

from __future__ import annotations

import math
from typing import Literal

from . import _native, use_native


def _compat_format_byte_size(n: int) -> str:
    """Mirror of image_compress.py::format_byte_size."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{math.floor(n / 1024 + 0.5)} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _compat_read_exif_orientation(data: bytes, start: int, end: int) -> int | None:
    """Mirror of image_compress.py::_read_exif_orientation."""
    bounded_end = min(end, len(data))
    if start + 6 > bounded_end or data[start : start + 6] != b"Exif\x00\x00":
        return None
    tiff = start + 6
    if tiff + 8 > bounded_end:
        return None

    byte_order = data[tiff : tiff + 2]
    if byte_order == b"II":
        little_endian = True
    elif byte_order == b"MM":
        little_endian = False
    else:
        return None
    order: Literal["little", "big"] = "little" if little_endian else "big"

    def u16(offset: int) -> int:
        return int.from_bytes(data[offset : offset + 2], order)

    def u32(offset: int) -> int:
        return int.from_bytes(data[offset : offset + 4], order)

    if u16(tiff + 2) != 42:
        return None

    ifd = tiff + u32(tiff + 4)
    if ifd + 2 > bounded_end:
        return None

    entry_count = u16(ifd)
    for i in range(entry_count):
        entry = ifd + 2 + i * 12
        if entry + 12 > bounded_end:
            return None
        if u16(entry) == 0x0112:
            value = u16(entry + 8)
            return value if 1 <= value <= 8 else None
    return None


def _compat_sniff_dimensions(data: bytes) -> tuple[int, int, bool] | None:
    """Mirror of image_compress.py::sniff_image_dimensions."""
    # PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
            False,
        )

    # GIF
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return (
            int.from_bytes(data[6:8], "little"),
            int.from_bytes(data[8:10], "little"),
            False,
        )

    # BMP
    if data.startswith(b"BM") and len(data) >= 26:
        return (
            int.from_bytes(data[18:22], "little", signed=True),
            abs(int.from_bytes(data[22:26], "little", signed=True)),
            False,
        )

    # WebP
    if data.startswith(b"RIFF") and len(data) >= 30:
        four_cc = data[12:16]
        if four_cc == b"VP8 ":
            return (
                int.from_bytes(data[26:28], "little") & 0x3FFF,
                int.from_bytes(data[28:30], "little") & 0x3FFF,
                False,
            )
        if four_cc == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return (
                (bits & 0x3FFF) + 1,
                ((bits >> 14) & 0x3FFF) + 1,
                False,
            )
        if four_cc == b"VP8X":
            width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
            height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
            return (width, height, False)

    # JPEG
    if data.startswith(b"\xff\xd8"):
        orientation: int | None = None
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height = int.from_bytes(data[offset + 5 : offset + 7], "big")
                width = int.from_bytes(data[offset + 7 : offset + 9], "big")
                if orientation is not None and orientation >= 5:
                    return (height, width, True)
                return (width, height, False)
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                offset += 2
                continue
            segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
            if segment_length < 2:
                break
            if marker == 0xE1 and orientation is None:
                orientation = _compat_read_exif_orientation(
                    data, offset + 4, offset + 2 + segment_length
                )
            offset += 2 + segment_length

    return None


def _compat_is_animated_webp(data: bytes) -> bool:
    """Mirror of image_compress.py::_is_animated_webp."""
    return (
        len(data) >= 21
        and data[0:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and data[12:16] == b"VP8X"
        and (data[20] & 0x02) != 0
    )


def _compat_read_exif_orientation_from_jpeg(data: bytes) -> int | None:
    """Scan a JPEG stream for the first APP1 EXIF orientation tag."""
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 4 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if segment_length < 2:
            break
        if marker == 0xE1:
            orientation = _compat_read_exif_orientation(
                data, offset + 4, offset + 2 + segment_length
            )
            if orientation is not None:
                return orientation
        offset += 2 + segment_length
    return None


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


def sniff_dimensions(data: bytes) -> tuple[int, int, bool] | None:
    if not use_native("IMAGE") or _native is None:
        return _compat_sniff_dimensions(data)
    return _native.image.sniff_dimensions(data)


def read_exif_orientation(data: bytes) -> int | None:
    if not use_native("IMAGE") or _native is None:
        return _compat_read_exif_orientation_from_jpeg(data)
    return _native.image.read_exif_orientation(data)


def is_animated_webp(data: bytes) -> bool:
    if not use_native("IMAGE") or _native is None:
        return _compat_is_animated_webp(data)
    return _native.image.is_animated_webp(data)


def format_byte_size(n: int) -> str:
    if not use_native("IMAGE") or _native is None:
        return _compat_format_byte_size(n)
    return _native.image.format_byte_size(n)
