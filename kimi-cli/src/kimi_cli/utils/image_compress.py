"""Shrink oversized images before they reach the model.

A multimodal request carries each image as a base64 data URL; an unbounded
screenshot or photo wastes context tokens and can blow past the provider's
per-image byte ceiling. This module downsamples and re-encodes such images
so they fit a pixel + byte budget, while leaving already-small images
untouched — the common case is a fast, codec-free pass-through.

Design notes:
- Pillow is the codec, imported lazily inside the functions that need it so
  startup and the fast path stay cheap (mirrors the TS lazy codec loading).
- Best effort: any decode/encode failure of :func:`compress_image_for_model`
  returns the original bytes unchanged (``changed=False``). Callers must
  verify that this unchanged result satisfies their delivery limits before
  forwarding it.
- PNG, JPEG, and (non-animated) WebP are re-encoded. GIF and animated WebP
  are passed through to preserve animation. Formats outside the
  provider-accepted set (see :mod:`kimi_cli.utils.image_format_policy`) are
  never forwarded by the tool — callers gate on
  ``is_model_accepted_image_mime`` first.
- Compression must never be silent to the model: results carry the original
  dimensions, and :func:`crop_image_for_model` lets a caller read a region
  of the original back at full fidelity.

Ported from ``packages/agent-core/src/tools/support/image-compress.ts`` and
the ``sniffImageDimensions`` part of ``.../support/file-type.ts`` in the Kimi
Code TypeScript monorepo. The TS module's telemetry events have no
Python-side equivalent and are not ported.
"""

from __future__ import annotations

import math
import os
import regex as re
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Literal

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kimi_cli.utils.image_format_policy import normalize_image_mime

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_IMAGE = _native_get_module("image")

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# ---------------------------------------------------------------------------
# Budgets and limits
# ---------------------------------------------------------------------------

# Built-in longest-edge ceiling (px). Larger images are scaled down to fit.
# This is the default only: the effective ceiling is resolved per call by
# resolve_max_image_edge_px (env var > this).
MAX_IMAGE_EDGE_PX = 2000

# Env var overriding the longest-edge ceiling (px). Read live on every
# resolution so it applies in any process without wiring; a value that is
# not a positive integer is ignored.
MAX_IMAGE_EDGE_ENV = "KIMI_IMAGE_MAX_EDGE_PX"

# Raw-byte budget for a single image. base64 inflates bytes by ~4/3, so a
# 3.75 MB raw payload stays under a 5 MB encoded ceiling. Tune to the active
# provider's per-image limit.
IMAGE_BYTE_BUDGET = int(3.75 * 1024 * 1024)

# Built-in raw-byte budget for images the model reads for itself
# (ReadMediaFile's default path). Far below IMAGE_BYTE_BUDGET: a session
# that keeps screenshotting and reading images accumulates every one of
# them in the request body on every turn, so per-image size — not the
# provider's per-image ceiling — is what keeps the total under the
# provider's request-size limit. 256 KB keeps a clean 2000px UI screenshot
# on the lossless fast path while capping dense content at a readable
# q80/1000px JPEG; fine detail stays reachable through the `region`
# readback, which deliberately ignores this budget.
READ_IMAGE_BYTE_BUDGET = 256 * 1024

# Env var overriding the read-image byte budget. Read live on every
# resolution; a value that is not a positive integer is ignored.
READ_IMAGE_BYTE_BUDGET_ENV = "KIMI_IMAGE_READ_BYTE_BUDGET"

# Pixel-count ceiling above which we skip compression entirely. A tiny-byte,
# huge-dimension image (e.g. a solid 30000x30000 PNG) would otherwise be
# fully decoded into a multi-gigabyte bitmap before any resize — a
# decompression-bomb OOM vector, since the byte budget alone never catches
# it. The header sniff gives us the dimensions without decoding, so we gate
# on them first. Set well above any legitimate photo/screenshot/scan
# (~100 MP); larger images pass through uncompressed.
MAX_DECODE_PIXELS = 100_000_000

# Raw-byte ceiling above which compression is skipped rather than decoded.
# The byte budget bounds the *output*, but the compressor still has to load
# the *input* first: a huge payload would be read into memory before any
# downstream cap can drop it. This bounds that input allocation. Set well
# above legitimate screenshots/photos; larger images pass through
# uncompressed.
MAX_IMAGE_DECODE_BYTES = 64 * 1024 * 1024

# Progressively lower JPEG quality until the payload fits the byte budget.
JPEG_QUALITY_STEPS = (80, 60, 40, 20)

# Longest-edge step-downs tried when the budget cannot be met at the fitted
# size. With the built-in 2000px ceiling the first step is a no-op; it
# matters when a larger ceiling is configured (env). The sub-1000px tail
# exists for small (read-scale) budgets: JPEG bytes shrink roughly linearly
# with pixel count, so stepping down to 256px lets even entropy-upper-bound
# content (noise, photos) land within any budget of a few tens of KB
# instead of stalling at the q20@1000px floor.
FALLBACK_EDGES_PX = (2000, 1000, 768, 512, 384, 256)

# PNG rescales stop at this edge; below it the ladder goes lossy instead.
# For text-bearing screenshots a q80 JPEG at 1000px reads better than a
# lossless PNG at 512px — resolution beats losslessness once both are
# degraded — so sub-floor edges are only ever tried with the JPEG ladder.
PNG_RESCALE_FLOOR_PX = 1000

# Formats we can decode and re-encode. GIF is never re-encoded (animation
# preservation); animated WebP is gated to a passthrough before decoding.
_RECODABLE_MIME = frozenset({"image/png", "image/jpeg", "image/webp"})

# Raw-byte ceiling for mipmap decode attempts — higher than MAX_IMAGE_DECODE_BYTES
# because numpy can handle larger arrays more efficiently than Pillow's
# processing pipeline.
MAX_MIPMAP_DECODE_BYTES = 128 * 1024 * 1024  # 128 MB

# Minimum dimension for mipmap levels — don't go below this.
MIN_MIPMAP_EDGE_PX = 2

_POSITIVE_INT_RE = re.compile(r"^[0-9]+$")


def _positive_int_from_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw or not _POSITIVE_INT_RE.match(raw):
        return None
    parsed = int(raw)
    return parsed if parsed > 0 else None


def resolve_max_image_edge_px() -> int:
    """Longest-edge ceiling (px): env var > built-in MAX_IMAGE_EDGE_PX."""
    return _positive_int_from_env(MAX_IMAGE_EDGE_ENV) or MAX_IMAGE_EDGE_PX


def resolve_read_image_byte_budget() -> int:
    """Read-image byte budget: env var > built-in READ_IMAGE_BYTE_BUDGET."""
    return _positive_int_from_env(READ_IMAGE_BYTE_BUDGET_ENV) or READ_IMAGE_BYTE_BUDGET


def format_byte_size(n: int) -> str:
    """Human-readable byte size: ``640 B`` / ``128 KB`` / ``3.8 MB``."""
    # Native acceleration: kimix_native.image.format_byte_size is
    # byte-identical (B / KB / MB formatting); the Python body is unchanged.
    if _native_use_native("IMAGE") and _NATIVE_IMAGE is not None:
        return _NATIVE_IMAGE.format_byte_size(n)
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        # JS Math.round semantics (round half up), matching the TS original.
        return f"{math.floor(n / 1024 + 0.5)} KB"
    return f"{n / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# Header-only dimension sniffing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageDimensions:
    width: int
    height: int
    # True when a JPEG EXIF orientation of 5-8 swapped the reported
    # width/height into display space.
    transposed: bool = False


def sniff_image_dimensions(data: bytes) -> ImageDimensions | None:
    """Best-effort pixel-dimension reader for common raster formats.

    Inspects only the fixed region near the start of the file where each
    format records its dimensions (the IHDR/DIB header, the RIFF chunk
    after the ``WEBP`` tag, or the first JPEG SOFn segment). Returns None
    for formats whose dimensions are not locatable from that region, or
    when the supplied buffer is too short to cover it.

    JPEG dimensions are reported in DISPLAY space: an EXIF Orientation of
    5-8 transposes the image at decode time, so the SOF width/height are
    swapped to match what decoders (and this codebase's crop regions and
    compression captions) actually operate in.
    """
    # Native acceleration: kimix_native.image.sniff_dimensions returns the
    # same (width, height, transposed) triple the header-scan below computes;
    # the pure-Python body is unchanged.
    if _native_use_native("IMAGE") and _NATIVE_IMAGE is not None:
        dims = _NATIVE_IMAGE.sniff_dimensions(data)
        if dims is None:
            return None
        width, height, transposed = dims
        return ImageDimensions(width=width, height=height, transposed=bool(transposed))
    # PNG — IHDR is the first chunk; width/height are big-endian uint32
    # at offsets 16 and 20.
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return ImageDimensions(
            width=int.from_bytes(data[16:20], "big"),
            height=int.from_bytes(data[20:24], "big"),
        )

    # GIF — logical-screen width/height are little-endian uint16 at
    # offsets 6 and 8.
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return ImageDimensions(
            width=int.from_bytes(data[6:8], "little"),
            height=int.from_bytes(data[8:10], "little"),
        )

    # BMP — DIB header width/height are little-endian int32 at offsets 18
    # and 22 (height may be negative for top-down bitmaps).
    if data.startswith(b"BM") and len(data) >= 26:
        return ImageDimensions(
            width=int.from_bytes(data[18:22], "little", signed=True),
            height=abs(int.from_bytes(data[22:26], "little", signed=True)),
        )

    # WEBP — RIFF container; VP8/VP8L/VP8X each store dimensions
    # differently in the chunk that follows the 'WEBP' tag.
    if data.startswith(b"RIFF") and len(data) >= 30:
        four_cc = data[12:16]
        if four_cc == b"VP8 ":
            return ImageDimensions(
                width=int.from_bytes(data[26:28], "little") & 0x3FFF,
                height=int.from_bytes(data[28:30], "little") & 0x3FFF,
            )
        if four_cc == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return ImageDimensions(
                width=(bits & 0x3FFF) + 1,
                height=((bits >> 14) & 0x3FFF) + 1,
            )
        if four_cc == b"VP8X":
            width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
            height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
            return ImageDimensions(width=width, height=height)

    # JPEG — scan segment markers for a Start-Of-Frame (SOFn) marker,
    # whose payload carries height/width as big-endian uint16. An EXIF
    # APP1 segment encountered on the way supplies the orientation.
    if data.startswith(b"\xff\xd8"):
        orientation: int | None = None
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            # SOFn markers carry frame dimensions; skip SOF4/SOF8/SOF12
            # (0xc4/0xc8/0xcc).
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height = int.from_bytes(data[offset + 5 : offset + 7], "big")
                width = int.from_bytes(data[offset + 7 : offset + 9], "big")
                if orientation is not None and orientation >= 5:
                    return ImageDimensions(width=height, height=width, transposed=True)
                return ImageDimensions(width=width, height=height)
            # Standalone markers (RSTn, SOI, EOI) carry no length field.
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                offset += 2
                continue
            segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
            if segment_length < 2:
                break
            if marker == 0xE1 and orientation is None:
                orientation = _read_exif_orientation(data, offset + 4, offset + 2 + segment_length)
            offset += 2 + segment_length

    return None


def _read_exif_orientation(data: bytes, start: int, end: int) -> int | None:
    """Read the Orientation tag (0x0112) out of a JPEG APP1 payload.

    Returns 1-8, or None when the payload is not EXIF, is truncated, or
    carries no valid orientation. Only IFD0 is examined — that is where the
    tag lives; nothing here follows nested IFDs.
    """
    bounded_end = min(end, len(data))
    # 'Exif\0\0' preamble, then the TIFF header.
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
            # Type SHORT: the value sits in the first two bytes of the
            # 4-byte value field, in the TIFF byte order.
            value = u16(entry + 8)
            return value if 1 <= value <= 8 else None
    return None


def _is_animated_webp(data: bytes) -> bool:
    """True when the payload is a WebP whose VP8X container header carries
    the ANIM flag. Animated WebP must be passed through, not re-encoded:
    decoding yields a single frame and would silently destroy the animation
    (the same reason GIF is passed through)."""
    # Native acceleration: kimix_native.image.is_animated_webp (bit-identical).
    if _native_use_native("IMAGE") and _NATIVE_IMAGE is not None:
        return _NATIVE_IMAGE.is_animated_webp(data)
    return (
        len(data) >= 21
        and data[0:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and data[12:16] == b"VP8X"
        and (data[20] & 0x02) != 0
    )


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressImageResult:
    # Bytes to send: the re-encoded image, or the original when unchanged.
    data: bytes
    # MIME of `data`. May differ from the input (e.g. png -> jpeg).
    mime_type: str
    # Pixel width of `data`; falls back to the input size when unknown.
    width: int
    # Pixel height of `data`; falls back to the input size when unknown.
    height: int
    # Pixel width of the input image, in display space (EXIF orientation
    # applied): the decoded width when re-encoded, the header sniff on
    # passthrough (0 when it cannot be determined).
    original_width: int
    # Pixel height of the input image; see original_width.
    original_height: int
    # True only when `data` differs from the input bytes.
    changed: bool
    original_byte_length: int
    final_byte_length: int


def compress_image_for_model(
    data: bytes,
    mime_type: str,
    *,
    max_edge: int | None = None,
    byte_budget: int | None = None,
    max_decode_bytes: int = MAX_IMAGE_DECODE_BYTES,
) -> CompressImageResult:
    """Downsample/re-encode `data` to fit the pixel + byte budget.

    Never raises: on any failure (unsupported format, decode error, a
    result that would be larger than the input) the original bytes are
    returned with ``changed=False``.
    """
    if max_edge is None:
        max_edge = resolve_max_image_edge_px()
    if byte_budget is None:
        byte_budget = IMAGE_BYTE_BUDGET
    normalized_mime = normalize_image_mime(mime_type)
    dims = sniff_image_dimensions(data)

    def passthrough() -> CompressImageResult:
        width = dims.width if dims else 0
        height = dims.height if dims else 0
        return CompressImageResult(
            data=data,
            mime_type=mime_type,
            width=width,
            height=height,
            original_width=width,
            original_height=height,
            changed=False,
            original_byte_length=len(data),
            final_byte_length=len(data),
        )

    if len(data) == 0:
        return passthrough()
    # Only re-encode formats the codec handles; everything else passes through.
    if normalized_mime not in _RECODABLE_MIME:
        return passthrough()
    # Animated WebP would be flattened to one frame by decoding — pass it
    # through whole, the same reason GIF is never re-encoded.
    if normalized_mime == "image/webp" and _is_animated_webp(data):
        return passthrough()

    # Fast path: already within both budgets — no codec load, no allocation.
    longest_edge = max(dims.width, dims.height) if dims else 0
    within_bytes = len(data) <= byte_budget
    within_edge = 0 < longest_edge <= max_edge
    if within_bytes and (within_edge or longest_edge == 0):
        return passthrough()

    # Decompression-bomb guard: refuse to decode absurd pixel counts. The
    # sniff above gave us the dimensions without decoding, so this costs
    # nothing.
    if dims and dims.width * dims.height > MAX_DECODE_PIXELS:
        return passthrough()
    # Refuse to decode very large byte payloads that would be loaded just
    # to be dropped downstream.
    if len(data) > max_decode_bytes:
        return passthrough()

    try:
        image = _decode_image(data)
        # WebP joins PNG on the lossless-first ladder: both carry alpha and
        # screenshot-grade detail that the PNG rungs preserve.
        prefer_lossless = normalized_mime != "image/jpeg"
        # The decoded bitmap is authoritative for the original size: the
        # decoder applies EXIF orientation, and this is the coordinate space
        # the encoded result and any later crop region (see
        # crop_image_for_model, which decodes the same way) actually live
        # in. The header sniff also reports display space, but can miss
        # formats or nonconforming EXIF that the decoder still handles.
        decoded_width, decoded_height = image.size

        # Scale so the longest edge fits max_edge (never enlarges).
        image = _fit_within_edge(image, max_edge)

        encoded = _encode_within_budget(
            image,
            prefer_lossless=prefer_lossless,
            byte_budget=byte_budget,
        )

        # Keep the result when it actually helps: fewer bytes, or fewer
        # pixels (a smaller image costs fewer vision tokens even if the
        # byte count is flat, as with near-solid graphics). Otherwise the
        # re-encode bought us nothing — return the original.
        original_pixels = decoded_width * decoded_height
        final_pixels = encoded.width * encoded.height
        shrank_bytes = len(encoded.data) < len(data)
        shrank_pixels = final_pixels < original_pixels
        if not shrank_bytes and not shrank_pixels:
            return passthrough()

        return CompressImageResult(
            data=encoded.data,
            mime_type=encoded.mime_type,
            width=encoded.width,
            height=encoded.height,
            original_width=decoded_width,
            original_height=decoded_height,
            changed=True,
            original_byte_length=len(data),
            final_byte_length=len(encoded.data),
        )
    except Exception:
        # Decode/encode failure — keep the original bytes.
        return passthrough()


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropRegion:
    """Crop rectangle in ORIGINAL-image pixel coordinates — the decoded,
    EXIF-rotated space that compression results report as the original
    size."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class CropImageSuccess:
    ok: Literal[True]
    data: bytes
    mime_type: str
    # Pixel size of the encoded crop actually produced.
    width: int
    height: int
    # Pixel size of the source image the region was cut from.
    original_width: int
    original_height: int
    # The region actually applied, after clamping to the image bounds.
    region: CropRegion
    # True when the crop was downscaled to fit the pixel/byte budget.
    resized: bool
    original_byte_length: int
    final_byte_length: int


@dataclass(frozen=True)
class CropImageFailure:
    ok: Literal[False]
    # Human/model-readable reason, safe to surface as a tool error.
    error: str


CropImageOutcome = CropImageSuccess | CropImageFailure


def crop_image_for_model(
    data: bytes,
    mime_type: str,
    region: CropRegion,
    *,
    skip_resize: bool = False,
    max_edge: int | None = None,
    byte_budget: int = IMAGE_BYTE_BUDGET,
    max_decode_bytes: int = MAX_IMAGE_DECODE_BYTES,
) -> CropImageOutcome:
    """Cut `region` out of `data` and encode it for the model.

    Unlike :func:`compress_image_for_model`, cropping is an explicit
    request: it never falls back to the full image. Anything that prevents
    an accurate crop (unsupported format, undecodable bytes, a region
    outside the image, a skip_resize result over the byte budget) returns a
    failure with a reason the caller can hand straight back to the model.

    The default path fits the crop to the usual pixel/byte budgets; a crop
    no larger than the edge cap is therefore delivered at native
    resolution.
    """
    if max_edge is None:
        max_edge = resolve_max_image_edge_px()
    normalized_mime = normalize_image_mime(mime_type)

    if len(data) == 0:
        return CropImageFailure(ok=False, error="The image is empty.")
    if normalized_mime not in _RECODABLE_MIME:
        return CropImageFailure(
            ok=False,
            error=f"Cropping is only supported for PNG, JPEG, and WebP images; got {mime_type}.",
        )
    # A crop is a still image by definition; decoding an animated WebP
    # would silently crop a single frame, so refuse explicitly.
    if normalized_mime == "image/webp" and _is_animated_webp(data):
        return CropImageFailure(
            ok=False, error="Cropping is not supported for animated WebP images."
        )
    dims = sniff_image_dimensions(data)
    if dims and dims.width * dims.height > MAX_DECODE_PIXELS:
        return CropImageFailure(
            ok=False,
            error=(
                f"The image ({dims.width}x{dims.height} pixels) "
                "is too large to decode for cropping."
            ),
        )
    if len(data) > max_decode_bytes:
        return CropImageFailure(
            ok=False, error="The image is too large to decode for cropping."
        )

    try:
        image = _decode_image(data)
        original_width, original_height = image.size

        x = region.x
        y = region.y
        if (
            x < 0
            or y < 0
            or x >= original_width
            or y >= original_height
            or region.width < 1
            or region.height < 1
        ):
            return CropImageFailure(
                ok=False,
                error=(
                    f"Region (x={region.x}, y={region.y}, width={region.width}, "
                    f"height={region.height}) lies outside the "
                    f"{original_width}x{original_height} image."
                ),
            )
        w = min(region.width, original_width - x)
        h = min(region.height, original_height - y)
        applied = CropRegion(x=x, y=y, width=w, height=h)
        cropped = image.crop((x, y, x + w, y + h))
        # WebP joins PNG on the lossless side: both carry alpha and
        # screenshot-grade detail that PNG output preserves.
        prefer_lossless = normalized_mime != "image/jpeg"

        if skip_resize:
            # Native resolution requested: encode once, favoring fidelity
            # (lossless PNG, or high-quality JPEG), and refuse rather than
            # degrade when the result cannot fit the byte budget.
            buffer = _encode_png(cropped) if prefer_lossless else _encode_jpeg(cropped, 90)
            if len(buffer) > byte_budget:
                return CropImageFailure(
                    ok=False,
                    error=(
                        f"The cropped region encodes to {len(buffer)} bytes "
                        f"({format_byte_size(len(buffer))}), over the {byte_budget}-byte "
                        f"({format_byte_size(byte_budget)}) per-image limit. "
                        "Choose a smaller region, or allow downscaling."
                    ),
                )
            return CropImageSuccess(
                ok=True,
                data=buffer,
                mime_type="image/png" if prefer_lossless else "image/jpeg",
                width=cropped.width,
                height=cropped.height,
                original_width=original_width,
                original_height=original_height,
                region=applied,
                resized=False,
                original_byte_length=len(data),
                final_byte_length=len(buffer),
            )

        fitted = _fit_within_edge(cropped, max_edge)
        encoded = _encode_within_budget(
            fitted,
            prefer_lossless=prefer_lossless,
            byte_budget=byte_budget,
        )
        return CropImageSuccess(
            ok=True,
            data=encoded.data,
            mime_type=encoded.mime_type,
            width=encoded.width,
            height=encoded.height,
            original_width=original_width,
            original_height=original_height,
            region=applied,
            resized=encoded.width != w or encoded.height != h,
            original_byte_length=len(data),
            final_byte_length=len(encoded.data),
        )
    except Exception as e:
        return CropImageFailure(
            ok=False,
            error=f"Failed to decode the image for cropping: {e}",
        )


# ---------------------------------------------------------------------------
# Internals (Pillow-based codec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EncodedImage:
    data: bytes
    mime_type: str
    width: int
    height: int


def _decode_image(data: bytes) -> Image.Image:
    """Decode bytes into a PIL image in display space (EXIF orientation
    applied), normalized to RGB/RGBA so resize and re-encode behave
    consistently across source modes."""
    from PIL import Image, ImageOps

    with warnings.catch_warnings():
        # The pixel-count guard above already gates decompression bombs at
        # MAX_DECODE_PIXELS; PIL's own warning must not abort the decode.
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as img:
            image = ImageOps.exif_transpose(img)
            image.load()
            image = image.copy()
    if image.mode in ("RGB", "RGBA"):
        return image
    if image.mode in ("LA", "PA") or "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def _encode_png(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    if image.mode != "RGB":
        # JPEG cannot carry alpha; converting drops the alpha channel.
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _fit_within_edge(image: Image.Image, edge: int) -> Image.Image:
    """Scale so the longest edge is at most `edge`, preserving aspect
    ratio. Returns the input unchanged when the image already fits (never
    enlarges)."""
    from PIL import Image

    longest = max(image.width, image.height)
    if longest <= edge:
        return image
    factor = edge / longest
    new_size = (
        max(1, math.floor(image.width * factor + 0.5)),
        max(1, math.floor(image.height * factor + 0.5)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _encode_within_budget(
    image: Image.Image,
    *,
    prefer_lossless: bool,
    byte_budget: int,
) -> _EncodedImage:
    """Encode `image` (already fitted to the edge ceiling) under the byte
    budget.

    Strategy — prefer the source format so a downscaled screenshot stays
    lossless PNG (preserving text and transparency), and only fall back to
    lossy JPEG when PNG cannot meet the byte budget:
    - PNG source: PNG at the fitted size -> smaller PNG rescales down to
      the PNG_RESCALE_FLOOR_PX floor -> JPEG ladder at that size -> JPEG
      ladder again at each sub-floor edge.
    - JPEG source: the full quality ladder at the fitted size, then again
      at each fallback edge — a smaller rescale must not skip the
      high-quality rungs its extra pixels just paid for.

    The sub-floor edges make the ladder converge for small (read-scale)
    budgets: any budget of a few tens of KB is met by q20 at 256px even
    for entropy-upper-bound content. Below that, the smallest buffer
    produced is still returned — the caller gates on whether it actually
    helped.
    """
    smallest: _EncodedImage | None = None

    def consider(data: bytes, mime_type: str, width: int, height: int) -> _EncodedImage:
        nonlocal smallest
        candidate = _EncodedImage(data=data, mime_type=mime_type, width=width, height=height)
        if smallest is None or len(candidate.data) < len(smallest.data):
            smallest = candidate
        return candidate

    def jpeg_ladder(img: Image.Image) -> _EncodedImage | None:
        for quality in JPEG_QUALITY_STEPS:
            jpeg = _encode_jpeg(img, quality)
            candidate = consider(jpeg, "image/jpeg", img.width, img.height)
            if len(jpeg) <= byte_budget:
                return candidate
        return None

    if prefer_lossless:
        # Lossless PNG first: best for screenshots/UI (sharp text) and
        # keeps alpha.
        current = image
        png = _encode_png(current)
        candidate = consider(png, "image/png", current.width, current.height)
        if len(png) <= byte_budget:
            return candidate

        # Over budget: progressively smaller PNGs (down to the floor)
        # before going lossy.
        for edge in FALLBACK_EDGES_PX:
            if edge < PNG_RESCALE_FLOOR_PX:
                break
            resized = _fit_within_edge(current, edge)
            if resized is current:
                continue
            current = resized
            smaller_png = _encode_png(current)
            candidate = consider(smaller_png, "image/png", current.width, current.height)
            if len(smaller_png) <= byte_budget:
                return candidate

        # Lossy JPEG ladder (drops transparency) at the floored size, then
        # at each sub-floor edge until the budget is met.
        at_floor = jpeg_ladder(current)
        if at_floor is not None:
            return at_floor
        for edge in FALLBACK_EDGES_PX:
            if edge >= PNG_RESCALE_FLOOR_PX:
                continue
            resized = _fit_within_edge(current, edge)
            if resized is current:
                continue
            current = resized
            at_edge = jpeg_ladder(current)
            if at_edge is not None:
                return at_edge
        assert smallest is not None
        return smallest

    # JPEG source: quality ladder at the fitted size, then the full
    # ladder again at each fallback rescale.
    at_fitted = jpeg_ladder(image)
    if at_fitted is not None:
        return at_fitted
    current = image
    for edge in FALLBACK_EDGES_PX:
        resized = _fit_within_edge(current, edge)
        if resized is current:
            continue
        current = resized
        at_edge = jpeg_ladder(current)
        if at_edge is not None:
            return at_edge

    assert smallest is not None
    return smallest


# ---------------------------------------------------------------------------
# Mip-map downsampling (numpy-based box-filter pyramid)
# ---------------------------------------------------------------------------


def _mipmap_one_level(arr: np.ndarray) -> np.ndarray:
    """Downsample one mip-map level: average each non-overlapping 2×2 block.

    Input shape ``(H, W, C)``, output shape ``(H//2, W//2, C)``.  Odd
    dimensions are floored (the right/bottom edge pixels are dropped).
    """
    h, w, c = arr.shape
    h2, w2 = h // 2, w // 2
    # Crop to even dimensions, reshape to group 2×2 blocks, mean over spatial axes
    return arr[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2, c).mean(axis=(1, 3)).astype(arr.dtype)


def _ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a numpy ``uint8`` array ``(H, W, 3)`` or ``(H, W, 4)`` back to a
    PIL Image (RGB or RGBA respectively)."""
    from PIL import Image

    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr, "RGB")


def mipmap_downsample(
    data: bytes,
    mime_type: str,
    *,
    max_edge: int,
    byte_budget: int,
    max_mipmap_decode_bytes: int = MAX_MIPMAP_DECODE_BYTES,
) -> CompressImageResult:
    """Build a mip-map pyramid via 2×2 bilinear averaging and return the
    first level that fits within both ``max_edge`` (longest-edge pixels) and
    ``byte_budget`` (encoded bytes).

    Falls back to a passthrough result (``changed=False``) on any failure
    — decode error, unsupported format, numpy memory error, etc. — never
    raises.
    """
    normalized_mime = normalize_image_mime(mime_type)
    dims = sniff_image_dimensions(data)

    def passthrough() -> CompressImageResult:
        width = dims.width if dims else 0
        height = dims.height if dims else 0
        return CompressImageResult(
            data=data,
            mime_type=mime_type,
            width=width,
            height=height,
            original_width=width,
            original_height=height,
            changed=False,
            original_byte_length=len(data),
            final_byte_length=len(data),
        )

    if len(data) == 0:
        return passthrough()
    if normalized_mime not in _RECODABLE_MIME:
        return passthrough()
    if normalized_mime == "image/webp" and _is_animated_webp(data):
        return passthrough()
    if dims and dims.width * dims.height > MAX_DECODE_PIXELS:
        return passthrough()
    # Fast path: already within both budgets — no decode, no pyramid.
    longest_edge = max(dims.width, dims.height) if dims else 0
    within_bytes = len(data) <= byte_budget
    within_edge = 0 < longest_edge <= max_edge
    if within_bytes and (within_edge or longest_edge == 0):
        return passthrough()

    if len(data) > max_mipmap_decode_bytes:
        return passthrough()

    try:
        image = _decode_image(data)
    except Exception:
        return passthrough()

    original_width, original_height = image.size
    prefer_lossless = normalized_mime != "image/jpeg"

    try:
        import numpy as np

        arr = np.asarray(image, dtype=np.uint8)
    except Exception:
        return passthrough()

    levels: list[tuple[np.ndarray, int, int]] = [(arr, original_width, original_height)]
    try:
        while True:
            h, w = arr.shape[0], arr.shape[1]
            if w <= MIN_MIPMAP_EDGE_PX or h <= MIN_MIPMAP_EDGE_PX:
                break
            arr = _mipmap_one_level(arr)
            h2, w2 = arr.shape[0], arr.shape[1]
            if h2 < MIN_MIPMAP_EDGE_PX or w2 < MIN_MIPMAP_EDGE_PX:
                break
            levels.append((arr, w2, h2))
    except (MemoryError, Exception):
        return passthrough()

    # Find the first level whose longest edge fits max_edge, then encode and
    # check the byte budget.  If none fits the edge, use the smallest level.
    first_that_fits_edge: int | None = None
    for idx in range(len(levels)):
        _, w, h = levels[idx]
        if max(w, h) <= max_edge:
            first_that_fits_edge = idx
            break
    if first_that_fits_edge is None:
        # Even the smallest level exceeds the edge cap — return smallest.
        first_that_fits_edge = len(levels) - 1

    smallest_level: tuple[np.ndarray, int, int] | None = None
    smallest_bytes: int | None = None

    for idx in range(first_that_fits_edge, len(levels)):
        level_arr, level_w, level_h = levels[idx]
        try:
            pil_img = _ndarray_to_pil(level_arr)
            encoded = _encode_within_budget(
                pil_img,
                prefer_lossless=prefer_lossless,
                byte_budget=byte_budget,
            )
        except Exception:
            continue

        if smallest_level is None or len(encoded.data) < (smallest_bytes or float("inf")):
            smallest_level = (level_arr, level_w, level_h)
            smallest_bytes = len(encoded.data)

        if len(encoded.data) <= byte_budget:
            return CompressImageResult(
                data=encoded.data,
                mime_type=encoded.mime_type,
                width=encoded.width,
                height=encoded.height,
                original_width=original_width,
                original_height=original_height,
                changed=True,
                original_byte_length=len(data),
                final_byte_length=len(encoded.data),
            )

    # No level met the byte budget — return smallest encoded result.
    if smallest_level is not None:
        pil_img = _ndarray_to_pil(smallest_level[0])
        try:
            encoded = _encode_within_budget(
                pil_img,
                prefer_lossless=prefer_lossless,
                byte_budget=byte_budget,
            )
            return CompressImageResult(
                data=encoded.data,
                mime_type=encoded.mime_type,
                width=encoded.width,
                height=encoded.height,
                original_width=original_width,
                original_height=original_height,
                changed=True,
                original_byte_length=len(data),
                final_byte_length=len(encoded.data),
            )
        except Exception:
            pass

    return passthrough()
