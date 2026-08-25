"""Shared helpers for delivering image/video media parts.

Used by both :mod:`read_media` and :mod:`read_pdf_pages` so PDF page
screenshots can reuse the same data-URL, compression, and media-note
machinery without duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pybase64

from kimi_cli.utils.image_compress import CropRegion, format_byte_size, IMAGE_BYTE_BUDGET

__all__ = [
    "ImageDelivery",
    "to_data_url",
    "build_image_delivery_limit_error",
    "build_media_note",
]


@dataclass(frozen=True)
class ImageDelivery:
    """How the image payload placed after the summary relates to the file on
    disk. Reported verbatim so the model always knows when it is looking at a
    degraded copy (and how to get the detail back)."""

    kind: Literal["untouched", "downsampled", "crop", "full"]
    # Pixel size of the payload actually sent; 0 when unknown.
    width: int
    height: int
    byte_length: int
    mime_type: str
    # The crop actually applied (clamped), for kind "crop".
    region: CropRegion | None = None
    # For kind "crop": the crop was additionally downscaled to fit budgets.
    resized: bool | None = None
    # True when mip-map (numpy-based 2x2 bilinear averaging) was used.
    mipmap: bool = False


def to_data_url(mime_type: str, data: bytes) -> str:
    encoded = pybase64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_image_delivery_limit_error(final_bytes: int, read_byte_budget: int, max_edge: int) -> str:
    return (
        f"Image is too large to send safely after compression ({final_bytes} bytes; "
        f"limit {read_byte_budget} bytes and {max_edge}px on the longest edge). "
        "The original image was not sent to the model. Do not retry the same file unchanged. "
        "Use Bash or an available image-processing tool to create a smaller copy within both "
        "limits, then call read_image on the smaller copy."
    )


def build_media_note(
    *,
    kind: Literal["image", "video"],
    mime_type: str,
    byte_size: int,
    dimensions: tuple[int, int] | None,
    delivery: ImageDelivery | None,
) -> str:
    """Build the media summary returned as the tool result's model-facing
    message, wrapped in a ``<system>`` block."""
    parts = [
        f"Read {kind} file.",
        f"Mime type: {mime_type}.",
        f"Size: {byte_size} bytes.",
    ]
    if kind == "image" and dimensions:
        parts.append(f"Original dimensions: {dimensions[0]}x{dimensions[1]} pixels.")
    if delivery and delivery.kind == "downsampled":
        parts.append(
            f"The attached image was downsampled to {delivery.width}x{delivery.height} "
            f"pixels ({delivery.mime_type}, {format_byte_size(delivery.byte_length)}) "
            "to fit model limits; fine detail may be lost."
        )
        parts.append(
            "To inspect fine detail, call read_image again with the region parameter "
            "(original-image pixel coordinates) to view a crop at full fidelity."
        )
        if delivery.mipmap:
            parts.append(
                "Warning: Mip-map downsampling (2x2 bilinear averaging) was used "
                "because standard compression could not meet the delivery limits; "
                "fine detail may be significantly reduced."
            )
    elif delivery and delivery.kind == "crop" and delivery.region:
        region = delivery.region
        how = (
            f", downsampled to {delivery.width}x{delivery.height} pixels"
            if delivery.resized
            else " at native resolution"
        )
        parts.append(
            f"Showing region (x={region.x}, y={region.y}, width={region.width}, "
            f"height={region.height}) of the original image{how}."
        )
        parts.append(
            "To output coordinates in original-image pixels, locate them within this "
            f"crop and add the region offset (x={region.x}, y={region.y})."
        )
    elif delivery and delivery.kind == "full":
        parts.append("Shown at native resolution; no downscaling applied.")
    if kind == "image" and dimensions and (delivery is None or delivery.kind != "crop"):
        parts.append(
            "If you need to output coordinates, output relative coordinates first "
            "and compute absolute coordinates using the original image size."
        )
    parts.append(
        "If you generate or edit images or videos via commands or scripts, "
        "read the result back immediately before continuing."
    )
    return f"<system>{' '.join(parts)}</system>"
