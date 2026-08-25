"""PDF page-screenshot helper for the ``read`` tool.

Renders a single PDF page to PNG via PyMuPDF and reuses the same image
compression / data-URL delivery pipeline as ``read_image``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from kosong.tooling import ToolError, ToolOk, ToolReturnValue

from kimi_cli.utils.image_compress import (
    compress_image_for_model,
    mipmap_downsample,
    resolve_max_image_edge_px,
    resolve_read_image_byte_budget,
)
from kimi_cli.utils.media_tags import wrap_media_part
from kimi_cli.wire.types import ImageURLPart, TextPart

from .read_media_shared import (
    ImageDelivery,
    build_image_delivery_limit_error,
    build_media_note,
    to_data_url,
)

__all__ = ["render_pdf_page"]

# DPI fallback sequence for over-budget screenshots.
_DPI_SEQUENCE = (150, 96, 72)


def _render_pdf_page_to_png(path: str | Path, page: int, dpi: int) -> bytes:
    """Render a single PDF page to PNG bytes using PyMuPDF."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(f"PyMuPDF is required for PDF page rendering: {exc}") from exc
    doc = pymupdf.open(path)
    try:
        if page < 1 or page > len(doc):
            raise ValueError(f"Page {page} is out of range (1..{len(doc)})")
        page_obj = doc.load_page(page - 1)
        pix = page_obj.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0))
        return pix.tobytes("png")
    finally:
        doc.close()


def _build_pdf_delivery_preview(delivery: ImageDelivery) -> str:
    return (
        f"[PDF page image: {delivery.kind}, {delivery.width}x{delivery.height}, "
        f"{delivery.byte_length} bytes]\n"
    )


def _compress_pdf_page(
    png_data: bytes,
    media_path: str,
    max_edge: int,
    byte_budget: int,
) -> tuple[list | str, ImageDelivery] | ToolError:
    """Compress a PNG page screenshot and wrap it for model delivery."""
    compressed = compress_image_for_model(
        png_data,
        "image/png",
        max_edge=max_edge,
        byte_budget=byte_budget,
    )
    if (
        compressed.final_byte_length > byte_budget
        or max(compressed.width, compressed.height) > max_edge
    ):
        # Try mipmap fallback.
        result = mipmap_downsample(
            png_data,
            "image/png",
            max_edge=max_edge,
            byte_budget=byte_budget,
        )
        if (
            result.changed
            and result.final_byte_length <= byte_budget
            and max(result.width, result.height) <= max_edge
        ):
            part = ImageURLPart(
                image_url=ImageURLPart.ImageURL(url=to_data_url(result.mime_type, result.data))
            )
            wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
            delivery = ImageDelivery(
                kind="downsampled",
                width=result.width,
                height=result.height,
                byte_length=result.final_byte_length,
                mime_type=result.mime_type,
                mipmap=True,
            )
            return wrapped, delivery
        return ToolError(
            message=build_image_delivery_limit_error(
                compressed.final_byte_length, byte_budget, max_edge
            ),
            brief="PDF page too large",
        )

    part = ImageURLPart(
        image_url=ImageURLPart.ImageURL(url=to_data_url(compressed.mime_type, compressed.data))
    )
    wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
    delivery = ImageDelivery(
        kind="downsampled" if compressed.changed else "untouched",
        width=compressed.width,
        height=compressed.height,
        byte_length=compressed.final_byte_length,
        mime_type=compressed.mime_type,
    )
    return wrapped, delivery


def render_pdf_page(
    path: str | Path,
    page: int,
    *,
    capabilities: set[str],
) -> ToolReturnValue:
    """Render *page* of *path* as an image part ready for the model.

    Returns a :class:`ToolError` when the model lacks image input support or
    the rendered page cannot be delivered within budgets.
    """
    if "image_in" not in capabilities:
        return ToolError(
            message=(
                "The current model does not support image input. "
                "Read the PDF as text by leaving pdf_page unset, or use a model with image input capability."
            ),
            brief="Unsupported media type",
        )

    media_path = str(path)
    max_edge = resolve_max_image_edge_px()
    byte_budget = resolve_read_image_byte_budget()

    last_error: ToolError | None = None
    for dpi in _DPI_SEQUENCE:
        try:
            png_data = _render_pdf_page_to_png(path, page, dpi)
        except ValueError as exc:
            return ToolError(message=str(exc), brief="Invalid PDF page")
        except Exception as exc:
            return ToolError(
                message=f"Failed to render PDF page {page}: {exc}",
                brief="PDF render failed",
            )

        result = _compress_pdf_page(png_data, media_path, max_edge, byte_budget)
        if isinstance(result, ToolError):
            last_error = result
            continue
        wrapped, delivery = result
        preview = _build_pdf_delivery_preview(delivery)
        if isinstance(wrapped, list):
            wrapped = [TextPart(text=preview)] + wrapped
        else:
            wrapped = f"{preview}{wrapped}"

        note = (
            f"<system>PDF page {page} of {media_path}, delivered as "
            f"{delivery.width}x{delivery.height} PNG "
            f"({delivery.byte_length} bytes)."
            "</system>"
        )
        return ToolOk(output=wrapped, message=note)

    if last_error is not None:
        # Suggest text extraction on final failure.
        msg = (
            last_error.message
            + " Consider reading the PDF as text instead by leaving pdf_page unset."
        )
        return ToolError(message=msg, brief=last_error.brief)
    return ToolError(
        message="Could not deliver PDF page within image budgets.",
        brief="PDF page too large",
    )
