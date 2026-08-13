"""ReadMediaFile — read image/video files as multi-modal content.

Returns a 3-part wrap as ``output`` via
:func:`kimi_cli.utils.media_tags.wrap_media_part`
(``[TextPart('<image|video path="…">'), media part, TextPart('</…>')]``)
plus a model-facing ``message`` wrapped in a ``<system>`` block, and gates
on the model's ``image_in`` / ``video_in`` capability.

The ``<system>`` message summarizes mime type, byte size and (for images)
the original pixel dimensions, states exactly how the image was delivered
(untouched, downsampled, cropped, or native resolution) so compression is
never silent, guides the model to derive absolute coordinates from the
original size, and reminds it to re-read any media it generates or edits.

Images support two opt-in delivery controls: ``region`` cuts a rectangle
(original-image pixel coordinates) out of the file so fine detail survives
at full fidelity, and ``full_resolution`` skips the default downscale when
the payload fits the per-image byte budget (refusing explicitly when it
does not, instead of silently degrading). Default image reads fail closed
when compression cannot meet the byte/longest-edge delivery budgets.

Images whose format the provider rejects (AVIF, HEIC, BMP, TIFF, ICO, …)
are refused with a conversion command: once such an ``image_url`` lands in
the session history every later request fails, so the bytes must never
reach the model.

Ported from ``packages/agent-core/src/tools/builtin/file/read-media.ts``
in the Kimi Code TypeScript monorepo. The TS tool result's ``note`` side
channel has no kosong equivalent, so it maps to the model-facing
``message`` string, keeping the ``<system>`` wrapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import pybase64
from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import AliasChoices, BaseModel, Field, model_validator

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.file.utils import MEDIA_SNIFF_BYTES, FileType, detect_file_type
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.image_compress import (
    IMAGE_BYTE_BUDGET,
    MAX_IMAGE_DECODE_BYTES,
    CropRegion,
    compress_image_for_model,
    crop_image_for_model,
    format_byte_size,
    mipmap_downsample,
    resolve_max_image_edge_px,
    resolve_read_image_byte_budget,
    sniff_image_dimensions,
)
from kimi_cli.utils.image_format_policy import (
    build_image_conversion_guidance,
    is_model_accepted_image_mime,
)
from kimi_cli.utils.logging import logger
from kimi_cli.utils.media_tags import wrap_media_part
from kimi_cli.utils.path import (
    is_within_workspace,
    kaos_path_from_tool_input,
    kaos_path_from_user_input,
)
from kimi_cli.wire.types import ImageURLPart, TextPart, VideoURLPart

if TYPE_CHECKING:
    from kosong.chat_provider.kimi import Kimi

MAX_MEDIA_MEGABYTES = 100


def _to_data_url(mime_type: str, data: bytes) -> str:
    encoded = pybase64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_image_delivery_limit_error(
    final_bytes: int, read_byte_budget: int, max_edge: int
) -> str:
    return (
        f"Image is too large to send safely after compression ({final_bytes} bytes; "
        f"limit {read_byte_budget} bytes and {max_edge}px on the longest edge). "
        "The original image was not sent to the model. Do not retry the same file unchanged. "
        "Use Bash or an available image-processing tool to create a smaller copy within both "
        "limits, then call ReadMediaFile on the smaller copy."
    )


def _build_image_decode_limit_error(final_bytes: int) -> str:
    return (
        "Image is too large to process safely for region or full_resolution "
        f"({final_bytes} bytes; safe decode limit {MAX_IMAGE_DECODE_BYTES} bytes). "
        "The original image was not sent to the model. Do not retry the same file unchanged. "
        "Use Bash or an available image-processing tool to create a smaller copy or crop the "
        "needed region into a separate image, then call ReadMediaFile on the resulting file."
    )


def _try_mipmap_fallback(
    data: bytes,
    mime_type: str,
    media_path: str,
    max_edge: int,
    byte_budget: int,
    original_dimensions: tuple[int, int] | None,
) -> tuple[ImageURLPart, _ImageDelivery] | None:
    """Attempt mip-map downsampling when the normal compression path produces
    an image too large to deliver. Returns ``(ImageURLPart, _ImageDelivery)``
    on success, ``None`` when even the mipmap result is over budget."""
    result = mipmap_downsample(
        data,
        mime_type,
        max_edge=max_edge,
        byte_budget=byte_budget,
    )
    if (
        result.changed
        and result.final_byte_length <= byte_budget
        and max(result.width, result.height) <= max_edge
    ):
        part = ImageURLPart(
            image_url=ImageURLPart.ImageURL(
                url=_to_data_url(result.mime_type, result.data)
            )
        )
        wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
        delivery = _ImageDelivery(
            kind="downsampled",
            width=result.width,
            height=result.height,
            byte_length=result.final_byte_length,
            mime_type=result.mime_type,
            mipmap=True,
        )
        return (part, delivery)
    return None


def _build_full_resolution_limit_error(path: str, final_bytes: int) -> str:
    return (
        f'"{path}" is {final_bytes} bytes ({format_byte_size(final_bytes)}), '
        f"over the {IMAGE_BYTE_BUDGET}-byte ({format_byte_size(IMAGE_BYTE_BUDGET)}) "
        "per-image limit, so full_resolution cannot be honored. "
        "Use region to view a crop at full fidelity instead."
    )


class Region(BaseModel):
    x: int = Field(ge=0, description="Left edge of the crop, in original-image pixels.")
    y: int = Field(ge=0, description="Top edge of the crop, in original-image pixels.")
    width: int = Field(ge=1, description="Crop width, in original-image pixels.")
    height: int = Field(ge=1, description="Crop height, in original-image pixels.")


class Params(BaseModel):
    file_path: str = Field(
        validation_alias=AliasChoices("file_path", "path"),
        description="Path to the image file, resolved by the filesystem backend. "
        "Relative paths resolve against the working directory; a path outside "
        "the working directory must be absolute. Directories and text files "
        "are not supported.",
    )
    region: Region | None = Field(
        default=None,
        description="Images only: view just this rectangle of the image (original-image "
        "pixel coordinates). Use after a downsampled full view to inspect fine detail — "
        "a region within the size limits is delivered at full fidelity.",
    )
    region_pct: str | None = Field(
        default=None,
        description="Images only: region as percentages instead of pixels. "
        "Format: 'x,y,width,height' where each is 0-100. "
        "Example: '10,10,50,50' for the center half of the image. "
        "Mutually exclusive with `region`.",
    )
    full_resolution: bool | None = Field(
        default=None,
        description="Images only: skip the default downscaling and view at native "
        "resolution. Fails with an explicit error when the payload would exceed the "
        "per-image byte limit; use region for files that large.",
    )
    info_only: bool = Field(
        default=False,
        description="When True, return only image metadata (dimensions, format, size) "
        "without loading the image into context.",
    )
    max_dimension: int | None = Field(
        default=None,
        description="Maximum width/height in pixels. If the image exceeds this, it is "
        "downsampled. Default (None) uses the model's built-in limit.",
    )
    quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="JPEG/WebP quality for compressed output (1-100). Higher = better "
        "quality, larger size.",
    )
    auto_convert: bool = Field(
        default=True,
        description="When True (default), automatically convert unsupported image formats "
        "(AVIF, HEIC, BMP, TIFF) to PNG before sending to the model. "
        "When False, refuse with a conversion command.",
    )

    @model_validator(mode="after")
    def _validate_region(self) -> "Params":
        if self.region is not None and self.region_pct is not None:
            raise ValueError("Specify either `region` or `region_pct`, not both.")
        return self

    @model_validator(mode="after")
    def _validate_video_params(self) -> "Params":
        if self.full_resolution and self.info_only:
            raise ValueError("Cannot set both full_resolution=True and info_only=True.")
        return self


@dataclass(frozen=True)
class _ImageDelivery:
    """How the image payload placed after the summary relates to the file on
    disk. Reported verbatim so the model always knows when it is looking at a
    degraded copy (and how to get the detail back) — silent downsampling
    reads as "the image is just blurry" and quietly degrades the model's
    work."""

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
    # True when mip-map (numpy-based 2x2 bilinear averaging) was used
    # instead of the standard Pillow downsampling path. Mip-map can
    # cause more aggressive detail loss.
    mipmap: bool = False


def _build_media_note(
    *,
    kind: Literal["image", "video"],
    mime_type: str,
    byte_size: int,
    dimensions: tuple[int, int] | None,
    delivery: _ImageDelivery | None,
) -> str:
    """Build the media summary returned as the tool result's model-facing
    message, wrapped in a ``<system>`` block.

    Carries mime type, byte size and (for images) the original pixel
    dimensions, plus the delivery note. When the dimensions are known it
    also guides the model to derive absolute coordinates from that original
    size (crops get offset-mapping guidance instead); it always reminds the
    model to re-read any media it generates or edits.
    """
    parts = [
        f"Read {kind} file.",
        f"Mime type: {mime_type}.",
        f"Size: {byte_size} bytes.",
    ]
    # Coordinate guidance is only emitted when the original size is actually
    # known — sniffing fails for some image formats, and telling the model
    # to use a size that is not in the block would mislead it.
    if kind == "image" and dimensions:
        parts.append(f"Original dimensions: {dimensions[0]}x{dimensions[1]} pixels.")
    if delivery and delivery.kind == "downsampled":
        parts.append(
            f"The attached image was downsampled to {delivery.width}x{delivery.height} "
            f"pixels ({delivery.mime_type}, {format_byte_size(delivery.byte_length)}) "
            "to fit model limits; fine detail may be lost."
        )
        parts.append(
            "To inspect fine detail, call ReadMediaFile again with the region parameter "
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


class ReadMediaFile(CallableTool2[Params]):
    name: str = "read_image"
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        capabilities = runtime.llm.capabilities if runtime.llm else set[str]()
        if "image_in" not in capabilities and "video_in" not in capabilities:
            raise SkipThisTool()

        description = load_desc(
            Path(__file__).parent / "read_media.md",
            {
                "MAX_MEDIA_MEGABYTES": MAX_MEDIA_MEGABYTES,
                "capabilities": capabilities,
            },
        )
        super().__init__(description=description)

        self._runtime = runtime
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._capabilities = capabilities

    async def _validate_path(self, path: KaosPath, raw_path: str) -> ToolError | None:
        """Validate that the path is safe to read."""
        resolved_path = path.canonical()
        original_is_absolute = kaos_path_from_user_input(raw_path).is_absolute()

        if (
            not is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
            and not original_is_absolute
        ):
            # Outside files can only be read with absolute paths
            return ToolError(
                message=(
                    f"`{raw_path}` is not an absolute path. "
                    "You must provide an absolute path to read a file "
                    "outside the working directory."
                ),
                brief="Invalid path",
            )
        return None

    def _os_kind(self) -> str:
        """OS kind of the execution environment, for conversion guidance."""
        return self._runtime.environment.os_kind or self._runtime.builtin_args.KIMI_OS

    async def _read_media(
        self,
        path: KaosPath,
        file_type: FileType,
        params: Params,
        size: int,
    ) -> ToolReturnValue:
        kind: Literal["image", "video"] = "image" if file_type.kind == "image" else "video"
        media_path = str(path)

        data = await path.read_bytes()

        # Info-only mode: return metadata without loading into context
        if params.info_only:
            if kind == "image":
                from PIL import Image as PILImage
                import io
                pil = PILImage.open(io.BytesIO(data))
                dims = f"{pil.width}x{pil.height}"
            else:
                dims = "N/A (video)"
            meta = (
                f"Format: {file_type.mime_type}\n"
                f"Size: {size} bytes\n"
                f"Dimensions: {dims}\n"
            )
            return ToolOk(output=meta, message=meta, brief="Media metadata")

        # The summary always reports the ORIGINAL pixel size and byte size:
        # the model derives relative coordinates and scales them by the
        # original dimensions, so it must see the pre-compression size even
        # when the image part below carries a downsampled copy.
        dimensions: tuple[int, int] | None = None
        delivery: _ImageDelivery | None = None

        if kind == "image":
            sniffed = sniff_image_dimensions(data)
            dimensions = (sniffed.width, sniffed.height) if sniffed else None

            # Resolve region_pct to pixel coordinates if provided
            region = params.region
            if params.region_pct is not None and dimensions is not None:
                parts = params.region_pct.split(",")
                if len(parts) == 4:
                    try:
                        pct_x, pct_y, pct_w, pct_h = map(float, parts)
                        orig_w, orig_h = dimensions
                        region = Region(
                            x=int(orig_w * pct_x / 100.0),
                            y=int(orig_h * pct_y / 100.0),
                            width=max(1, int(orig_w * pct_w / 100.0)),
                            height=max(1, int(orig_h * pct_h / 100.0)),
                        )
                    except (ValueError, ZeroDivisionError):
                        return ToolError(
                            message=f"Invalid region_pct '{params.region_pct}'. "
                            "Format: 'x,y,width,height' with each value 0-100.",
                            brief="Invalid region_pct",
                        )

            # Resolve max_edge: use params.max_dimension if set, else model default
            if params.max_dimension is not None:
                max_edge = params.max_dimension
            else:
                max_edge = resolve_max_image_edge_px()
            read_byte_budget = resolve_read_image_byte_budget()

            if region is not None:
                # Explicit crop: read a rectangle of the original back,
                # typically at full fidelity, so a prior downsampled view
                # can be zoomed into.
                outcome = crop_image_for_model(
                    data,
                    file_type.mime_type,
                    CropRegion(
                        x=region.x,
                        y=region.y,
                        width=region.width,
                        height=region.height,
                    ),
                    skip_resize=bool(params.full_resolution),
                    max_edge=max_edge,
                )
                if not outcome.ok:
                    return ToolError(
                        message=f"Cannot read region from `{params.file_path}`: {outcome.error}",
                        brief="Cannot read region",
                    )
                part = ImageURLPart(
                    image_url=ImageURLPart.ImageURL(
                        url=_to_data_url(outcome.mime_type, outcome.data)
                    )
                )
                wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
                delivery = _ImageDelivery(
                    kind="crop",
                    width=outcome.width,
                    height=outcome.height,
                    byte_length=outcome.final_byte_length,
                    mime_type=outcome.mime_type,
                    region=outcome.region,
                    resized=outcome.resized,
                )
                # The decode is authoritative: it covers formats and
                # nonconforming EXIF the header sniff cannot read, and
                # region coordinates live in the decoded space, so the note
                # must report it.
                dimensions = (outcome.original_width, outcome.original_height)
            elif params.full_resolution:
                # Native resolution on request — but the provider's
                # per-image byte ceiling is a hard limit, so refuse
                # explicitly rather than degrade.
                if len(data) > IMAGE_BYTE_BUDGET:
                    return ToolError(
                        message=_build_full_resolution_limit_error(params.file_path, len(data)),
                        brief="Image too large",
                    )
                part = ImageURLPart(
                    image_url=ImageURLPart.ImageURL(
                        url=_to_data_url(file_type.mime_type, data)
                    )
                )
                wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
                delivery = _ImageDelivery(
                    kind="full",
                    width=dimensions[0] if dimensions else 0,
                    height=dimensions[1] if dimensions else 0,
                    byte_length=len(data),
                    mime_type=file_type.mime_type,
                )
            else:
                # Shrink oversized images so a large screenshot neither
                # wastes context tokens nor trips the provider's per-image
                # byte ceiling. The compressor is best-effort and may
                # return the original bytes after a safety guard or codec
                # failure, so enforce both delivery limits before creating
                # any model-visible media part.
                compressed = compress_image_for_model(
                    data,
                    file_type.mime_type,
                    max_edge=max_edge,
                    byte_budget=read_byte_budget,
                )
                if (
                    compressed.final_byte_length > read_byte_budget
                    or max(compressed.width, compressed.height) > max_edge
                ):
                    # The normal compressor could not meet budgets — try the
                    # mip-map fallback (numpy-based 2×2 bilinear averaging).
                    mip_result = _try_mipmap_fallback(
                        data,
                        file_type.mime_type,
                        media_path,
                        max_edge,
                        read_byte_budget,
                        dimensions,
                    )
                    if mip_result is not None:
                        part, delivery = mip_result
                        wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
                        # The mipmap decode dimensions are authoritative.
                        dimensions = (
                            compressed.original_width,
                            compressed.original_height,
                        )
                    else:
                        return ToolError(
                            message=_build_image_delivery_limit_error(
                                compressed.final_byte_length, read_byte_budget, max_edge
                            ),
                            brief="Image too large",
                        )
                else:
                    part = ImageURLPart(
                        image_url=ImageURLPart.ImageURL(
                            url=_to_data_url(compressed.mime_type, compressed.data)
                        )
                    )
                    wrapped = wrap_media_part(part, tag="image", attrs={"path": media_path})
                    delivery = _ImageDelivery(
                        kind="downsampled" if compressed.changed else "untouched",
                        width=compressed.width,
                        height=compressed.height,
                        byte_length=compressed.final_byte_length,
                        mime_type=compressed.mime_type,
                    )
                    if compressed.changed:
                        # Same as the crop path: once a decode happened, its
                        # dimensions are authoritative over the header sniff.
                        dimensions = (compressed.original_width, compressed.original_height)
        else:
            from kosong.chat_provider.kimi import Kimi

            if (llm := self._runtime.llm) and isinstance(llm.chat_provider, Kimi):
                part = await llm.chat_provider.files.upload_video(
                    data=data,
                    mime_type=file_type.mime_type,
                )
                wrapped = wrap_media_part(part, tag="video", attrs={"path": media_path})
            else:
                data_url = _to_data_url(file_type.mime_type, data)
                part = VideoURLPart(video_url=VideoURLPart.VideoURL(url=data_url))
                wrapped = wrap_media_part(part, tag="video", attrs={"path": media_path})

        note = _build_media_note(
            kind=kind,
            mime_type=file_type.mime_type,
            byte_size=size,
            dimensions=dimensions,
            delivery=delivery,
        )

        # Prepend delivery summary to output so the model sees it before the image
        if delivery is not None:
            preview = (
                f"[Image: {delivery.kind}, {delivery.width}x{delivery.height}, "
                f"{delivery.byte_length} bytes]\n"
            )
            if isinstance(wrapped, list):
                wrapped = [TextPart(text=preview)] + wrapped
            else:
                wrapped = f"{preview}{wrapped}"

        return ToolOk(output=wrapped, message=note)

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if not params.file_path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )

        try:
            p = kaos_path_from_tool_input(params.file_path, self._work_dir)
            if err := await self._validate_path(p, params.file_path):
                return err
            p = p.canonical()

            if not await p.exists():
                return ToolError(
                    message=f"`{params.file_path}` does not exist.",
                    brief="File not found",
                )
            if not await p.is_file():
                return ToolError(
                    message=f"`{params.file_path}` is not a file.",
                    brief="Invalid path",
                )

            # For media input, the bytes are authoritative; the extension is
            # only a fallback for formats that cannot be sniffed from the
            # header.
            header = await p.read_bytes(MEDIA_SNIFF_BYTES)
            file_type = detect_file_type(str(p), header=header)
            if file_type.kind == "text":
                return ToolError(
                    message=f"`{params.file_path}` is a text file. Use read to read text files.",
                    brief="Unsupported file type",
                )
            if file_type.kind == "unknown":
                return ToolError(
                    message=(
                        f"`{params.file_path}` seems not readable as an image or video file. "
                        "You may need to read it with proper shell commands, Python tools "
                        "or MCP tools if available. "
                        "If you read/operate it with Python, you MUST ensure that any "
                        "third-party packages are installed in a virtual environment (venv)."
                    ),
                    brief="File not readable",
                )

            if file_type.kind == "image" and "image_in" not in self._capabilities:
                return ToolError(
                    message=(
                        "The current model does not support image input. "
                        "Tell the user to use a model with image input capability."
                    ),
                    brief="Unsupported media type",
                )
            # Formats outside the provider-accepted set (AVIF, HEIC, BMP,
            # TIFF, ICO, …) must never reach the model: once the image_url
            # lands in the history every subsequent request in the session
            # is rejected. Refuse with a conversion command for the
            # execution environment instead — the model can run it through
            # Shell (under the normal permission flow) and read the
            # converted file.
            if file_type.kind == "image" and not is_model_accepted_image_mime(
                file_type.mime_type
            ):
                if params.auto_convert:
                    # Auto-convert unsupported format to PNG
                    from PIL import Image as PILImage
                    import io
                    try:
                        full_data = await p.read_bytes()
                        pil_image = PILImage.open(io.BytesIO(full_data))
                        converted_buf = io.BytesIO()
                        pil_image.save(converted_buf, format="PNG")
                        # Write converted data to a temp file next to the original
                        converted_path_str = str(p) + ".converted.png"
                        import anyio
                        async with await anyio.open_file(converted_path_str, 'wb') as f:
                            await f.write(converted_buf.getvalue())
                        p = KaosPath(converted_path_str)
                        file_type.mime_type = "image/png"
                        file_type.kind = "image"
                    except Exception as conv_e:
                        return ToolError(
                            message=f"Failed to convert `{params.file_path}` to PNG: {conv_e}. "
                            + build_image_conversion_guidance(
                                params.file_path, file_type.mime_type, self._os_kind()
                            ),
                            brief="Conversion failed",
                        )
                else:
                    return ToolError(
                        message=build_image_conversion_guidance(
                            params.file_path, file_type.mime_type, self._os_kind()
                        ),
                        brief="Unsupported image format",
                    )
            if file_type.kind == "video" and "video_in" not in self._capabilities:
                return ToolError(
                    message=(
                        "The current model does not support video input. "
                        "Tell the user to use a model with video input capability."
                    ),
                    brief="Unsupported media type",
                )

            stat = await p.stat()
            size = stat.st_size
            if size == 0:
                return ToolError(
                    message=f"`{params.file_path}` is empty.",
                    brief="Empty file",
                )
            if size > (MAX_MEDIA_MEGABYTES << 20):
                return ToolError(
                    message=(
                        f"`{params.file_path}` is {size} bytes, which exceeds the max "
                        f"{MAX_MEDIA_MEGABYTES}MB bytes for media files."
                    ),
                    brief="File too large",
                )

            if file_type.kind == "video" and (
                params.region is not None or params.full_resolution
            ):
                return ToolError(
                    message="region and full_resolution apply only to image files.",
                    brief="Invalid parameters",
                )

            if (
                file_type.kind == "image"
                and size > MAX_IMAGE_DECODE_BYTES
                and (params.region is not None or params.full_resolution)
            ):
                return ToolError(
                    message=_build_image_decode_limit_error(size),
                    brief="Image too large",
                )

            if (
                file_type.kind == "image"
                and params.region is None
                and params.full_resolution
                and size > IMAGE_BYTE_BUDGET
            ):
                return ToolError(
                    message=_build_full_resolution_limit_error(params.file_path, size),
                    brief="Image too large",
                )

            return await self._read_media(p, file_type, params, size)
        except Exception as e:
            logger.warning("ReadMediaFile failed: {path}: {error}", path=params.file_path, error=e)
            return ToolError(
                message=f"Failed to read {params.file_path}. Error: {e}",
                brief="Failed to read file",
            )
