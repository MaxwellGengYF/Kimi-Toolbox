"""Tests for the ReadMediaFile tool."""

from __future__ import annotations

import base64
import random
from io import BytesIO
from typing import cast

import pytest
from inline_snapshot import snapshot
from kaos.path import KaosPath
from PIL import Image

import kimi_cli.tools.file.read_media as read_media_module
from kimi_cli.llm import ModelCapability
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.file.read_media import Params, ReadMediaFile, Region
from kimi_cli.utils.image_compress import sniff_image_dimensions
from kimi_cli.wire.types import ImageURLPart, TextPart, VideoURLPart


def _make_png(size: tuple[int, int], color: tuple[int, int, int] = (51, 102, 204)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_noisy_png(size: tuple[int, int]) -> bytes:
    """True-random pixels — incompressible, so byte budgets are really hit."""
    width, height = size
    rng = random.Random(42)
    img = Image.frombytes("RGB", size, rng.randbytes(width * height * 3))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _part_dimensions(part: ImageURLPart) -> tuple[int, int]:
    url = part.image_url.url
    payload = url.split(",", 1)[1]
    dims = sniff_image_dimensions(base64.b64decode(payload))
    assert dims is not None
    return (dims.width, dims.height)


async def test_read_image_file(read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath):
    """Test reading an image file."""
    image_file = temp_work_dir / "sample.png"
    data = b"\x89PNG\r\n\x1a\n" + b"pngdata"
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    assert isinstance(result.output, list)
    # output is [TextPart(preview), TextPart('<image path=...'), ImageURLPart, TextPart('</image>')]
    assert len(result.output) == 4
    assert result.output[0].text.startswith("[Image:")
    assert result.output[1] == TextPart(text=f'<image path="{image_file}">')
    assert result.output[3] == TextPart(text="</image>")
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    assert part.image_url.url.startswith("data:image/png;base64,")
    assert result.message == snapshot(
        "<system>Read image file. Mime type: image/png. Size: 15 bytes. "
        "If you generate or edit images or videos via commands or scripts, "
        "read the result back immediately before continuing.</system>"
    )


async def test_read_extensionless_image_file(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """Test reading an extensionless image file."""
    image_file = temp_work_dir / "sample"
    data = b"\x89PNG\r\n\x1a\n" + b"pngdata"
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    assert isinstance(result.output, list)
    assert len(result.output) == 4
    # Index 0 = preview, 1 = <image>, 2 = ImageURLPart, 3 = </image>
    assert result.output[1] == TextPart(text=f'<image path="{image_file}">')
    assert result.output[3] == TextPart(text="</image>")
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    assert part.image_url.url.startswith("data:image/png;base64,")
    assert result.message == snapshot(
        "<system>Read image file. Mime type: image/png. Size: 15 bytes. "
        "If you generate or edit images or videos via commands or scripts, "
        "read the result back immediately before continuing.</system>"
    )


async def test_read_image_file_with_size(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """Test reading an image file with detectable dimensions."""
    image_file = temp_work_dir / "valid.png"
    data = _make_png((3, 4), color=(0, 0, 0))
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    assert result.message == snapshot(
        f"<system>Read image file. Mime type: image/png. Size: {len(data)} bytes. "
        "Original dimensions: 3x4 pixels. "
        "If you need to output coordinates, output relative coordinates first and "
        "compute absolute coordinates using the original image size. "
        "If you generate or edit images or videos via commands or scripts, "
        "read the result back immediately before continuing.</system>"
    )


async def test_untouched_small_image_does_not_claim_downsampling(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """A small image sent untouched must not claim downsampling."""
    image_file = temp_work_dir / "small.png"
    await image_file.write_bytes(_make_png((30, 20)))

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    assert "downsampled" not in result.message.lower()
    # media part is at index 2 (index 0 = preview, 1 = <image>, 2 = ImageURLPart)
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    assert _part_dimensions(part) == (30, 20)


async def test_read_large_image_is_downsampled(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """Large images are downsampled and the note points to region readback."""
    image_file = temp_work_dir / "big.png"
    data = _make_png((2200, 2200))
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    message = result.message
    assert message.startswith("<system>") and message.endswith("</system>")
    assert f"Size: {len(data)} bytes." in message
    assert "Original dimensions: 2200x2200 pixels." in message
    assert "The attached image was downsampled to 2000x2000 pixels" in message
    assert "fine detail may be lost" in message
    assert "call read_image again with the region parameter" in message
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    width, height = _part_dimensions(part)
    assert max(width, height) <= 2000


async def test_read_image_region_at_native_resolution(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """A region within the limits is delivered at full fidelity."""
    image_file = temp_work_dir / "big.png"
    await image_file.write_bytes(_make_png((2100, 2100)))

    result = await read_media_file_tool(
        Params(path=str(image_file), region=Region(x=100, y=50, width=400, height=300))
    )

    assert not result.is_error
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    assert _part_dimensions(part) == (400, 300)
    message = result.message
    assert "Original dimensions: 2100x2100 pixels." in message
    assert "Showing region (x=100, y=50, width=400, height=300) of the original image" in message
    assert "at native resolution" in message
    assert "add the region offset (x=100, y=50)" in message


async def test_read_image_region_out_of_bounds(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """An out-of-bounds region fails and names the original size."""
    image_file = temp_work_dir / "big.png"
    await image_file.write_bytes(_make_png((2100, 2100)))

    result = await read_media_file_tool(
        Params(path=str(image_file), region=Region(x=5000, y=0, width=100, height=100))
    )

    assert result.is_error
    assert result.message.startswith(f"Cannot read region from `{image_file}`: ")
    assert "lies outside the 2100x2100 image." in result.message


async def test_read_image_full_resolution_served_under_budget(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """full_resolution serves the raw bytes when they fit the per-image budget."""
    image_file = temp_work_dir / "big.png"
    data = _make_png((2100, 1050))
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file), full_resolution=True))

    assert not result.is_error
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    expected_url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    assert part.image_url.url == expected_url
    assert "Shown at native resolution; no downscaling applied." in result.message


async def test_read_image_full_resolution_over_budget_refused(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """full_resolution refuses explicitly when the payload exceeds the budget."""
    image_file = temp_work_dir / "huge.png"
    data = _make_noisy_png((2200, 1100))
    assert len(data) > 3932160  # over the 3.75 MB per-image budget
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file), full_resolution=True))

    assert result.is_error
    assert f'"{image_file}" is {len(data)} bytes' in result.message
    assert "over the 3932160-byte (3.8 MB) per-image limit" in result.message
    assert "full_resolution cannot be honored" in result.message
    assert "Use region to view a crop at full fidelity instead." in result.message


async def test_read_image_decode_cap_mipmap_fallback(
    read_media_file_tool: ReadMediaFile,
    temp_work_dir: KaosPath,
    monkeypatch: pytest.MonkeyPatch,
):
    """Default reads over the safe decode allocation now fall back to
    mip-map downsampling instead of erroring."""
    image_file = temp_work_dir / "sample.png"
    data = _make_png((100, 100))
    await image_file.write_bytes(data)
    assert len(data) > 100

    monkeypatch.setattr(read_media_module, "MAX_IMAGE_DECODE_BYTES", 100)
    monkeypatch.setenv("KIMI_IMAGE_READ_BYTE_BUDGET", "100")
    monkeypatch.delenv("KIMI_IMAGE_MAX_EDGE_PX", raising=False)

    result = await read_media_file_tool(Params(path=str(image_file)))

    # The mipmap fallback should succeed - 100x100 down to ~25x25 fits 100 bytes.
    assert not result.is_error
    assert "downsampled" in result.message
    part = result.output[2]
    assert isinstance(part, ImageURLPart)


async def test_large_image_falls_back_to_mipmap(
    read_media_file_tool: ReadMediaFile,
    temp_work_dir: KaosPath,
    monkeypatch: pytest.MonkeyPatch,
):
    """A large noisy image that would normally error now succeeds via mipmap fallback."""
    image_file = temp_work_dir / "large_noisy.png"
    data = _make_noisy_png((4000, 3000))
    assert len(data) > 256 * 1024  # over READ_IMAGE_BYTE_BUDGET
    await image_file.write_bytes(data)

    monkeypatch.setenv("KIMI_IMAGE_READ_BYTE_BUDGET", "256000")
    monkeypatch.delenv("KIMI_IMAGE_MAX_EDGE_PX", raising=False)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    assert "downsampled" in result.message
    part = result.output[2]
    assert isinstance(part, ImageURLPart)
    # The mipmap should have shrunk it — output should be ≤ 2000 edge
    url = part.image_url.url
    payload = url.split(",", 1)[1]
    dims = sniff_image_dimensions(base64.b64decode(payload))
    assert dims is not None
    assert max(dims.width, dims.height) <= 2000


async def test_large_image_mipmap_note(
    read_media_file_tool: ReadMediaFile,
    temp_work_dir: KaosPath,
    monkeypatch: pytest.MonkeyPatch,
):
    """The <system> message indicates downsampling when mipmap fallback is used."""
    image_file = temp_work_dir / "big_for_mipmap.png"
    data = _make_noisy_png((4000, 3000))
    await image_file.write_bytes(data)

    monkeypatch.setenv("KIMI_IMAGE_READ_BYTE_BUDGET", "256000")
    monkeypatch.delenv("KIMI_IMAGE_MAX_EDGE_PX", raising=False)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert not result.is_error
    message = result.message
    assert message.startswith("<system>") and message.endswith("</system>")
    assert "Original dimensions: 4000x3000 pixels." in message
    assert "The attached image was downsampled to" in message
    assert "fine detail may be lost" in message
    assert "call read_image again with the region parameter" in message


async def test_extremely_large_image_still_errors(
    read_media_file_tool: ReadMediaFile,
    temp_work_dir: KaosPath,
):
    """An image over MAX_MEDIA_MEGABYTES (100 MB) still errors."""
    image_file = temp_work_dir / "huge_fake.png"
    # Write a minimal valid PNG header with a large enough IHDR that
    # the file is just over 100 MB of actual bytes.
    one_mb = b"x" * (1024 * 1024)
    data = b"\x89PNG\r\n\x1a\n" + one_mb * 101  # ~101 MB
    # The header won't parse as a real PNG, but the size gate fires first.
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert result.is_error
    assert "exceeds the max 100MB" in result.message


async def test_read_image_decode_cap_precheck_region(
    read_media_file_tool: ReadMediaFile,
    temp_work_dir: KaosPath,
    monkeypatch: pytest.MonkeyPatch,
):
    """region/full_resolution sources over the safe decode allocation fail early."""
    image_file = temp_work_dir / "sample.png"
    data = _make_png((100, 100))
    await image_file.write_bytes(data)

    monkeypatch.setattr(read_media_module, "MAX_IMAGE_DECODE_BYTES", 100)

    read_calls: list[int | None] = []
    original_read_bytes = KaosPath.read_bytes

    async def spy_read_bytes(self: KaosPath, n: int | None = None) -> bytes:
        read_calls.append(n)
        return await original_read_bytes(self, n)

    monkeypatch.setattr(KaosPath, "read_bytes", spy_read_bytes)

    result = await read_media_file_tool(
        Params(path=str(image_file), region=Region(x=0, y=0, width=10, height=10))
    )

    assert result.is_error
    assert result.message == (
        "Image is too large to process safely for region or full_resolution "
        f"({len(data)} bytes; safe decode limit 100 bytes). "
        "The original image was not sent to the model. Do not retry the same file unchanged. "
        "Use Bash or an available image-processing tool to create a smaller copy or crop the "
        "needed region into a separate image, then call read_image on the resulting file."
    )
    assert read_calls == [512]


@pytest.mark.parametrize(
    ("filename", "magic", "mime"),
    [
        ("photo.heic", b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heic\x00\x00\x00\x00", "image/heic"),
        ("photo.bmp", b"BM" + b"\x00" * 64, "image/bmp"),
        ("photo.tiff", b"II*\x00" + b"\x00" * 64, "image/tiff"),
    ],
)
async def test_read_unsupported_image_format_refused_with_guidance(
    read_media_file_tool: ReadMediaFile,
    temp_work_dir: KaosPath,
    filename: str,
    magic: bytes,
    mime: str,
):
    """Formats the provider rejects are refused with conversion guidance."""
    image_file = temp_work_dir / filename
    await image_file.write_bytes(magic)

    result = await read_media_file_tool(Params(path=str(image_file), auto_convert=False))

    assert result.is_error
    assert result.brief == "Unsupported image format"
    assert f"is an {mime} image, which the provider does not accept." in result.message
    assert "Convert it to JPEG first, then read the converted file." in result.message
    converted = str(image_file).rsplit(".", 1)[0] + ".jpg"
    assert converted in result.message


async def test_read_video_file(read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath):
    """Test reading a video file."""
    video_file = temp_work_dir / "sample.mp4"
    data = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    await video_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(video_file)))

    assert not result.is_error
    assert isinstance(result.output, list)
    assert len(result.output) == 3
    assert result.output[0] == TextPart(text=f'<video path="{video_file}">')
    assert result.output[2] == TextPart(text="</video>")
    part = result.output[1]
    assert isinstance(part, VideoURLPart)
    assert part.video_url.url.startswith("data:video/mp4;base64,")
    assert result.message == snapshot(
        f"<system>Read video file. Mime type: video/mp4. Size: {len(data)} bytes. "
        "If you generate or edit images or videos via commands or scripts, "
        "read the result back immediately before continuing.</system>"
    )


async def test_read_video_file_with_region_or_full_resolution_rejected(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """region and full_resolution apply only to image files."""
    video_file = temp_work_dir / "sample.mp4"
    data = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    await video_file.write_bytes(data)

    with_region = await read_media_file_tool(
        Params(path=str(video_file), region=Region(x=0, y=0, width=10, height=10))
    )
    assert with_region.is_error
    assert with_region.message == "region and full_resolution apply only to image files."
    assert with_region.brief == "Invalid parameters"

    with_full_resolution = await read_media_file_tool(
        Params(path=str(video_file), full_resolution=True)
    )
    assert with_full_resolution.is_error
    assert with_full_resolution.message == "region and full_resolution apply only to image files."
    assert with_full_resolution.brief == "Invalid parameters"


async def test_read_text_file(read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath):
    """Test reading a text file with ReadMediaFile."""
    text_file = temp_work_dir / "sample.txt"
    await text_file.write_text("hello")

    result = await read_media_file_tool(Params(path=str(text_file)))

    assert result.is_error
    assert result.message == snapshot(
        f"`{text_file}` is a text file. Use read to read text files."
    )
    assert result.brief == snapshot("Unsupported file type")


async def test_read_video_file_without_capability(runtime: Runtime, temp_work_dir: KaosPath):
    """Test reading a video file without video capability."""
    assert runtime.llm is not None
    runtime.llm.capabilities = cast(set[ModelCapability], {"image_in"})
    read_media_file_tool = ReadMediaFile(runtime)

    video_file = temp_work_dir / "sample.mp4"
    data = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    await video_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(video_file)))

    assert result.is_error
    assert result.message == snapshot(
        "The current model does not support video input. "
        "Tell the user to use a model with video input capability."
    )
    assert result.brief == snapshot("Unsupported media type")


async def test_read_image_file_without_capability(runtime: Runtime, temp_work_dir: KaosPath):
    """Test reading an image file without image capability."""
    assert runtime.llm is not None
    runtime.llm.capabilities = cast(set[ModelCapability], {"video_in"})
    read_media_file_tool = ReadMediaFile(runtime)

    image_file = temp_work_dir / "sample.png"
    await image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pngdata")

    result = await read_media_file_tool(Params(path=str(image_file)))

    assert result.is_error
    assert result.message == snapshot(
        "The current model does not support image input. "
        "Tell the user to use a model with image input capability."
    )
    assert result.brief == snapshot("Unsupported media type")


# ── info_only tests ─────────────────────────────────────────────────────────


async def test_info_only_image(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """info_only with an image file returns metadata without loading into context."""
    image_file = temp_work_dir / "info_sample.png"
    data = _make_png((32, 16))
    await image_file.write_bytes(data)

    result = await read_media_file_tool(Params(path=str(image_file), info_only=True))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "Format:" in result.output
    assert "Size:" in result.output
    assert "Dimensions:" in result.output
    assert "32x16" in result.output
    assert result.brief == "Media metadata"


async def test_info_only_video(
    read_media_file_tool: ReadMediaFile, temp_work_dir: KaosPath
):
    """info_only with a video file should NOT crash (PIL cannot open video)."""
    video_file = temp_work_dir / "info_sample.mp4"
    data = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    await video_file.write_bytes(data)

    # Prior to fix, this would crash with PIL.UnidentifiedImageError
    result = await read_media_file_tool(Params(path=str(video_file), info_only=True))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "Format:" in result.output
    assert "Size:" in result.output
    assert "Dimensions:" in result.output
    assert "N/A (video)" in result.output
    assert result.brief == "Media metadata"
