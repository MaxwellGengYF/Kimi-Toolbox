from __future__ import annotations

from typing import cast

# ruff: noqa

import pytest
from inline_snapshot import snapshot

from kimi_cli.llm import ModelCapability
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.file.read_media import ReadMediaFile


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (
            {"image_in", "video_in"},
            snapshot(
                """\
Read a PNG/JPEG/WebP/GIF file and return the image itself. Requires the current model to accept image input. Max size: 100MB.

**Tips:**
- A `<system>` tag accompanies the media: mime type, byte size, and for images the original pixel dimensions, plus delivery mode (untouched, downsampled, cropped, or native). Report coordinates relative to the original size, never the displayed copy. After generating or editing media, read the result back before continuing.
- Large images are downsampled by default, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need detail, re-read with `region` (original-image pixel coordinates) for a full-fidelity crop, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely fit model limits, the tool errors and does not send the original. Resize via Shell or an image-processing tool, then read the copy — do not retry the unchanged file.
- Only image/video files. For text files use `read`; to list directories use `ls` via Shell, or `glob` for pattern search.

**Capabilities**
- This tool supports image and video files for the current model.
"""
            ),
        ),
        (
            {"image_in"},
            snapshot(
                """\
Read a PNG/JPEG/WebP/GIF file and return the image itself. Requires the current model to accept image input. Max size: 100MB.

**Tips:**
- A `<system>` tag accompanies the media: mime type, byte size, and for images the original pixel dimensions, plus delivery mode (untouched, downsampled, cropped, or native). Report coordinates relative to the original size, never the displayed copy. After generating or editing media, read the result back before continuing.
- Large images are downsampled by default, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need detail, re-read with `region` (original-image pixel coordinates) for a full-fidelity crop, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely fit model limits, the tool errors and does not send the original. Resize via Shell or an image-processing tool, then read the copy — do not retry the unchanged file.
- Only image/video files. For text files use `read`; to list directories use `ls` via Shell, or `glob` for pattern search.

**Capabilities**
- This tool supports image files for the current model.
- Video files are not supported by the current model.
"""
            ),
        ),
        (
            {"video_in"},
            snapshot(
                """\
Read a PNG/JPEG/WebP/GIF file and return the image itself. Requires the current model to accept image input. Max size: 100MB.

**Tips:**
- A `<system>` tag accompanies the media: mime type, byte size, and for images the original pixel dimensions, plus delivery mode (untouched, downsampled, cropped, or native). Report coordinates relative to the original size, never the displayed copy. After generating or editing media, read the result back before continuing.
- Large images are downsampled by default, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need detail, re-read with `region` (original-image pixel coordinates) for a full-fidelity crop, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely fit model limits, the tool errors and does not send the original. Resize via Shell or an image-processing tool, then read the copy — do not retry the unchanged file.
- Only image/video files. For text files use `read`; to list directories use `ls` via Shell, or `glob` for pattern search.

**Capabilities**
- This tool supports video files for the current model.
- Image files are not supported by the current model.
"""
            ),
        ),
    ],
)
def test_read_media_file_description_by_capabilities(
    runtime: Runtime, capabilities: set[str], expected: str
) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = cast(set[ModelCapability], capabilities)
    assert ReadMediaFile(runtime).base.description == expected


def test_read_media_file_description_without_capabilities(runtime: Runtime) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = cast(set[ModelCapability], set())
    with pytest.raises(SkipThisTool):
        ReadMediaFile(runtime)
