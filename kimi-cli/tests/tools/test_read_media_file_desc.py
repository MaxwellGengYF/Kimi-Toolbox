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
Read media content from a file.

**Tips:**
- A `<system>` tag accompanies the media content; it summarizes the mime type, byte size and, for images, the original pixel dimensions, and states how the image was delivered (untouched, downsampled, cropped, or native resolution). When outputting coordinates, give relative coordinates first and compute absolute coordinates from the original image size (never measure the displayed copy). After generating or editing media, read the result back before continuing.
- Large images are downsampled by default when they can safely fit model limits, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need that detail, call this tool again with `region` (original-image pixel coordinates) to view a crop at full fidelity, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely produce an image within model limits, the tool returns an error and does not send the original. Use Shell or an available image-processing tool to create a smaller copy, then read that copy. Do not retry the unchanged file.
- This tool reads only image or video files. For text files use ReadFile; to list directories use `ls` via Shell, or Glob for pattern search.
- The maximum size that can be read is 100MB.

**Capabilities**
- This tool supports image and video files for the current model.
"""
            ),
        ),
        (
            {"image_in"},
            snapshot(
                """\
Read media content from a file.

**Tips:**
- A `<system>` tag accompanies the media content; it summarizes the mime type, byte size and, for images, the original pixel dimensions, and states how the image was delivered (untouched, downsampled, cropped, or native resolution). When outputting coordinates, give relative coordinates first and compute absolute coordinates from the original image size (never measure the displayed copy). After generating or editing media, read the result back before continuing.
- Large images are downsampled by default when they can safely fit model limits, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need that detail, call this tool again with `region` (original-image pixel coordinates) to view a crop at full fidelity, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely produce an image within model limits, the tool returns an error and does not send the original. Use Shell or an available image-processing tool to create a smaller copy, then read that copy. Do not retry the unchanged file.
- This tool reads only image or video files. For text files use ReadFile; to list directories use `ls` via Shell, or Glob for pattern search.
- The maximum size that can be read is 100MB.

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
Read media content from a file.

**Tips:**
- A `<system>` tag accompanies the media content; it summarizes the mime type, byte size and, for images, the original pixel dimensions, and states how the image was delivered (untouched, downsampled, cropped, or native resolution). When outputting coordinates, give relative coordinates first and compute absolute coordinates from the original image size (never measure the displayed copy). After generating or editing media, read the result back before continuing.
- Large images are downsampled by default when they can safely fit model limits, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need that detail, call this tool again with `region` (original-image pixel coordinates) to view a crop at full fidelity, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely produce an image within model limits, the tool returns an error and does not send the original. Use Shell or an available image-processing tool to create a smaller copy, then read that copy. Do not retry the unchanged file.
- This tool reads only image or video files. For text files use ReadFile; to list directories use `ls` via Shell, or Glob for pattern search.
- The maximum size that can be read is 100MB.

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
