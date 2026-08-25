Read a PNG/JPEG/WebP/GIF file and return the image itself. Requires the current model to accept image input. Max size: ${MAX_MEDIA_MEGABYTES}MB.

**Tips:**
- A `<system>` tag accompanies the media: mime type, byte size, and for images the original pixel dimensions, plus delivery mode (untouched, downsampled, cropped, or native resolution). Report coordinates relative to the original image size (never the displayed copy). After generating or editing media, read the result back before continuing.
- Large images are downsampled by default, which can blur fine detail (small text, dense UI). When the `<system>` tag reports downsampling and you need detail, re-read with `region` (original-image pixel coordinates) for a full-fidelity crop, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely fit model limits, the tool errors and does not send the original. Resize via Shell or an image-processing tool, then read the copy — do not retry the unchanged file.
- Only image/video files. For text files use `read`; to list directories use `ls` via Shell, or `glob` for pattern search.

**Capabilities**
{% if "image_in" in capabilities and "video_in" in capabilities %}
- This tool supports image and video files for the current model.
{% elif "image_in" in capabilities %}
- This tool supports image files for the current model.
- Video files are not supported by the current model.
{% elif "video_in" in capabilities %}
- This tool supports video files for the current model.
- Image files are not supported by the current model.
{% else %}
- The current model does not support image or video input.
{% endif %}
