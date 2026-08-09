Read media content from a file.

**Tips:**
- A `<system>` tag accompanies the media content; it summarizes the mime type, byte size and, for images, the original pixel dimensions, and states how the image was delivered (untouched, downsampled, cropped, or native resolution). When outputting coordinates, give relative coordinates first and compute absolute coordinates from the original image size. After generating or editing media via commands or scripts, read the result back before continuing.
- Large images are downsampled by default when automatic compression can safely fit them within model limits, which can blur fine detail (small text, dense UI). Compute absolute coordinates from the original dimensions reported in the `<system>` block, never by measuring the displayed copy. When the `<system>` tag reports downsampling and you need that detail, call this tool again with the `region` parameter (original-image pixel coordinates) to view a crop at full fidelity, or set `full_resolution` to true when the whole file fits the per-image byte limit. Re-reading the same file without these parameters just reproduces the same downsampled image.
- If automatic compression cannot safely produce an image within model limits, the tool returns an error and does not send the original image. Follow the error: use Shell or an available image-processing tool to create a smaller copy, then read that copy. Do not retry the unchanged file.
- This tool can only read image or video files. To read text files, use the ReadFile tool. To list directories, use `ls` via Shell for a known directory, or Glob for pattern search.
- If the file doesn't exist or path is invalid, an error will be returned.
- The maximum size that can be read is ${MAX_MEDIA_MEGABYTES}MB. An error will be returned if the file is larger than this limit.

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
