Read a UTF-8 text file and return line-numbered content.

`file_path` may be a single file path or a list of paths (aliases: `path`). `offset` (alias `line_offset`) and `limit` (alias `n_lines`) may each be a single value applied to all files, or a list with one value per file path. Lines over ${MAX_LINE_LENGTH} chars truncated. Max ${MAX_LINES} lines per file. Bytes per file scale with the model's context window (at least ${MAX_BYTES} bytes, up to 1MiB). Negative offset = tail mode.

Each `file_path` may also be a glob pattern such as `./*.md` to read all matching files in a directory. The total number of files read in one call cannot exceed ${MAX_FILES}.

Prefer `glob` to find files by name, `grep` for content search, then read the paths found.
