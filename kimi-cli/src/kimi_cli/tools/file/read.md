Read a UTF-8 text file and return line-numbered content.
`file_path` may be a single path or a list of paths; `offset` and `limit` may each be a single value or one per file. Lines over ${MAX_LINE_LENGTH} chars truncated; max ${MAX_LINES} lines per file. Bytes per file scale with the model's context window (at least ${MAX_BYTES} bytes, up to 1MiB). Negative offset = tail mode.
Each `file_path` may also be a glob pattern such as `./*.md` to read all matching files in a directory (max ${MAX_FILES} files per call).
Prefer `glob` to find files by name, `grep` for content search, then read the paths found.
