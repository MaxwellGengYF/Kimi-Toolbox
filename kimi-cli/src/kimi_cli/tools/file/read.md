Read a UTF-8 text file and return line-numbered content.
`file_path`: single path or list; `offset`/`limit`: scalar or one per file. Lines over ${MAX_LINE_LENGTH} chars truncated; max ${MAX_LINES} lines per file; bytes scale with the model's context window (at least ${MAX_BYTES}, up to 1MiB). Negative offset = tail mode. A `file_path` glob (e.g. `./*.md`) reads up to ${MAX_FILES} matching files. Prefer `glob`/`grep` to find/search, then `read` the paths.

Rich formats (one per call; scalar params apply to every file in a multi-file read):
- Archives (zip/jar/war/apk/whl/cbz, tar/tgz/tbz2/txz, bare gz/bz2/xz): `read data.zip` lists up to 500 root entries; `archive_member="src/main.py"` reads one member as text. Traversal (`..`, absolute, backslash) rejected; binary members get an explicit notice.
- SQLite (.sqlite/.sqlite3/.db/.db3): `read app.db` lists tables with counts; `sql_table`/`sql_where`/`sql_order`/`sql_limit`/`sql_offset` paginate rows; `sql_query` runs raw read-only SELECT (≤1000 rows; rejects `;`, comments, LIMIT/UNION/ATTACH).
- PDF screenshots: `read doc.pdf` with `pdf_page=3` renders page 3 as PNG (requires `image_in`); DPI falls back 150→96→72 on over-budget.
- Document markdown: `render_markdown=True` converts .docx to markdown (headings, code fences, pipe tables) and .md/.html to clean text; False uses the legacy plain-text extractor.
- Profiles: `read *.cpuprofile` / `*.sample.txt` returns a compact bottleneck summary (hot paths, top-20 self time, idle excluded); `profile_raw=True` returns raw JSON/text.
