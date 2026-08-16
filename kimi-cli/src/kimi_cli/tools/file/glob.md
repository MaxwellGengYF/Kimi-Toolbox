Find files whose paths match a glob pattern. Returns matching file paths — never directories — including hidden and ignored files (VCS metadata directories are excluded). Results come in modification-time order: up to 100 paths, or the first 100 with a note and the full list saved elsewhere. This tool does not enumerate directory entries.
Use `read` to read the paths found (up to ${MAX_MATCHES} matches are collected; results are head+tail folded with the omitted count reported in `message`).
${WINDOWS_PATH_HINT}
