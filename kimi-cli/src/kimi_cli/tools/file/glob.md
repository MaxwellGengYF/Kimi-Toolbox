Find files whose paths match a glob pattern. Returns matching file paths — never directories — including hidden and ignored files (VCS metadata directories are excluded). Up to 100 paths come back in modification-time order; a larger result returns the first 100 paths in modification-time order, says so, and reports where the complete sorted list was saved. This tool does not enumerate directory entries.

Use `read` to read the paths found (up to ${MAX_MATCHES} matches are collected; results are head+tail folded with the omitted count reported in `message`).
${WINDOWS_PATH_HINT}
