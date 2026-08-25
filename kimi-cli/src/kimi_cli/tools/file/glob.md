Find files whose paths match a glob pattern. Returns file paths — never directories — including hidden and ignored files (VCS metadata excluded). Results in modification-time order: up to 100 paths, or the first 100 with a note and the full list saved elsewhere. Does not enumerate directory entries.
Use `read` to read the paths found (up to ${MAX_MATCHES} matches collected; omitted count reported in `message`).
${WINDOWS_PATH_HINT}
