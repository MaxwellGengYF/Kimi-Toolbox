Find files by glob. Returns file paths — never directories — including hidden/ignored (VCS metadata excluded), in modification-time order: up to 100 paths (first 100 with a note; full list saved elsewhere). Does not enumerate directory entries.
Use `read` to open matches (up to ${MAX_MATCHES} collected; omitted count reported in `message`).
${WINDOWS_PATH_HINT}
