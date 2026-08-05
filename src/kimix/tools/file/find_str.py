
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field
from kimix.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kimix.tools.common import _maybe_export_output


class FindStrParams(BaseModel):
    content: str = Field(
        description="Text to search for."
    )
    path: str = Field(
        description="Target path. Supports glob patterns: '*.ext', '**.ext'."
    )
    case_sensitive: bool = Field(
        default=False,
        description="Enable case-sensitive search."
    )


class FindStr(CallableTool2):
    name: str = "FindStr"
    description: str = "Search text in files."
    params: type[FindStrParams] = FindStrParams

    async def __call__(self, params: FindStrParams) -> ToolReturnValue:
        import os
        import fnmatch

        def find_files(target_path: str) -> list[str]:
            """Find files matching the path pattern."""
            # If path is a file, return it directly
            if os.path.isfile(target_path):
                return [target_path]
            
            # Check if path contains glob patterns
            has_glob = '*' in target_path or '?' in target_path
            
            if has_glob:
                # Split into base directory and pattern
                # Handle patterns like "dir/*.ext" or "dir/**.ext"
                dir_part = target_path
                pattern = "*"
                
                # Find the last path separator that is not part of a glob
                # We need to find where the directory ends and the file pattern begins
                parts = target_path.replace('\\', '/').split('/')
                
                # Rebuild to find the split point
                base_dir = ""
                pattern_part = ""
                
                for i, part in enumerate(parts):
                    if '*' in part or '?' in part:
                        # This part contains glob, everything before is directory
                        base_dir = '/'.join(parts[:i])
                        pattern_part = '/'.join(parts[i:])
                        break
                else:
                    # No glob found in parts (shouldn't happen due to has_glob check)
                    base_dir = os.path.dirname(target_path)
                    pattern_part = os.path.basename(target_path)
                
                if not base_dir:
                    base_dir = "."
                
                # Check for recursive pattern (**)
                if '**' in pattern_part:
                    # Recursive search
                    ext_pattern = pattern_part.replace('**', '*')
                    files = []
                    for root, dirnames, filenames in os.walk(base_dir):
                        for filename in filenames:
                            if fnmatch.fnmatch(filename, ext_pattern):
                                files.append(os.path.join(root, filename))
                    return files
                else:
                    # Single directory search
                    if os.path.isdir(base_dir):
                        files = []
                        for item in os.listdir(base_dir):
                            if fnmatch.fnmatch(item, pattern_part):
                                full_path = os.path.join(base_dir, item)
                                if os.path.isfile(full_path):
                                    files.append(full_path)
                        return files
                    return []
            else:
                # No glob, treat as directory
                if os.path.isdir(target_path):
                    files = []
                    for item in os.listdir(target_path):
                        full_path = os.path.join(target_path, item)
                        if os.path.isfile(full_path):
                            files.append(full_path)
                    return files
                return []

        def find_in_file(file_path: str, search_content: str, case_sensitive: bool) -> list[dict]:
            """Find all occurrences of search_content in file."""
            results = []

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
            except Exception:
                return []

            # Native acceleration: kimix_native.tools.find_in_file. Only safe
            # when content and needle are pure ASCII and the needle contains
            # no line endings (the native kernel splits on \n and searches
            # within the line body, so a needle containing \n/\r would behave
            # differently from the Python per-line scan).
            if (
                _native_use_native("TOOLS")
                and "\n" not in search_content
                and "\r" not in search_content
            ):
                _mod = _native_get_module("tools")
                if _mod is not None:
                    content = "".join(lines)
                    if content.isascii() and search_content.isascii():
                        return _mod.find_in_file(
                            content, search_content, case_sensitive, file_path
                        )

            if not case_sensitive:
                search_lower = search_content.lower()
            else:
                search_lower = search_content

            for line_num, line in enumerate(lines, start=1):
                if not case_sensitive:
                    line_to_search = line.lower()
                else:
                    line_to_search = line

                # Find all occurrences in this line
                start = 0
                while True:
                    idx = line_to_search.find(search_lower, start)
                    if idx == -1:
                        break
                    results.append({
                        'file': file_path,
                        'line': line_num,
                        'column': idx + 1,  # 1-based column
                        'content': line.rstrip('\n\r')
                    })
                    start = idx + 1

            return results

        try:
            files = find_files(params.path)
            
            if not files:
                return ToolOk(output=_maybe_export_output(f"No files found matching path: {params.path}"))
            
            all_matches = []
            for file_path in files:
                matches = find_in_file(file_path, params.content, params.case_sensitive)
                all_matches.extend(matches)
            
            if not all_matches:
                return ToolOk(output=_maybe_export_output(f"No matches found for '{params.content}' in {params.path}"))
            
            # Format results
            result_lines = [
                f"Found {len(all_matches)} match(es) for '{params.content}':",
                ""
            ]
            
            current_file = None
            for match in all_matches:
                if match['file'] != current_file:
                    current_file = match['file']
                    result_lines.append(f"File: {current_file}")
                result_lines.append(f"  Line {match['line']}, Col {match['column']}: {match['content']}")
            
            return ToolOk(output=_maybe_export_output("\n".join(result_lines)))
            
        except Exception as exc:
            return ToolError(
                output="",
                message=str(exc),
                brief="Failed to find string",
            )
