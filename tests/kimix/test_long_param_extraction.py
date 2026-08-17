"""Tests for the long-content param extraction logic in kimix.tools.common.

Tests cover:
- _looks_like_malformed_json_param: detection of malformed params
- _extract_content_from_malformed: extraction of content from malformed params
- _extract_and_save_long_param: full extraction + save pipeline
- _build_long_param_retry_msg: error message building
- Integration with various tool types
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path (src/ first so it takes precedence over kimi-cli/src/)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "kimi-cli" / "src"))
sys.path.insert(0, str(project_root / "src"))

from kimix.tools.common import (
    _LONG_CONTENT_PARAMS,
    _LONG_PARAM_MIN_LENGTH,
    _looks_like_malformed_json_param,
    _extract_content_from_malformed,
    _extract_and_save_long_param,
    _build_long_param_retry_msg,
    _create_script_file,
    _temp_folder,
)

import orjson


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ensure_temp_dir():
    """Ensure the temp folder exists before each test."""
    _temp_folder.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup temp files created during test
    for f in _temp_folder.iterdir():
        if f.is_file():
            f.unlink()


# ── _LONG_CONTENT_PARAMS ────────────────────────────────────────────────────


class TestLongContentParamsRegistry:
    """Verify the registry of tools with long content params."""

    def test_has_expected_tools(self):
        assert "bash" in _LONG_CONTENT_PARAMS
        assert "pwsh" in _LONG_CONTENT_PARAMS
        assert "Run" in _LONG_CONTENT_PARAMS
        assert "python" in _LONG_CONTENT_PARAMS
        assert "write" in _LONG_CONTENT_PARAMS
        assert "edit" in _LONG_CONTENT_PARAMS
        assert "subagent" in _LONG_CONTENT_PARAMS

    def test_has_expected_params(self):
        assert "command" in _LONG_CONTENT_PARAMS["bash"]
        assert "cmd" in _LONG_CONTENT_PARAMS["bash"]
        assert "code" in _LONG_CONTENT_PARAMS["python"]
        assert "content" in _LONG_CONTENT_PARAMS["write"]
        assert "text" in _LONG_CONTENT_PARAMS["write"]
        assert "old_string" in _LONG_CONTENT_PARAMS["edit"]
        assert "new_string" in _LONG_CONTENT_PARAMS["edit"]
        assert "prompt" in _LONG_CONTENT_PARAMS["subagent"]
        assert "task" in _LONG_CONTENT_PARAMS["subagent"]

    def test_unknown_tool_returns_empty(self):
        """Unknown tool should return None from _extract_and_save_long_param."""
        result = _extract_and_save_long_param({"command": "echo hi"}, "unknown_tool")
        assert result is None


# ── _looks_like_malformed_json_param ────────────────────────────────────────


class TestLooksLikeMalformedJsonParam:
    """Test the detection of malformed JSON-encoded params."""

    def test_short_string_returns_false(self):
        """Strings shorter than _LONG_PARAM_MIN_LENGTH should return False."""
        assert _looks_like_malformed_json_param("short") is False
        assert _looks_like_malformed_json_param("") is False

    def test_plain_string_returns_false(self):
        """Plain multi-line string should not be detected as malformed."""
        long_text = "Hello, world!\n" * 50  # 700+ chars, no JSON encoding
        assert _looks_like_malformed_json_param(long_text) is False

    def test_json_encoded_string_detected(self):
        """A JSON-encoded string (starting with \") should be detected."""
        content = "print('hello')\n" * 20
        encoded = orjson.dumps(content).decode("utf-8")
        assert len(encoded) > _LONG_PARAM_MIN_LENGTH
        assert _looks_like_malformed_json_param(encoded) is True

    def test_json_array_of_strings_detected(self):
        """A JSON array of strings (should be a single string) is detected."""
        lines = ["line " + str(i) for i in range(50)]
        encoded = orjson.dumps(lines).decode("utf-8")
        assert len(encoded) > _LONG_PARAM_MIN_LENGTH
        assert _looks_like_malformed_json_param(encoded) is True

    def test_json_object_detected(self):
        """A JSON object (should be a single string) is detected."""
        obj = {"content": "x" * 300, "path": "/tmp/test.txt"}
        encoded = orjson.dumps(obj).decode("utf-8")
        assert len(encoded) > _LONG_PARAM_MIN_LENGTH
        assert _looks_like_malformed_json_param(encoded) is True

    def test_escaped_newlines_detected(self):
        """Content with escaped newlines (\\n) instead of real newlines is detected."""
        text = "line1\\nline2\\n" + "x" * 200
        assert len(text) > _LONG_PARAM_MIN_LENGTH
        assert _looks_like_malformed_json_param(text) is True

    def test_whitespace_string_returns_false(self):
        """Whitespace-only strings should return False."""
        assert _looks_like_malformed_json_param("   " * 100) is False

    def test_bracket_in_command_not_detected(self):
        """A command that starts with '[' but is not JSON should not be detected.
        e.g. bash test operators like '[ -f /tmp/file ]'."""
        cmd = "[ -f /tmp/test ] && echo 'exists'\n" * 30
        assert len(cmd) > _LONG_PARAM_MIN_LENGTH
        assert _looks_like_malformed_json_param(cmd) is False

    def test_none_or_not_string_returns_false(self):
        """Non-string values should not be checked."""
        assert _looks_like_malformed_json_param(None) is False  # type: ignore


# ── _extract_content_from_malformed ─────────────────────────────────────────


class TestExtractContentFromMalformed:
    """Test the extraction of content from malformed params."""

    def test_plain_string_returns_none(self):
        """Plain strings should return None (no extraction needed)."""
        long_text = "Hello, world!\n" * 50
        assert _extract_content_from_malformed(long_text) is None

    def test_empty_string_returns_none(self):
        assert _extract_content_from_malformed("") is None
        assert _extract_content_from_malformed("   ") is None

    def test_json_encoded_string_extracted(self):
        """A JSON-encoded string should be unescaped."""
        original = "print('hello')\n" * 20
        encoded = orjson.dumps(original).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == original

    def test_json_array_of_strings_joined(self):
        """A JSON array of strings should be joined with newlines."""
        lines = ["line " + str(i) for i in range(10)]
        encoded = orjson.dumps(lines).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        # The result should have actual newlines, not \\n literal
        assert result == "\n".join(lines)

    def test_json_object_with_content_key(self):
        """A JSON object with a 'content' key should extract that key."""
        obj = {"content": "x" * 300, "path": "/tmp/test.txt"}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == obj["content"]

    def test_json_object_with_code_key(self):
        """A JSON object with a 'code' key should extract that key."""
        obj = {"code": "def foo():\n    pass\n" * 20}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == obj["code"]

    def test_json_object_with_command_key(self):
        """A JSON object with a 'command' key should extract that key."""
        obj = {"command": "echo hello\n" * 30}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == obj["command"]

    def test_json_object_with_list_value(self):
        """A JSON object with a list value for a content key."""
        obj = {"content": ["line1", "line2", "line3"]}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == "line1\nline2\nline3"

    def test_json_object_without_known_key(self):
        """A JSON object without a known content key returns None."""
        obj = {"foo": "bar", "baz": 123}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        # Should return None since no known content key and no long value
        assert result is None

    def test_json_object_with_text_key(self):
        """A JSON object with a 'text' key should extract that key."""
        obj = {"text": "x" * 300, "encoding": "utf-8"}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == obj["text"]

    def test_json_object_with_value_key(self):
        """A JSON object with a 'value' key should extract that key."""
        obj = {"value": "important data\n" * 30}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == obj["value"]

    def test_json_object_with_cmd_key(self):
        """A JSON object with a 'cmd' key should extract that key."""
        obj = {"cmd": "dir /s\n" * 30}
        encoded = orjson.dumps(obj).decode("utf-8")
        result = _extract_content_from_malformed(encoded)
        assert result is not None
        assert result == obj["cmd"]

    def test_escaped_newlines_replaced(self):
        """Escaped newlines (\\n) should be replaced with real newlines."""
        text = "line1\\nline2\\nline3\\n" + "x" * 200
        result = _extract_content_from_malformed(text)
        assert result is not None
        assert "\\n" not in result
        assert "\n" in result


# ── _extract_and_save_long_param ────────────────────────────────────────────


class TestExtractAndSaveLongParam:
    """Test the full extraction + save pipeline."""

    def test_plain_string_no_extraction(self):
        """Plain string params should not be extracted."""
        args = {"command": "echo hello\n" * 50}
        result = _extract_and_save_long_param(args, "bash")
        assert result is None

    def test_short_string_no_extraction(self):
        """Short strings should not be extracted even if malformed-looking."""
        args = {"command": '"short"'}
        result = _extract_and_save_long_param(args, "bash")
        assert result is None

    def test_json_encoded_string_saved(self):
        """JSON-encoded string should be saved to a temp file."""
        original = "print('hello')\n" * 20
        encoded = orjson.dumps(original).decode("utf-8")
        args = {"command": encoded}
        result = _extract_and_save_long_param(args, "bash")
        assert result is not None
        assert "command" in result
        # Verify the file exists and contains the extracted content
        file_path = result["command"]
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == original

    def test_list_param_saved(self):
        """List of strings should be saved joined."""
        lines = ["line " + str(i) for i in range(20)]
        args = {"code": lines}
        result = _extract_and_save_long_param(args, "python")
        assert result is not None
        assert "code" in result
        file_path = result["code"]
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == "\n".join(lines)

    def test_dict_param_saved(self):
        """Dict param should be saved as JSON."""
        obj = {"content": "x" * 300, "path": "/tmp/test.txt"}
        args = {"content": obj}
        result = _extract_and_save_long_param(args, "write")
        assert result is not None
        assert "content" in result
        file_path = result["content"]
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "x" * 300 in content

    def test_multiple_params_saved(self):
        """Multiple malformed params should all be saved."""
        args = {
            "old_string": orjson.dumps("old content\n" * 30).decode("utf-8"),
            "new_string": orjson.dumps("new content\n" * 30).decode("utf-8"),
        }
        result = _extract_and_save_long_param(args, "edit")
        assert result is not None
        assert "old_string" in result
        assert "new_string" in result
        for file_path in result.values():
            assert os.path.isfile(file_path)

    def test_unknown_tool_returns_none(self):
        """Unknown tool should return None."""
        args = {"command": "x" * 300}
        result = _extract_and_save_long_param(args, "nonexistent_tool")
        assert result is None

    def test_mixed_valid_and_malformed(self):
        """Mix of valid and malformed params should only extract malformed ones."""
        args = {
            "command": orjson.dumps("echo hello\n" * 20).decode("utf-8"),  # malformed, long
        }
        # bash should only check "command" and "cmd"
        result = _extract_and_save_long_param(args, "bash")
        assert result is not None
        assert "command" in result
    def test_ext_with_custom_extension(self):
        """Custom extension should be used for the temp file."""
        original = "print('hello')" * 20
        encoded = orjson.dumps(original).decode("utf-8")
        args = {"code": encoded}
        result = _extract_and_save_long_param(args, "python", ext=".py")
        assert result is not None
        file_path = result["code"]
        assert file_path.endswith(".py"), f"Expected .py extension, got: {file_path}"

    def test_empty_args_returns_none(self):
        """Empty args dict should return None."""
        result = _extract_and_save_long_param({}, "bash")
        assert result is None

    def test_non_string_list_items(self):
        """List with non-string items should still be saved (joined as str)."""
        args = {"code": [42, True, "hello"]}
        result = _extract_and_save_long_param(args, "python")
        assert result is not None
        assert "code" in result
        file_path = result["code"]
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "42" in content
        assert "True" in content
        assert "hello" in content

    def test_source_code_param(self):
        """Python 'source_code' param should be checked."""
        encoded = orjson.dumps("import os\n" * 30).decode("utf-8")
        args = {"source_code": encoded}
        result = _extract_and_save_long_param(args, "python")
        assert result is not None
        assert "source_code" in result

    def test_file_param(self):
        """Python 'file' param should be checked."""
        encoded = orjson.dumps("data.txt" * 50).decode("utf-8")
        args = {"file": encoded}
        result = _extract_and_save_long_param(args, "python")
        assert result is not None
        assert "file" in result

    def test_old_param(self):
        """Edit 'old' param should be checked."""
        encoded = orjson.dumps("old content\n" * 30).decode("utf-8")
        args = {"old": encoded}
        result = _extract_and_save_long_param(args, "edit")
        assert result is not None
        assert "old" in result

    def test_new_param(self):
        """Edit 'new' param should be checked."""
        encoded = orjson.dumps("new content\n" * 30).decode("utf-8")
        args = {"new": encoded}
        result = _extract_and_save_long_param(args, "edit")
        assert result is not None
        assert "new" in result

    def test_text_param(self):
        """Write 'text' param should be checked."""
        encoded = orjson.dumps("text content\n" * 30).decode("utf-8")
        args = {"text": encoded}
        result = _extract_and_save_long_param(args, "write")
        assert result is not None
        assert "text" in result

    def test_cmd_param_for_bash(self):
        """Bash 'cmd' param should be checked."""
        encoded = orjson.dumps("echo hello\n" * 30).decode("utf-8")
        args = {"cmd": encoded}
        result = _extract_and_save_long_param(args, "bash")
        assert result is not None
        assert "cmd" in result

    def test_cmd_param_for_pwsh(self):
        """Pwsh 'cmd' param should be checked."""
        encoded = orjson.dumps("Write-Host hello\n" * 30).decode("utf-8")
        args = {"cmd": encoded}
        result = _extract_and_save_long_param(args, "pwsh")
        assert result is not None
        assert "cmd" in result

    def test_cmd_param_for_run(self):
        """Run 'cmd' param should be checked."""
        encoded = orjson.dumps("echo hello\n" * 30).decode("utf-8")
        args = {"cmd": encoded}
        result = _extract_and_save_long_param(args, "Run")
        assert result is not None
        assert "cmd" in result

    def test_no_matching_params(self):
        """Args with no matching params for the tool should return None."""
        args = {"unrelated_key": "x" * 300}
        result = _extract_and_save_long_param(args, "bash")
        assert result is None


# ── _build_long_param_retry_msg
# ── _build_long_param_retry_msg ─────────────────────────────────────────────


class TestBuildLongParamRetryMsg:
    """Test the error message builder."""

    def test_single_file_message(self):
        """Message should include the single file path."""
        saved_files = {"command": "/tmp/test/0.txt"}
        msg = _build_long_param_retry_msg(saved_files, "Original error")
        assert "Original error" in msg
        assert "command" in msg
        assert "0.txt" in msg
        assert "read" in msg

    def test_multiple_files_message(self):
        """Message should include all file paths."""
        saved_files = {
            "old_string": "/tmp/test/0.txt",
            "new_string": "/tmp/test/1.txt",
        }
        msg = _build_long_param_retry_msg(saved_files, "Validation failed")
        assert "Validation failed" in msg
        assert "old_string" in msg
        assert "new_string" in msg
        assert "0.txt" in msg
        assert "1.txt" in msg

    def test_empty_saved_files(self):
        """Empty saved_files should produce a minimal message."""
        msg = _build_long_param_retry_msg({}, "Some error")
        assert "Some error" in msg
        assert "read" in msg


# ── Integration-like tests ──────────────────────────────────────────────────


class TestIntegration:
    """Test the full pipeline end-to-end."""

    def test_bash_json_encoded_command(self):
        """Simulate a bash call with a JSON-encoded command."""
        original_cmd = "ls -la\n" * 30
        encoded = orjson.dumps(original_cmd).decode("utf-8")
        args = {"command": encoded}
        saved = _extract_and_save_long_param(args, "bash")
        assert saved is not None
        assert "command" in saved
        file_path = saved["command"]
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == original_cmd

    def test_python_code_as_list(self):
        """Simulate a python call with code as a list of lines."""
        lines = ["def foo():", "    return 42", "", "print(foo())"]
        args = {"code": lines}
        saved = _extract_and_save_long_param(args, "python")
        assert saved is not None
        assert "code" in saved
        file_path = saved["code"]
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "\n".join(lines)

    def test_write_content_as_dict(self):
        """Simulate a write call with content as a dict with nested data."""
        obj = {
            "content": '{\n  "key": "value"\n}',
            "path": "/tmp/test.json",
        }
        args = {"content": obj}
        saved = _extract_and_save_long_param(args, "write")
        assert saved is not None
        assert "content" in saved
        file_path = saved["content"]
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # The dict should be serialized as JSON
        assert "key" in content
        assert "value" in content

    def test_subagent_prompt_as_dict(self):
        """Simulate a subagent call with prompt as a dict."""
        obj = {
            "prompt": "Write a Python script that...",
            "description": "coding task",
        }
        args = {"prompt": obj}
        saved = _extract_and_save_long_param(args, "subagent")
        # A dict where a string is expected is always a format error,
        # regardless of the content length
        assert saved is not None
        assert "prompt" in saved

    def test_subagent_long_prompt_as_dict(self):
        """Simulate a subagent call with a long prompt as a dict."""
        obj = {
            "prompt": "Write a Python script that..." * 20,  # > 200 chars
            "description": "coding task",
        }
        args = {"prompt": obj}
        saved = _extract_and_save_long_param(args, "subagent")
        assert saved is not None
        assert "prompt" in saved
        file_path = saved["prompt"]
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # The dict should be serialized as JSON
        assert "Write a Python script" in content

    def test_no_false_positive_for_normal_code(self):
        """Normal Python code should not be detected as malformed."""
        code = "import os\nimport sys\n\ndef main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()\n"
        code = code * 10  # Make it long enough
        args = {"code": code}
        saved = _extract_and_save_long_param(args, "python")
        assert saved is None, "Normal code should not be extracted"

    def test_no_false_positive_for_normal_command(self):
        """Normal bash command should not be detected as malformed."""
        cmd = "ls -la /tmp\n" * 30
        args = {"command": cmd}
        saved = _extract_and_save_long_param(args, "bash")
        assert saved is None, "Normal command should not be extracted"

    def test_no_false_positive_for_normal_content(self):
        """Normal file content should not be detected as malformed."""
        content = "Hello, World!\n" * 50
        args = {"content": content}
        saved = _extract_and_save_long_param(args, "write")
        assert saved is None, "Normal content should not be extracted"