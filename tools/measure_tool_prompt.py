"""Measure the wire-visible tool-prompt budget (plan.md §5).

Serializes the four shell/python tools the way the chat provider does
(``tool.description`` + ``tool.params.model_json_schema()``, see
``kosong/chat_provider/openai_common.py``) and reports per-tool and total
char counts.  When ``.baseline_tools.json`` exists (captured before the
refactor), also prints the delta.

``--full`` measures the whole builtin tool list: the four kimix shell/python
tools plus the kimi-cli description ``.md`` files (with their known template
vars substituted by their default lengths).  The full-list baseline is saved
to ``tools/baseline_tool_prompts_full.json`` on first run and compared on
later runs.

Usage:
    uv run python tools/measure_tool_prompt.py
    uv run python tools/measure_tool_prompt.py --full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "tools" / "baseline_tool_prompts.json"
FULL_BASELINE = ROOT / "tools" / "baseline_tool_prompts_full.json"

# ── --full: kimi-cli description .md files + default template substitutions ─
# The ${...} vars are substituted with the same default values the owning
# tools pass to ``load_desc`` (read.py: MAX_LINE_LENGTH=4000, MAX_LINES=5000,
# MAX_BYTES=102400, MAX_FILES=32; glob.py: WINDOWS_PATH_HINT; read_media.py:
# MAX_MEDIA_MEGABYTES=100).
_WINDOWS_PATH_HINT = (
    "Windows: `directory` accepts native (`C:\\Users\\foo`) and POSIX-style "
    "(`/c/Users/foo`) paths. Results use backslashes — convert to forward "
    "slashes for shell commands."
)

FULL_DESC_FILES: list[tuple[str, Path, dict[str, str]]] = [
    ("agent/description.md", ROOT / "kimi-cli/src/kimi_cli/tools/agent/description.md", {}),
    ("ask_user/description.md", ROOT / "kimi-cli/src/kimi_cli/tools/ask_user/description.md", {}),
    ("file/glob.md", ROOT / "kimi-cli/src/kimi_cli/tools/file/glob.md", {"WINDOWS_PATH_HINT": _WINDOWS_PATH_HINT}),
    ("file/read.md", ROOT / "kimi-cli/src/kimi_cli/tools/file/read.md",
     {"MAX_LINE_LENGTH": "4000", "MAX_LINES": "5000", "MAX_BYTES": "102400", "MAX_FILES": "32"}),
    ("file/read_media.md", ROOT / "kimi-cli/src/kimi_cli/tools/file/read_media.md",
     {"MAX_MEDIA_MEGABYTES": "100"}),
    ("file/write.md", ROOT / "kimi-cli/src/kimi_cli/tools/file/write.md", {}),
    ("web/fetch.md", ROOT / "kimi-cli/src/kimi_cli/tools/web/fetch.md", {}),
    ("web/search.md", ROOT / "kimi-cli/src/kimi_cli/tools/web/search.md", {}),
]


class _FakeSession:
    """Minimal stand-in for kimi_cli.session.Session used by tool __init__s."""

    def __init__(self) -> None:
        self.custom_data: dict[str, object] = {}
        self.custom_config = mock.MagicMock()
        self.custom_config.get.side_effect = lambda key, default=None: (
            {} if key == "config_json" else default
        )


def _build_tools() -> dict[str, object]:
    """Instantiate the four tools with Windows platform semantics."""
    from kimix.tools.file import run as rt
    from kimix.tools.file.bash import bash_tool as bt
    from kimix.tools.file.bash import pwsh_tool as pt
    from kimix.tools.file.bash.bash_tool import Bash
    from kimix.tools.file.bash.pwsh_tool import Powershell
    from kimix.tools.file.run import Run
    from kimix.tools.py import python

    session = _FakeSession()
    tools: dict[str, object] = {}
    with mock.patch.object(bt.sys, "platform", "win32"), mock.patch.object(
        bt, "_should_enable_bash", return_value=True
    ), mock.patch.object(bt, "find_bash", return_value=r"C:\Git\bin\bash.exe"):
        tools["bash"] = Bash(session)
    with mock.patch.object(pt.sys, "platform", "win32"), mock.patch.object(
        pt._bash_tool, "_should_enable_powershell", return_value=True
    ), mock.patch.object(pt, "find_pwsh", return_value=r"C:\pwsh\pwsh.exe"), mock.patch.object(
        pt, "load_desc", return_value="<PWSH_MD>"
    ):
        tools["Powershell"] = Powershell(session)
    tools["python"] = python(session)
    with mock.patch.object(rt, "USE_SYSTEM_SHELL", True), mock.patch.object(
        rt, "USE_SYSTEM_PWSH_ON_WINDOWS", False
    ), mock.patch.object(rt, "find_bash", return_value=None):
        tools["Run"] = Run(session)
    return tools


def serialize(tools: dict[str, object]) -> dict[str, dict[str, str]]:
    """Serialize the tool list the way the chat provider sends it to the LLM."""
    out: dict[str, dict[str, str]] = {}
    for name, tool in tools.items():
        out[name] = {
            "description": tool.description,  # type: ignore[attr-defined]
            "schema": json.dumps(  # type: ignore[attr-defined]
                tool.params.model_json_schema(), sort_keys=True, ensure_ascii=False
            ),
        }
    return out


def render_md(path: Path, subs: dict[str, str]) -> str:
    """Render a description .md with the given ``${VAR}`` substitutions."""
    text = path.read_text(encoding="utf-8")
    for key, value in subs.items():
        text = text.replace("${" + key + "}", value)
    return text


def full_sources() -> dict[str, dict[str, str]]:
    """The whole builtin tool list: 4 kimix tools + kimi-cli description .md."""
    sources = serialize(_build_tools())
    for label, path, subs in FULL_DESC_FILES:
        sources[label] = {"description": render_md(path, subs), "schema": ""}
    return sources


def _print_table(sources: dict[str, dict[str, str]]) -> int:
    total = 0
    print(f"{'source':<28}{'desc':>8}{'schema':>8}{'total':>8}")
    for name, item in sorted(sources.items()):
        n = len(item["description"]) + len(item["schema"])
        total += n
        print(f"{name:<28}{len(item['description']):>8}{len(item['schema']):>8}{n:>8}")
    print(f"{'TOTAL':<28}{'':>8}{'':>8}{total:>8}")
    return total


def full_main() -> int:
    """--full: measure the whole builtin tool list and track its baseline."""
    sources = full_sources()
    total = _print_table(sources)

    if not FULL_BASELINE.exists():
        # First run: capture the baseline (description + schema, like the
        # default 4-tool baseline).
        FULL_BASELINE.write_text(
            json.dumps(sources, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nfull-list baseline saved to: {FULL_BASELINE}")
        print(f"full-list total: {total} chars")
    else:
        old = json.loads(FULL_BASELINE.read_text(encoding="utf-8"))
        old_total = sum(
            len(v["description"]) + len(v.get("schema", "")) for v in old.values()
        )
        delta = total - old_total
        print(f"\nfull-list baseline total: {old_total}")
        print(f"delta (negative = saved):   {delta}")
        if delta < 0:
            print(f"approx tokens saved @4 ch/token: {-delta // 4}")
        else:
            print(f"approx tokens added @4 ch/token: {delta // 4}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the wire-visible tool-prompt budget (plan.md §5)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="measure the whole builtin tool list (4 kimix tools + kimi-cli .md descriptions)",
    )
    args = parser.parse_args()
    if args.full:
        return full_main()

    tools = _build_tools()
    serialized = serialize(tools)
    _print_table(serialized)

    if BASELINE.exists():
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
        old_total = sum(
            len(v["description"]) + len(json.dumps(v["schema"], sort_keys=True, ensure_ascii=False))
            for v in old.values()
        )
        total = sum(len(v["description"]) + len(v["schema"]) for v in serialized.values())
        delta = total - old_total
        print(f"\npre-refactor baseline total: {old_total}")
        print(f"delta (negative = saved):   {delta}")
        print(f"approx tokens saved @4 ch/token: {-delta // 4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
