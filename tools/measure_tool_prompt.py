"""Measure the wire-visible tool-prompt budget (plan.md §5).

Serializes the four shell/python tools the way the chat provider does
(``tool.description`` + ``tool.params.model_json_schema()``, see
``kosong/chat_provider/openai_common.py``) and reports per-tool and total
char counts.  When ``.baseline_tools.json`` exists (captured before the
refactor), also prints the delta.

Usage:
    uv run python tools/measure_tool_prompt.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "tools" / "baseline_tool_prompts.json"


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
    from kimix.tools.py import Python

    session = _FakeSession()
    tools: dict[str, object] = {}
    with mock.patch.object(bt.sys, "platform", "win32"), mock.patch.object(
        bt, "_should_enable_bash", return_value=True
    ), mock.patch.object(bt, "find_bash", return_value=r"C:\Git\bin\bash.exe"):
        tools["Bash"] = Bash(session)
    with mock.patch.object(pt.sys, "platform", "win32"), mock.patch.object(
        pt._bash_tool, "_should_enable_powershell", return_value=True
    ), mock.patch.object(pt, "find_pwsh", return_value=r"C:\pwsh\pwsh.exe"), mock.patch.object(
        pt, "load_desc", return_value="<PWSH_MD>"
    ):
        tools["Powershell"] = Powershell(session)
    tools["Python"] = Python(session)
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


def main() -> int:
    tools = _build_tools()
    serialized = serialize(tools)
    total = 0
    print(f"{'tool':<12}{'desc':>8}{'schema':>8}{'total':>8}")
    for name, item in sorted(serialized.items()):
        n = len(item["description"]) + len(item["schema"])
        total += n
        print(f"{name:<12}{len(item['description']):>8}{len(item['schema']):>8}{n:>8}")
    print(f"{'TOTAL':<12}{'':>8}{'':>8}{total:>8}")

    if BASELINE.exists():
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
        old_total = sum(
            len(v["description"]) + len(json.dumps(v["schema"], sort_keys=True, ensure_ascii=False))
            for v in old.values()
        )
        delta = total - old_total
        print(f"\npre-refactor baseline total: {old_total}")
        print(f"delta (negative = saved):   {delta}")
        print(f"approx tokens saved @4 ch/token: {-delta // 4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
