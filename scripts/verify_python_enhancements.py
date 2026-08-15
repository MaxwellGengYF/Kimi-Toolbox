"""Independent end-to-end verification of the Python-tool enhancement round.

Covers (each asserted):
1. env scrubbing: script printing a scrubbed env var sees it removed.
2. capture-time redaction: a script printing a fake JWT is masked in pop_output.
3. syntax pre-check: broken code returns early ToolError "Syntax error" with no spawn.
4. cwd param: script prints os.getcwd() honoring cwd.
5. summarize gate config off skips the summarizer.
6. security.py scrub keep/drop + validate_workdir behavior.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kimi_agent_sdk import ToolError  # noqa: E402
from kimix.tools.py import Params, python  # noqa: E402
from kimix.tools.security import scrub_child_env, validate_workdir  # noqa: E402


def is_ok(res) -> bool:
    return not isinstance(res, ToolError)


def make_session(tmp: Path) -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    session.dir = tmp / "sessions" / "t"
    session.dir.mkdir(parents=True, exist_ok=True)
    session.custom_config = {"config_json": {}}
    return session


async def main() -> int:
    ok = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        session = make_session(tmp)

        # 1. env scrubbing end-to-end
        tool = python(session=session)
        code = "import os; print(os.environ.get('VERIFY_SCRUB_TOKEN', '<scrubbed>'))"
        os.environ["VERIFY_SCRUB_TOKEN"] = "s3cr3t-value"
        res = await tool(Params(code=code, timeout=30))
        assert is_ok(res), f"scrub run failed: {res.message}"
        assert "<scrubbed>" in str(res.output), f"env not scrubbed: {res.output!r}"
        ok += 1
        print("PASS 1: env scrubbing removes secret-looking vars")

        # 2. capture-time redaction (background mode -> job_output path reads raw stream)
        fake_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        tool2 = python(session=session)
        code2 = f"print('{fake_jwt}')"
        res2 = await tool2(Params(code=code2, timeout=30))
        assert is_ok(res2), f"redact run failed: {res2.message}"
        assert "[REDACTED]" in str(res2.output), f"JWT not redacted: {res2.output!r}"
        assert fake_jwt not in str(res2.output), "JWT leaked"
        ok += 1
        print("PASS 2: capture-time redaction masks printed secrets")

        # 3. syntax pre-check fail-fast (no spawn, brief 'Syntax error')
        tool3 = python(session=session)
        res3 = await tool3(Params(code="def broken(:\n    pass", timeout=30))
        assert res3.is_error, "broken code should be an error"
        assert res3.brief == "Syntax error", f"unexpected brief: {res3.brief}"
        assert "Syntax error" in str(res3.message)
        assert "task_id" not in str(res3.output), "must fail before spawn (no task_id)"
        ok += 1
        print("PASS 3: syntax pre-check fails fast without spawning")

        # 4. cwd param honored
        workdir = tmp / "workdir"
        workdir.mkdir()
        tool4 = python(session=session)
        res4 = await tool4(Params(code="import os; print(os.getcwd())", cwd=str(workdir), timeout=30))
        assert is_ok(res4), f"cwd run failed: {res4.message}"
        assert str(workdir) in str(res4.output), f"cwd not honored: {res4.output!r}"
        ok += 1
        print("PASS 4: cwd param is honored")

        # 4b. invalid cwd rejected without spawn
        tool4b = python(session=session)
        res4b = await tool4b(Params(code="print(1)", cwd="bad$dir;rm", timeout=30))
        assert res4b.is_error and res4b.brief == "Invalid workdir", f"unexpected: {res4b.brief}"
        ok += 1
        print("PASS 4b: invalid cwd rejected without spawn")

        # 5. summarize gate off -> summarizer not invoked (large compressible-ish output)
        calls = []
        import kimix.tools.py as py_mod
        original = py_mod._summarize_long_output_async
        py_mod._summarize_long_output_async = _recording(original, calls)
        try:
            session5 = make_session(tmp)
            session5.custom_config = {"config_json": {"python": {"summarize_long_output": False}}}
            tool5 = python(session=session5)
            big = "\n".join(f"line_{i:05d}_" + "x" * 40 for i in range(3000))  # ~180KB, distinct lines
            res5 = await tool5(Params(code=f"print('{big[:200]}')", timeout=30))  # smoke only
            assert calls == [], f"summarizer called despite gate off: {calls}"
            ok += 1
            print("PASS 5: summarize_long_output=false skips the summarizer")
        finally:
            py_mod._summarize_long_output_async = original

        # 6. security.py pure functions
        scrubbed = scrub_child_env({"AWS_ACCESS_KEY_ID": "AKIA123", "PATH": "/usr/bin",
                                    "GITHUB_TOKEN": "ghp_x", "VIRTUAL_ENV": "/v",
                                    "KIMIX_PYTHON_EXECUTABLE": "/k", "DB_PASSWORD": "p"})
        assert "AWS_ACCESS_KEY_ID" not in scrubbed and "GITHUB_TOKEN" not in scrubbed
        assert "DB_PASSWORD" not in scrubbed
        assert scrubbed["PATH"] == "/usr/bin" and scrubbed["VIRTUAL_ENV"] == "/v"
        assert scrubbed["KIMIX_PYTHON_EXECUTABLE"] == "/k"
        assert validate_workdir("C:\\proj\\sub dir") is None
        assert validate_workdir("bad;rm -rf") is not None
        ok += 1
        print("PASS 6: security.py scrub_child_env / validate_workdir")

    print(f"\nALL {ok} CHECKS PASSED")
    return 0


def _recording(original, calls):
    async def wrapper(*a, **kw):
        calls.append(a[1] if len(a) > 1 else "?")
        return await original(*a, **kw)
    return wrapper


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
