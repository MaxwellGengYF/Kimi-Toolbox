"""Tests for the shell-tool hardline safety floor (safety.py)."""

import os
import sys

import pytest

from kimix.tools.file.bash.safety import (
    check_hardline_blocked,
    command_detection_variants,
    detect_self_kill,
    foreground_background_guidance,
    self_kill_hint,
    validate_workdir,
)


# ============================================================================
# check_hardline_blocked — every destructive pattern is blocked
# ============================================================================

class TestHardlineBlocked:
    @pytest.mark.parametrize(
        "command",
        [
            # recursive deletes of protected roots
            "rm -rf /",
            "rm -fr /",
            "rm -rf /*",
            "rm -rf /.",
            "rm -rf /./",
            "rm -rf /../",
            'rm -rf "/"',
            "rm -rf $HOME",
            "rm -rf ${HOME}",
            "rm -rf ~",
            "rm -rf $HOME/",
            'rm -rf "$HOME"',
            "rm -rf /tmp/build 2>/dev/null; rm -rf /",
            # Windows recursive deletes of drive roots
            "rmdir /s /q C:\\",
            "del /f /s /q C:\\*",
            # obfuscation variants
            r"r\m -rf /",
            'rm "" -rf /',
            "Rm -Rf /",
            r"rM -fR /",
            "rm -r -f /",
            "rm --recursive --force /",
            # mkfs (any subcommand)
            "mkfs.ext4 /dev/sda1",
            "mkfs.fat /dev/sdb",
            "mkfs /dev/sdc",
            "mkfs --help",
            # dd to raw devices
            "dd if=/dev/zero of=/dev/sda",
            "dd of=/dev/nvme0n1",
            "dd of=/dev/disk2",
            "dd if=/dev/urandom of=/dev/rdisk3",
            # system power commands
            "shutdown -h now",
            "shutdown",
            "reboot",
            "poweroff",
            "halt",
            # fork bomb
            ":(){ :|:& };:",
            # kill targeting PID 1 / $PPID
            "kill -9 1",
            "kill -1 1",
            "kill -KILL 1",
            "kill -9 $PPID",
            "kill 1",
            # Windows format of a drive
            "format C:",
            "format C:\\",
        ],
    )
    def test_destructive_command_blocked(self, command: str) -> None:
        blocked, desc = check_hardline_blocked(command)
        assert blocked, f"expected hardline block for {command!r}"
        assert desc

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /tmp/build",
            "rm -rf ./build",
            "rm -rf build/",
            "rm -f file.txt",
            "dd if=/dev/zero of=backup.img",
            "dd if=/dev/zero of=/tmp/backup.bin bs=1M count=10",
            "kill 123",
            "kill -9 12345",
            "kill -TERM 42",
            "python -c \"print('reboot')\"",
            "ls /",
            "cat ~/.bashrc",
            "echo $HOME",
            "git diff",
            "sleep 5",
            "format /dev/sda1",
            "format --help",
            "test -f /etc/passwd",
            "find / -name '*.py'",
        ],
    )
    def test_benign_lookalikes_not_blocked(self, command: str) -> None:
        blocked, _ = check_hardline_blocked(command)
        assert not blocked, f"expected benign command to pass: {command!r}"


# ============================================================================
# command_detection_variants — deobfuscation
# ============================================================================

class TestCommandDetectionVariants:
    def test_includes_collapsed_original(self) -> None:
        variants = command_detection_variants("  echo    hi  ")
        assert "echo hi" in variants

    def test_includes_deobfuscated_variant(self) -> None:
        variants = command_detection_variants(r"r\m -rf /")
        assert r"r\m -rf /" in variants
        assert "rm -rf /" in variants  # backslash-escape removed + lowercased

    def test_includes_lowercase_variant(self) -> None:
        variants = command_detection_variants("Rm -Rf /")
        assert "rm -rf /" in variants

    def test_deduped(self) -> None:
        variants = command_detection_variants("echo HI")
        assert len(variants) == len(set(variants))
        assert len(variants) <= 3

    def test_empty_command(self) -> None:
        assert command_detection_variants("") == []


# ============================================================================
# validate_workdir
# ============================================================================

class TestValidateWorkdir:
    @pytest.mark.parametrize(
        "workdir",
        [
            None,
            "",
            "C:/Users/me",
            r"C:\Users\me",
            "/home/user",
            "~/projects",
            "my project",
            ".",
            "./sub/dir",
            r"D:\work\a b\c",
        ],
    )
    def test_valid(self, workdir: str | None) -> None:
        assert validate_workdir(workdir) is None

    @pytest.mark.parametrize(
        "workdir",
        [
            "a;b",
            "$HOME",
            "a|b",
            "a&b",
            "a>b",
            "a<b",
            "a`b",
            "a(b)",
            "a)b",
            'a"b',
            "a'b",
            "a*b",
            "a?b",
            "a!b",
            "a{b}",
            "a}b",
            "x\x00y",
            "x\ty",
        ],
    )
    def test_invalid(self, workdir: str) -> None:
        err = validate_workdir(workdir)
        assert err is not None
        assert "Invalid workdir" in err
        assert "not allowed" in err


# ============================================================================
# foreground_background_guidance
# ============================================================================

class TestForegroundBackgroundGuidance:
    @pytest.mark.parametrize(
        "command",
        [
            "npm run dev",
            "npm run start",
            "pnpm run serve",
            "yarn run watch",
            "bun run dev",
            "next dev",
            "vite",
            "vite --port 3000",
            "nodemon app.js",
            "uvicorn app:app",
            "gunicorn app:app",
            "python -m http.server 8000",
            "docker compose up",
            "docker-compose up -d",
            "sleep 1000 &",
            "nohup python server.py &",
            "setsid python server.py",
        ],
    )
    def test_long_running_detected(self, command: str) -> None:
        hint = foreground_background_guidance(command)
        assert hint is not None
        assert "background" in hint

    @pytest.mark.parametrize(
        "command",
        [
            'echo "npm run dev"',
            'python -c "print(\'vite\')"',
            "ls -la",
            "echo hello",
            "git status",
            "npm run build",
            "python -m http.client",
            "cat docker-compose.yml",
        ],
    )
    def test_not_long_running(self, command: str) -> None:
        assert foreground_background_guidance(command) is None

    def test_empty_command(self) -> None:
        assert foreground_background_guidance("") is None
        assert foreground_background_guidance(None) is None


# ============================================================================
# detect_self_kill — kill-style commands targeting the agent process
# ============================================================================

class TestDetectSelfKill:
    # Fake agent tree: self=4100, parents=2100/900.  Names mirror a
    # ``python.exe``-hosted ``kimi`` console script.
    PIDS = {4100, 2100, 900}
    NAMES = {"python.exe", "python", "kimi"}
    CMDLINE = r"C:\venv\Scripts\python.exe C:\venv\Scripts\kimi.exe serve"

    def detect(self, command: str) -> str | None:
        return detect_self_kill(
            command,
            protected_pids=set(self.PIDS),
            image_names=set(self.NAMES),
            cmdline=self.CMDLINE,
        )

    @pytest.mark.parametrize(
        "command",
        [
            # POSIX kill / Windows tskill targeting self or ancestors
            "kill 4100",
            "kill -9 4100",
            "kill -TERM 2100",
            "kill 900",
            "kill -9 -9 4100",
            "echo hi; kill 4100",
            "sleep 1 && kill 4100",
            "KILL 4100",
            "tskill 4100",
            # taskkill /PID and /FI pid filters
            "taskkill /PID 4100",
            "taskkill /F /PID 2100 /T",
            "taskkill /pid 900 /f",
            'taskkill /FI "PID eq 4100"',
            "TASKKILL /PID 4100",
            # taskkill /IM image-name kills matching the agent's image
            "taskkill /IM python.exe /F",
            "taskkill /im python",
            "taskkill /IM PYTHON.EXE",
            # Stop-Process -Id / -Name
            "Stop-Process -Id 4100",
            "Stop-Process -Id 100,4100",
            "stop-process -id 2100",
            "Stop-Process -Name python",
            "Stop-Process -Name python*",
            "Stop-Process -Name python.exe",
            "Stop-Process -Force -Id 900",
            # Get-Process piped into a kill / .Kill()
            "Get-Process -Id 4100 | Stop-Process -Force",
            "Get-Process python | Stop-Process",
            "(Get-Process -Id 2100).Kill()",
            # pkill / killall name + full-cmdline patterns
            "pkill python",
            "pkill -9 python",
            "pkill -f python",
            'pkill -9 -f "kimi serve"',
            "pkill -f kimi",
            "killall python",
            "killall -9 python",
            # wmic process delete / terminate
            "wmic process where ProcessId=4100 delete",
            'wmic process where "ProcessId=2100" call terminate',
        ],
    )
    def test_self_kill_detected(self, command: str) -> None:
        desc = self.detect(command)
        assert desc is not None, f"expected self-kill detection for {command!r}"

    @pytest.mark.parametrize(
        "command",
        [
            # kills of unrelated PIDs are fine
            "kill 12345",
            "kill -9 9999",
            "kill -TERM 42",
            "taskkill /PID 1234 /F",
            "taskkill /FI \"PID eq 9999\"",
            "Stop-Process -Id 4242",
            "tskill 7777",
            "wmic process where ProcessId=999 delete",
            # kills of unrelated image names are fine
            "taskkill /IM node.exe /F",
            "taskkill /IM notepad.exe",
            "Stop-Process -Name node",
            "pkill -f node",
            "pkill node",
            "killall node",
            # signal listing / no target
            "kill -l",
            "kill -l 9",
            "kill",
            # container kills are not host PID kills
            "docker kill 4100",
            "docker compose kill 4100",
            # bare Get-Process queries are read-only
            "Get-Process -Id 4100",
            "Get-Process python",
            # unrelated commands
            "echo 4100",
            "ps aux",
            "git status",
            # bash fallback function definitions (compatibility prelude style)
            'pkill() { command pkill "$@"; }',
            'if ! command -v pkill >/dev/null 2>&1; then pkill() { command pkill "$@"; }; fi',
            "",
        ],
    )
    def test_benign_commands_not_detected(self, command: str) -> None:
        desc = self.detect(command)
        assert desc is None, f"unexpected self-kill detection for {command!r}: {desc}"

    def test_detection_mentions_pid_and_tool(self) -> None:
        desc = self.detect("kill -9 4100")
        assert desc is not None
        assert "4100" in desc
        assert "kill" in desc

    def test_none_command(self) -> None:
        assert detect_self_kill(None) is None


# ============================================================================
# self_kill_hint — default (live process) targets + deobfuscation variants
# ============================================================================

class TestSelfKillHint:
    def test_kill_own_pid(self) -> None:
        hint = self_kill_hint(f"kill {os.getpid()}")
        assert hint is not None
        assert str(os.getpid()) in hint

    def test_kill_own_pid_forced(self) -> None:
        assert self_kill_hint(f"kill -9 {os.getpid()}") is not None

    def test_kill_parent_pid(self) -> None:
        assert self_kill_hint(f"kill {os.getppid()}") is not None

    def test_taskkill_own_pid(self) -> None:
        assert self_kill_hint(f"taskkill /F /PID {os.getpid()}") is not None

    def test_stop_process_own_pid(self) -> None:
        assert self_kill_hint(f"Stop-Process -Id {os.getpid()}") is not None

    def test_taskkill_own_image_name(self) -> None:
        image = os.path.basename(sys.executable)
        assert self_kill_hint(f"taskkill /F /IM {image}") is not None

    def test_obfuscated_spelling_detected(self) -> None:
        # Quote tricks are removed by command_detection_variants.
        assert self_kill_hint(f'k""ill {os.getpid()}') is not None

    def test_uppercase_detected(self) -> None:
        assert self_kill_hint(f"KiLl {os.getpid()}") is not None

    def test_hint_contains_guidance(self) -> None:
        hint = self_kill_hint(f"kill {os.getpid()}")
        assert hint is not None
        assert "agent session" in hint
        assert "ask the user" in hint

    def test_unrelated_pid_allowed(self) -> None:
        # Far outside any realistic PID range on Windows/POSIX.
        assert self_kill_hint("kill 999999999") is None

    def test_unrelated_image_allowed(self) -> None:
        assert self_kill_hint("taskkill /F /IM node.exe") is None

    def test_empty_command(self) -> None:
        assert self_kill_hint("") is None
        assert self_kill_hint(None) is None
