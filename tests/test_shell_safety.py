"""Tests for the shell-tool hardline safety floor (safety.py)."""

import pytest

from kimix.tools.file.bash.safety import (
    check_hardline_blocked,
    command_detection_variants,
    foreground_background_guidance,
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
