"""Behavior-equivalence gate for the shell/security kernels (plan: commit
0582e09 "Study from hermes").

Runs the SAME inputs through the native path (gate forced on) and the
pure-Python path (gate forced off) and asserts identical results:

- kimix.tools.security: redact_sensitive_output, scrub_child_env
- kimix.tools.file.bash.safety: check_hardline_blocked,
  foreground_background_guidance
- kimix.tools.file.bash.output_enhance: interpret_exit_code, annotate_failure
- kimix.tools.background.utils: bounded_append (StringIO contract)

Corpora are adversarial: ASCII edge cases (which run natively) + non-ASCII
cases (which route to the Python bodies by construction) + empty inputs.
"""

from __future__ import annotations

import io

import pytest

from kimi_cli.native_loader import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)


def _force_gate(module, state: bool):
    attr = "_native_use_native"
    original = getattr(module, attr, None)
    setattr(module, attr, lambda kernel: state)
    return lambda: setattr(module, attr, original) if original is not None else delattr(
        module, attr
    )


def _assert_equivalent(native_result, python_result, case):
    assert native_result == python_result, (
        f"native != python for {case!r}:\n"
        f"  native={native_result!r}\n  python={python_result!r}"
    )


# ---------------------------------------------------------------------------
# redact_sensitive_output (kimix.tools.security)
# ---------------------------------------------------------------------------

REDACT_CORPUS = [
    "",
    "hello world",
    "https://user:pass@example.com/path",
    "http://user@host:8080/path",
    "http://a:b@c/",
    "prefix https://u:p@h suffix",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    "token: eyJh.eyJ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n"
    "-----END RSA PRIVATE KEY-----\n",
    "-----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY-----",
    "no end marker -----BEGIN PRIVATE KEY-----",
    "ghp_abcdefghijklmnopqrstuvwxyz",
    "github_pat_abcdefghijklmnopqrstuvwxyz",
    "glpat-abcdefghijklmnopqrstuv",
    "AKIAIOSFODNN7EXAMPLE",
    "Authorization: Bearer abc123",
    "x-api-key: k1234567",
    "Proxy-Authorization: xyz",
    "password=supersecret",
    "password='secret123'",
    "password=x",
    "token: abcdefghij",
    "api_key=abcdef123456",
    "PASSWORD = 'abcdef12'",
    "Bearer abcdefghijklmnopqrstuvwxyz",
    "mypassword=abcdef123",
    "url https://u:p@h; Authorization: Bearer abc",
    "combined: eyJh.eyJ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c "
    "and ghp_abcdefghijklmnopqrstuvwxyz",
    # non-ASCII routes to the Python body (equal by construction)
    "caf\u00e9 password=motdepasse123",
    "\u4e2d\u6587 https://user:pass@host/path",
    "Bearer \u00e9\u00e8\u00ea12345678901234567890",
    "\U0001f600 token: abcdef123456",
]


@pytest.mark.parametrize("text", REDACT_CORPUS)
def test_redact_sensitive_output_equivalence(text):
    import kimix.tools.security as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.redact_sensitive_output(text)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.redact_sensitive_output(text)
    finally:
        restore()
    _assert_equivalent(native, python, text)


# ---------------------------------------------------------------------------
# scrub_child_env (kimix.tools.security)
# ---------------------------------------------------------------------------

SCRUB_ENVS = [
    {},
    {"PATH": "/usr/bin", "HOME": "/home/u"},
    {
        "PATH": "/usr/bin",
        "AWS_SECRET_ACCESS_KEY": "x",
        "DATABASE_URL": "postgres://u:p@h/db",
        "SSH_AUTH_SOCK": "/tmp/ssh",
        "MY_TOKEN": "t",
        "KIMIX_API_KEY": "k",
        "USER": "u",
        "AWS_ACCESS_KEY_ID": "ak",
        "GIT_ASKPASS": "g",
        "LC_ALL": "C",
        "PWD": "/x",
        "SHLVL": "1",
        "_": "x",
        "api_secret_key": "x",
        "DATABASE_PASSWORD": "p",
        "WEBHOOK_URL": "w",
        "BEARER_TOKEN": "b",
    },
    {"a": "1", "b": "2", "c": "3"},
    {"\u00e9\u00e8key": "non-ascii", "PATH": "/x"},  # non-ASCII key
]


@pytest.mark.parametrize("env", SCRUB_ENVS)
def test_scrub_child_env_equivalence(env):
    import kimix.tools.security as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.scrub_child_env(dict(env))
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.scrub_child_env(dict(env))
    finally:
        restore()
    _assert_equivalent(list(native.items()), list(python.items()), env)


# ---------------------------------------------------------------------------
# check_hardline_blocked / foreground_background_guidance (bash.safety)
# ---------------------------------------------------------------------------

HARDLINE_CORPUS = [
    "",
    "   ",
    "ls -la",
    "echo hello",
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    "rm -rf ${HOME}",
    "rm -r /",
    "rm /",
    "rm -rf /tmp/build",
    "rmdir -r /",
    "rmdir /",
    "del /f /s /q C:\\*",
    "del /q C:\\Windows",
    "rm -rf C:\\",
    r"r\m -rf /",
    "r'm' -rf /",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    "dd if=/dev/zero of=/tmp/out",
    "shutdown -h now",
    "reboot",
    ":(){ :|:& };:",
    "kill 1",
    "kill $PPID",
    "kill 123",
    "format C:",
    "format C:\\Windows",
    "sudo rm -rf /",
    "rm -rf /\u00e9",  # non-ASCII routes to the Python body
]


@pytest.mark.parametrize("command", HARDLINE_CORPUS)
def test_check_hardline_blocked_equivalence(command):
    import kimix.tools.file.bash.safety as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.check_hardline_blocked(command)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.check_hardline_blocked(command)
    finally:
        restore()
    _assert_equivalent(native, python, command)


GUIDANCE_CORPUS = [
    "",
    "   ",
    "ls -la",
    "npm run dev",
    "npm run build",
    "pnpm run watch",
    "next dev",
    "vite",
    "nodemon app.js",
    "uvicorn app:app",
    "python -m http.server",
    "docker compose up",
    "docker-compose up",
    "echo done &",
    "nohup python app.py",
    "setsid bash",
    "echo 'npm run dev'",
    'echo "docker compose up"',
    "grep npm run dev x",
    "vite \u00e9",  # non-ASCII
]


@pytest.mark.parametrize("command", GUIDANCE_CORPUS)
def test_foreground_background_guidance_equivalence(command):
    import kimix.tools.file.bash.safety as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.foreground_background_guidance(command)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.foreground_background_guidance(command)
    finally:
        restore()
    _assert_equivalent(native, python, command)


# ---------------------------------------------------------------------------
# interpret_exit_code / annotate_failure (bash.output_enhance)
# ---------------------------------------------------------------------------

EXIT_CODE_CORPUS = [
    ("grep foo", 1),
    ("grep foo", 2),
    ("grep foo", 0),
    ("grep foo", None),
    ("diff a b", 1),
    ("find . -name x", 1),
    ("test -f x", 1),
    ("[ -f x ]", 1),
    ("curl https://x", 6),
    ("curl https://x", 7),
    ("curl https://x", 22),
    ("curl https://x", 28),
    ("curl https://x", 99),
    ("git diff", 1),
    ("ls", 1),
    ("", 1),
    ("egrep foo", 1),
    ("rg foo", 1),
    ("C:\\tools\\grep.exe foo", 1),
    ("grep \u00e9", 1),  # non-ASCII
]


@pytest.mark.parametrize("command,code", EXIT_CODE_CORPUS)
def test_interpret_exit_code_equivalence(command, code):
    import kimix.tools.file.bash.output_enhance as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.interpret_exit_code(command, code)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.interpret_exit_code(command, code)
    finally:
        restore()
    _assert_equivalent(native, python, (command, code))


ANNOTATE_CORPUS = [
    ("", "x", 1),
    ("bash: foo: command not found", "foo", 127),
    ("'foo' is not recognized as an internal or external command", "foo", 1),
    ("ls: cannot access 'x': No such file or directory", "ls", 2),
    ("ModuleNotFoundError: No module named 'requests'", "python", 1),
    ("MODULENOTFOUNDERROR: No Module Named 'FooBar'", "python", 1),
    ("Permission denied", "cat", 1),
    ("everything fine", "ls", 0),
    ("x" * 5000 + "command not found", "x", 1),
    ("no such file or directory", "ls", 1),
    ("caf\u00e9 command not found", "x", 1),  # non-ASCII output
]


@pytest.mark.parametrize("output,command,code", ANNOTATE_CORPUS)
def test_annotate_failure_equivalence(output, command, code):
    import kimix.tools.file.bash.output_enhance as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.annotate_failure(output, command, code)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.annotate_failure(output, command, code)
    finally:
        restore()
    _assert_equivalent(native, python, (output, command, code))


# ---------------------------------------------------------------------------
# bounded_append (background.utils, StringIO contract)
# ---------------------------------------------------------------------------

BOUNDED_CORPUS = [
    ("", "hello", 100),
    ("a" * 90, "b" * 20, 100),
    ("", "x", 0),
    ("hello", "", 100),
    ("abcdefghijklmnopqrstuvwxyz", "0123456789", 20),
    ("x" * 5, "y" * 5, 3),
    ("", "", 0),
    ("\u00e9" * 50, "x" * 50, 40),  # non-ASCII content
    ("a" * 100, "b" * 100, 200),
    ("a" * 100, "b" * 100, 199),
]


@pytest.mark.parametrize("content,text,cap", BOUNDED_CORPUS)
def test_bounded_append_equivalence(content, text, cap):
    from kimix.tools.background import utils as mod

    def run(native: bool) -> tuple[bool, str]:
        buf = io.StringIO()
        buf.write(content)
        restore = _force_gate(mod, native)
        try:
            truncated = mod.bounded_append(buf, text, cap)
        finally:
            restore()
        return truncated, buf.getvalue()

    native = run(True)
    python = run(False)
    _assert_equivalent(native, python, (content, text, cap))
