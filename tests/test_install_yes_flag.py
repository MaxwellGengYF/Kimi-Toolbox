"""Tests for the ``-y``/``--yes`` command-line flag in install.py.

When the install script is invoked with ``-y`` every yes/no prompt must be
answered affirmatively without user input (``_ask_yes_no`` returns True
immediately), while interactive behavior is preserved when the flag is absent.
"""

from __future__ import annotations

import builtins

import pytest

import install as install_mod


@pytest.fixture(autouse=True)
def _reset_assume_yes():
    """Reset the module-level ``_ASSUME_YES`` flag around each test."""
    install_mod._ASSUME_YES = False
    yield
    install_mod._ASSUME_YES = False


@pytest.fixture
def _tty_stdin(monkeypatch):
    """Simulate an interactive terminal so the non-flag code path is exercised."""
    monkeypatch.setattr(install_mod.sys.stdin, "isatty", lambda: True)


@pytest.fixture
def _noop_install_steps(monkeypatch):
    """Stub out every step of ``main()`` that touches the real system."""
    monkeypatch.setattr(install_mod, "command_exists", lambda command: True)
    monkeypatch.setattr(install_mod, "_install_coreutils", lambda: (False, False))
    monkeypatch.setattr(install_mod, "_install_ripgrep", lambda: (False, False))
    monkeypatch.setattr(install_mod, "_install_rtk", lambda: (False, False))
    monkeypatch.setattr(install_mod, "_install_git", lambda: (False, False))
    monkeypatch.setattr(install_mod, "_sync_kimix_native_version", lambda version: True)
    monkeypatch.setattr(install_mod, "_install_kimix_native", lambda **kw: True)
    monkeypatch.setattr(install_mod, "run_command", lambda cmd, description: True)
    monkeypatch.setattr(install_mod, "_remove_gui_tool_launchers", lambda: True)


# ---------------------------------------------------------------------------
# _ask_yes_no
# ---------------------------------------------------------------------------


def test_ask_yes_no_auto_accepts_when_flag_set(_tty_stdin, monkeypatch):
    """With ``-y`` the prompt is answered 'yes' without reading stdin."""
    install_mod._ASSUME_YES = True
    # If the flag works, input() must never be called — even on a tty.
    monkeypatch.setattr(builtins, "input", lambda *a, **k: pytest.fail("input() called"))
    assert install_mod._ask_yes_no("Install it?") is True
    assert install_mod._ask_yes_no("Install it?", default=False) is True


def test_ask_yes_no_still_prompts_without_flag(_tty_stdin, monkeypatch):
    """Without ``-y`` an interactive tty still reads the user's answer."""
    monkeypatch.setattr(builtins, "input", lambda prompt: "n")
    assert install_mod._ask_yes_no("Install it?") is False
    monkeypatch.setattr(builtins, "input", lambda prompt: "y")
    assert install_mod._ask_yes_no("Install it?") is True


# ---------------------------------------------------------------------------
# main() argument parsing
# ---------------------------------------------------------------------------


def test_main_short_flag_sets_assume_yes(_noop_install_steps, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep Path("uv.lock") out of the real repo
    prompts: list[str] = []
    monkeypatch.setattr(
        install_mod, "_ask_yes_no", lambda prompt, default=True: prompts.append(prompt) or True
    )
    assert install_mod.main(["-y"]) == 0
    assert install_mod._ASSUME_YES is True
    assert prompts  # prompts were still issued, but auto-accepted


def test_main_long_flag_sets_assume_yes(_noop_install_steps, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(install_mod, "_ask_yes_no", lambda prompt, default=True: True)
    assert install_mod.main(["--yes"]) == 0
    assert install_mod._ASSUME_YES is True


def test_main_without_flag_leaves_assume_yes_false(_noop_install_steps, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(install_mod, "_ask_yes_no", lambda prompt, default=True: True)
    assert install_mod.main([]) == 0
    assert install_mod._ASSUME_YES is False


def test_main_accepts_unknown_arguments():
    """Arguments other than -y/--yes are rejected with a usage error."""
    with pytest.raises(SystemExit) as excinfo:
        install_mod.main(["--bogus"])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# GUI installation choice
# ---------------------------------------------------------------------------


def test_main_installs_gui_extra_in_both_uv_environments(
    _noop_install_steps, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(install_mod, "_ask_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(
        install_mod,
        "run_command",
        lambda command, description: commands.append(command) or True,
    )
    monkeypatch.setattr(
        install_mod,
        "_remove_gui_tool_launchers",
        lambda: pytest.fail("GUI launchers should be kept"),
    )

    assert install_mod.main([]) == 0
    assert ["uv", "sync", "--extra", "gui"] in commands
    assert ["uv", "tool", "install", "--force", "-e", ".[gui]"] in commands


def test_main_excludes_gui_dependencies_and_launcher(
    _noop_install_steps, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    commands: list[list[str]] = []
    removed_launchers: list[bool] = []

    def answer_prompt(prompt: str, default: bool = True) -> bool:
        if "PySide6 desktop GUI" in prompt:
            assert default is False
            return False
        return True

    monkeypatch.setattr(install_mod, "_ask_yes_no", answer_prompt)
    monkeypatch.setattr(
        install_mod,
        "run_command",
        lambda command, description: commands.append(command) or True,
    )
    monkeypatch.setattr(
        install_mod,
        "_remove_gui_tool_launchers",
        lambda: removed_launchers.append(True) or True,
    )

    assert install_mod.main([]) == 0
    assert ["uv", "sync", "--no-dev"] in commands
    assert ["uv", "tool", "install", "--force", "-e", "."] in commands
    assert removed_launchers == [True]


def test_remove_gui_tool_launchers_preserves_cli_launcher(monkeypatch, tmp_path):
    cli_launcher = tmp_path / "kimix.exe"
    gui_launcher = tmp_path / "kimix-gui.exe"
    gui_script = tmp_path / "kimix-gui.ps1"
    cli_launcher.touch()
    gui_launcher.touch()
    gui_script.touch()
    monkeypatch.setattr(install_mod, "_uv_tool_bin_dir", lambda: tmp_path)

    assert install_mod._remove_gui_tool_launchers() is True
    assert cli_launcher.exists()
    assert not gui_launcher.exists()
    assert not gui_script.exists()
