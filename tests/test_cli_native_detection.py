"""Tests for the native-acceleration detection in ``kimix.cli_impl.core``.

``core._check_native`` logs the status of the optional native library
(``runtime_py.pyd`` on Windows / ``runtime_py.so`` on Linux & macOS, wrapped
by the ``kimix_native`` shim):
a concise info log when it loads, a concise warning when the binary is missing
or the library is invalid, and nothing when the user explicitly opted out with
``KIMIX_NATIVE=0``.
"""

from __future__ import annotations

import pytest

from kimix import native_loader
from kimix.cli_impl import core


@pytest.fixture(autouse=True)
def _unset_kimix_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the env toggle is cleared unless a test sets it explicitly."""
    monkeypatch.delenv("KIMIX_NATIVE", raising=False)


def _capture(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(core, "print_debug", infos.append)
    monkeypatch.setattr(core, "print_warning", warnings.append)
    return infos, warnings


def test_info_when_native_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native present → a brief info log, no warning."""
    monkeypatch.setattr(native_loader, "NATIVE_AVAILABLE", True)
    infos, warnings = _capture(monkeypatch)

    core._check_native()

    assert infos == ["Native acceleration enabled."]
    assert warnings == []


def test_warning_when_native_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native binary missing / library invalid → a concise warning, no info."""
    monkeypatch.setattr(native_loader, "NATIVE_AVAILABLE", False)
    infos, warnings = _capture(monkeypatch)

    core._check_native()

    assert infos == []
    assert warnings == ["Native acceleration unavailable, falling back to pure-Python."]


def test_no_log_when_native_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """KIMIX_NATIVE=0 (explicit opt-out) → no info or warning, even when unavailable."""
    monkeypatch.setenv("KIMIX_NATIVE", "0")
    monkeypatch.setattr(native_loader, "NATIVE_AVAILABLE", False)
    infos, warnings = _capture(monkeypatch)

    core._check_native()

    assert infos == []
    assert warnings == []


def test_run_cli_invokes_native_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_run_cli`` runs the native detection before dispatching subcommands."""
    checked: list[str] = []
    monkeypatch.setattr(core, "_check_native", lambda: checked.append("checked"))
    monkeypatch.setattr(core, "set_arg", lambda: ("client", object()))

    def _stop() -> None:
        raise RuntimeError("stop after _check_native")

    monkeypatch.setattr(core, "_client_cli", _stop)

    with pytest.raises(RuntimeError, match="stop after _check_native"):
        core._run_cli()

    assert checked == ["checked"]
