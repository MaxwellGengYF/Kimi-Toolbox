from __future__ import annotations

from pathlib import Path

import pytest

from kimix_gui import __main__ as gui_main
from kimix_gui.backend import SessionOptions
from kimix_gui.llm import CONFIGURED_VARIANT, ProviderFileTarget


def run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> SessionOptions:
    received: list[SessionOptions] = []

    class FakeApp:
        def __init__(self, options: SessionOptions) -> None:
            received.append(options)

        def run(self) -> None:
            return None

    monkeypatch.setattr(gui_main, "_load_gui_app", lambda: FakeApp)
    gui_main.main(argv)
    assert len(received) == 1
    return received[0]


def test_cli_no_longer_accepts_thinking() -> None:
    with pytest.raises(SystemExit):
        gui_main.build_parser().parse_args(["--thinking"])


def test_cli_config_and_model_become_one_provider_file_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_file = tmp_path / "provider.json"

    options = run_main(
        monkeypatch,
        [
            "--work-dir",
            str(tmp_path),
            "--config",
            str(provider_file),
            "--model",
            "override-model",
        ],
    )

    assert options.llm_selection is not None
    assert options.llm_selection.target == ProviderFileTarget(provider_file, "override-model")
    assert options.llm_selection.variant == CONFIGURED_VARIANT


def test_cli_without_provider_flags_leaves_saved_default_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = run_main(monkeypatch, ["--work-dir", str(tmp_path)])

    assert options.llm_selection is None
