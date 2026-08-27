"""Command-line entry point for ``kimix-gui``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from kimix_gui.backend import SessionOptions
from kimix_gui.llm import (
    ProviderFileTarget,
    configured_selection,
    default_provider_file_path,
)

if TYPE_CHECKING:
    from kimix_gui.app import KimixGuiApp


def _load_gui_app() -> type[KimixGuiApp]:
    """Import Qt lazily and explain how to install the optional GUI runtime."""

    try:
        from kimix_gui.app import KimixGuiApp
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing != "PySide6" and not missing.startswith("PySide6."):
            raise
        raise SystemExit(
            'PySide6 is required for kimix-gui. Install it with `pip install "kimix[gui]"` '
            "or run `uv sync` in a KimiX source checkout."
        ) from None
    return KimixGuiApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PySide6 desktop GUI for Kimix")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Project working directory (default: current directory)",
    )
    parser.add_argument(
        "--session",
        help="Resume this session id, or create it when it does not exist",
    )
    parser.add_argument("--config", type=Path, help="Kimix provider JSON configuration file")
    parser.add_argument("--model", help="SDK model name")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Automatically approve SDK permission requests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    llm_selection = (
        configured_selection(
            ProviderFileTarget(args.config or default_provider_file_path(), args.model)
        )
        if args.config is not None or args.model is not None
        else None
    )
    options = SessionOptions(
        work_dir=args.work_dir,
        session_id=args.session,
        llm_selection=llm_selection,
        yolo=args.yolo,
    )
    _load_gui_app()(options).run()


if __name__ == "__main__":
    main()
