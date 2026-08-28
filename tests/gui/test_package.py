"""Distribution-level contracts for the optional desktop package."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import kimix_gui
from kimix_gui.backend import SessionOptions

REPOSITORY = Path(__file__).resolve().parents[2]


def test_gui_version_matches_the_containing_distribution() -> None:
    assert kimix_gui.__version__ == version("kimix")


def test_public_session_options_export_is_lazy_but_compatible() -> None:
    assert kimix_gui.SessionOptions is SessionOptions


def test_packaging_exposes_gui_without_making_qt_mandatory() -> None:
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["optional-dependencies"]["gui"] == ["PySide6>=6.8"]
    assert "PySide6>=6.8" in project["project"]["optional-dependencies"]["all"]
    assert all(
        not dependency.lower().startswith("pyside6")
        for dependency in project["project"]["dependencies"]
    )
    assert project["project"]["scripts"]["kimix-gui"] == "kimix_gui.__main__:main"
    assert "src/kimix_gui" in project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert {"PySide6>=6.8", "pytest-qt>=4.5"} <= set(project["dependency-groups"]["dev"])


def test_distribution_requires_the_matching_cli_release() -> None:
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    cli_project = tomllib.loads(
        (REPOSITORY / "kimi-cli" / "pyproject.toml").read_text(encoding="utf-8")
    )
    kosong_project = tomllib.loads(
        (REPOSITORY / "kimi-cli" / "packages" / "kosong" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    cli_version = cli_project["project"]["version"]
    kosong_version = kosong_project["project"]["version"]
    assert f"kimi-cli-x>={cli_version}" in project["project"]["dependencies"]
    assert f"kosong-x=={kosong_version}" in cli_project["project"]["dependencies"]


def test_importing_the_optional_package_does_not_load_qt() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kimix_gui; print(any(n == 'PySide6' for n in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False"


def test_gui_command_explains_how_to_install_missing_qt() -> None:
    code = """
import importlib.abc
import sys

from kimix_gui.__main__ import main


class BlockPySide(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PySide6" or fullname.startswith("PySide6."):
            raise ModuleNotFoundError("blocked for test", name=fullname)
        return None


sys.meta_path.insert(0, BlockPySide())
main([])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert 'pip install "kimix[gui]"' in completed.stderr
