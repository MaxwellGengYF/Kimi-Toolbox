"""Refresh the translation catalogs committed inside the GUI package.

Not wired into pytest: run it after touching any ``tr()`` string, and once after a
PySide6 upgrade that changes the Linguist output. Commit source and compiled catalogs
together so clean builds contain both supported languages.

Examples:
    uv run python scripts/gui/build_translations.py                # update + release
    uv run python scripts/gui/build_translations.py --update-only  # sources -> .ts
    uv run python scripts/gui/build_translations.py --release-only  # .ts -> .qm
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import regex as re

SCRIPT = Path(__file__).resolve()
REPOSITORY = SCRIPT.parents[2]

#: Reviewable ``.ts`` sources and compiled ``.qm`` catalogs share one package directory.
TRANSLATIONS_DIR = REPOSITORY / "src" / "kimix_gui" / "translations"

#: Only the Qt layer can call ``tr()`` (the pure layer must not import PySide6), but
#: the whole package is scanned so a stray ``tr()`` outside ``qt/`` is still caught.
SCAN_DIR = REPOSITORY / "src" / "kimix_gui"

LANGUAGES = ("en", "zh_CN")

#: ``en`` msgids *are* the source text, so its catalog is filled mechanically rather
#: than by hand; see ``docs/gui/i18n.md``.
IDENTITY_LANGUAGE = "en"

#: One ``<message>`` element, matched non-greedily so blocks never merge. Filling is
#: done per block because a translation can only be built from *its own* source.
_MESSAGE = re.compile(r"<message[^>]*>.*?</message>", re.DOTALL)

#: ``<source>`` text within a single block. ``[^<]*`` rather than ``.*?`` on purpose:
#: a lazy dot-all group backtracks past ``</source>`` when the following translation
#: is already finished, swallowing whole messages and emitting nested tags. Source
#: text cannot contain a raw ``<`` (the format escapes it), so this cannot cross a tag.
_SOURCE = re.compile(r"<source>(?P<source>[^<]*)</source>")

#: A plain unfinished translation, i.e. one with no plural forms inside it.
_UNFINISHED = re.compile(r"<translation type=\"unfinished\">(?P<body>[^<]*)</translation>")

#: The opening tag of an unfinished *numerus* translation. lupdate marks the
#: ``<translation>`` element unfinished and leaves the ``<numerusform>`` children bare,
#: so both have to be handled: strip the attribute here, fill the forms below.
_UNFINISHED_PLURAL_OPEN = re.compile(r"<translation type=\"unfinished\">(?=\s*<numerusform)")
_EMPTY_PLURAL_FORM = re.compile(
    r"<numerusform(?: type=\"unfinished\")?>(?P<body>[^<]*)</numerusform>"
)


def catalog_source(language: str) -> Path:
    return TRANSLATIONS_DIR / f"kimix_gui_{language}.ts"


def catalog_output(language: str) -> Path:
    return TRANSLATIONS_DIR / f"kimix_gui_{language}.qm"


def _tool(name: str) -> str:
    """Locate a PySide6 Linguist tool on PATH, or explain how to get one."""

    found = shutil.which(name)
    if found is None:
        raise SystemExit(
            f"{name} not found on PATH.\n"
            "It ships with PySide6, so run this script through the project "
            f"environment: uv run python {SCRIPT.relative_to(REPOSITORY).as_posix()}\n"
            "If that still fails, reinstall the dependencies with `uv sync`."
        )
    return found


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{Path(command[0]).name} failed with exit code {completed.returncode}")


def _python_sources() -> list[str]:
    return [
        str(path.relative_to(REPOSITORY).as_posix())
        for path in sorted(SCAN_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def update() -> None:
    """Extract ``tr()`` strings from the package into every ``.ts`` file."""

    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    targets: list[str] = []
    for language in LANGUAGES:
        targets.append(str(catalog_source(language).relative_to(REPOSITORY).as_posix()))
    # ``-locations relative`` keeps the committed XML free of machine-specific paths
    # and of churn from unrelated edits shifting line numbers.
    _run([_tool("pyside6-lupdate"), *_python_sources(), "-locations", "relative", "-ts", *targets])
    filled = fill_identical(catalog_source(IDENTITY_LANGUAGE))
    if filled:
        print(
            f"filled {filled} identity translation(s) in {catalog_source(IDENTITY_LANGUAGE).name}"
        )


def fill_identical(path: Path) -> int:
    """Copy each source string into its own empty translation, in place.

    Keeps ``kimix_gui_en.ts`` complete without hand work: an English "translation" of
    an English msgid is the msgid. Editing the text instead of re-serializing the XML
    keeps the diff limited to the elements that actually changed.
    """

    if not path.exists():
        return 0
    original = path.read_text(encoding="utf-8")
    filled = 0

    def _fill_block(block: re.Match[str]) -> str:
        nonlocal filled
        text = block.group(0)
        source = _SOURCE.search(text)
        if source is None:
            return text
        msgid = source.group("source")

        def _fill_form(match: re.Match[str]) -> str:
            nonlocal filled
            if match.group("body").strip():
                return match.group(0)
            filled += 1
            # Every plural form of an English msgid is that msgid. Qt needs each form
            # present; an empty one renders as an empty label rather than falling back.
            return f"<numerusform>{msgid}</numerusform>"

        if _UNFINISHED_PLURAL_OPEN.search(text):
            text = _UNFINISHED_PLURAL_OPEN.sub("<translation>", text)
            return _EMPTY_PLURAL_FORM.sub(_fill_form, text)

        def _fill_single(match: re.Match[str]) -> str:
            nonlocal filled
            if match.group("body").strip():
                return match.group(0)
            filled += 1
            return f"<translation>{msgid}</translation>"

        return _UNFINISHED.sub(_fill_single, text)

    updated = _MESSAGE.sub(_fill_block, original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
    return filled


def release() -> None:
    """Compile every ``.ts`` into the package's ``.qm`` directory."""

    missing = [language for language in LANGUAGES if not catalog_source(language).exists()]
    if missing:
        raise SystemExit(
            "missing translation sources: "
            + ", ".join(catalog_source(language).name for language in missing)
            + "\nRun this script without --release-only to create them."
        )
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    tool = _tool("pyside6-lrelease")
    for language in LANGUAGES:
        _run(
            [
                tool,
                str(catalog_source(language).relative_to(REPOSITORY).as_posix()),
                "-qm",
                str(catalog_output(language).relative_to(REPOSITORY).as_posix()),
            ]
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--update-only",
        action="store_true",
        help="only run pyside6-lupdate (sources -> .ts)",
    )
    stage.add_argument(
        "--release-only",
        action="store_true",
        help="only run pyside6-lrelease (.ts -> .qm)",
    )
    args = parser.parse_args(argv)
    if not args.release_only:
        update()
    if not args.update_only:
        release()


if __name__ == "__main__":
    main(sys.argv[1:])
