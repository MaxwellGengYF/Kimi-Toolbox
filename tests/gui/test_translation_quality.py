"""Enforce conventions from ``src/kimix_gui/translations/GLOSSARY.md``.

These read the committed ``.ts`` sources directly rather than relying on compiled
``.qm`` catalogs. Both failures they catch are silent at runtime: a mismatched
placeholder raises ``KeyError`` only on the code path that formats it, and half-width
punctuation just looks wrong to a reader without breaking anything.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest
import regex as re

REPOSITORY = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPOSITORY / "src" / "kimix_gui" / "translations"

#: ``{name}`` placeholders fed to ``str.format``. Every one present in a msgid has to
#: survive translation verbatim, or formatting raises ``KeyError`` at runtime.
_PLACEHOLDER = re.compile(r"\{(\w+)\}")

_HAN = re.compile(r"[\u4e00-\u9fff]")

#: Half-width marks that have a full-width counterpart in Chinese typography. Kept to
#: sentence punctuation: parentheses and quotes are left alone because they legitimately
#: wrap Latin fragments such as file extensions.
_HALF_WIDTH = re.compile(r"[?!,;:]")


def _messages(language: str) -> list[tuple[str, str]]:
    catalog = CATALOG_DIR / f"kimix_gui_{language}.ts"
    root = ElementTree.fromstring(catalog.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for message in root.iter("message"):
        source = message.find("source")
        translation = message.find("translation")
        if source is None or translation is None:
            continue
        pairs.append((source.text or "", translation.text or ""))
    return pairs


@pytest.mark.parametrize("language", ["en", "zh_CN"])
def test_placeholders_survive_translation(language: str) -> None:
    """Translations format with ``str.format``, so a renamed key is a runtime crash."""
    mismatched = [
        (source, translated)
        for source, translated in _messages(language)
        if translated and set(_PLACEHOLDER.findall(source)) != set(_PLACEHOLDER.findall(translated))
    ]
    assert not mismatched, (
        f"{language}: placeholder sets differ between msgid and translation; "
        f"str.format would raise KeyError: {mismatched}"
    )


def test_chinese_translations_use_full_width_punctuation() -> None:
    """Convention from ``src/kimix_gui/translations/GLOSSARY.md``."""
    offenders = [
        (source, translated)
        for source, translated in _messages("zh_CN")
        if _HAN.search(translated) and _HALF_WIDTH.search(translated)
    ]
    assert not offenders, f"use full-width punctuation (，。？！：；) in Chinese copy: {offenders}"


def test_no_internal_notes_leaked_into_the_catalogs() -> None:
    """A ``#:`` comment in the Qt layer becomes an ``<extracomment>`` a translator reads.

    ``pyside6-lupdate`` treats ``#:`` as an explicit translator note and attaches the
    last one it saw to the next translatable string -- however far away that is, and
    whatever the comment was actually about. It absorbed eleven internal notes that way,
    including cache-size rationales and Qt event-dispatch explanations, and filed them
    against unrelated msgids.

    So the Qt layer documents itself with plain ``#`` and the catalogs carry no notes at
    all. If a string ever does need translator guidance, that is a deliberate change:
    write the ``#:`` immediately above it and update this test to allow exactly that one.
    """
    for language in ("en", "zh_CN"):
        catalog = CATALOG_DIR / f"kimix_gui_{language}.ts"
        root = ElementTree.fromstring(catalog.read_text(encoding="utf-8"))
        leaked = [
            (message.findtext("source"), note.text)
            for message in root.iter("message")
            if (note := message.find("extracomment")) is not None
        ]
        assert leaked == [], f"{language}: {leaked}"


def test_the_qt_layer_does_not_use_sphinx_attribute_comments() -> None:
    """Stops the leak at the source rather than after the fact.

    Nothing in this repository generates documentation, so ``#:`` buys no rendering --
    it only feeds ``lupdate``. The pure layer is exempt: it holds no translatable
    strings, so nothing there can be attached to one.
    """
    offenders = []
    for path in sorted((REPOSITORY / "src" / "kimix_gui" / "qt").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#:"):
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], f"use a plain `#` comment instead: {offenders}"
