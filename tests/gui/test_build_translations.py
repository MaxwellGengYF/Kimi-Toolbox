"""Guard ``scripts/gui/build_translations.py``, which is not importable as a package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "gui" / "build_translations.py"

#: A catalog in the exact shape ``pyside6-lupdate`` emits, with a *finished* message
#: ahead of an unfinished one. That order is the whole point: the first version of
#: ``fill_identical`` used a lazy dot-all group for ``<source>``, which backtracked
#: past the finished message's ``</source>`` looking for the next
#: ``type="unfinished"``. It then wrote every intervening tag inside a
#: ``<translation>`` element, producing XML that ``lrelease`` refused to parse. The
#: mix only appears once at least one string has been translated, so the very first
#: build looked fine and the corruption started with the next added ``tr()``.
MIXED_CATALOG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1">
<context>
    <name>Probe</name>
    <message>
        <location filename="probe.py" line="+6"/>
        <source>Already done</source>
        <translation>Already done</translation>
    </message>
    <message>
        <location line="+1"/>
        <source>Still empty</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location line="+1"/>
        <source>Also empty</source>
        <translation type="unfinished"></translation>
    </message>
</context>
</TS>
"""

#: lupdate marks the ``<translation>`` element unfinished and leaves the
#: ``<numerusform>`` children bare, without a ``type`` attribute of their own.
PLURAL_CATALOG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1">
<context>
    <name>Probe</name>
    <message numerus="yes">
        <location filename="probe.py" line="+8"/>
        <source>%n file(s)</source>
        <translation type="unfinished">
            <numerusform></numerusform>
            <numerusform></numerusform>
        </translation>
    </message>
</context>
</TS>
"""


@pytest.fixture(scope="module")
def build_script() -> ModuleType:
    """Import the standalone script by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("build_translations", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _translations(text: str) -> dict[str, str]:
    """Map every ``<source>`` to its translated text, failing if the XML is broken."""
    root = ElementTree.fromstring(text)
    pairs: dict[str, str] = {}
    for message in root.iter("message"):
        source = message.find("source")
        translation = message.find("translation")
        assert source is not None and translation is not None
        pairs[source.text or ""] = translation.text or ""
    return pairs


def test_filling_a_mixed_catalog_leaves_valid_xml(build_script: ModuleType, tmp_path: Path) -> None:
    catalog = tmp_path / "kimix_gui_en.ts"
    catalog.write_text(MIXED_CATALOG, encoding="utf-8")

    filled = build_script.fill_identical(catalog)

    written = catalog.read_text(encoding="utf-8")
    # Parsing is the assertion that matters: the old implementation produced nested
    # tags here and ``ElementTree`` raised a mismatched-tag ParseError.
    assert _translations(written) == {
        "Already done": "Already done",
        "Still empty": "Still empty",
        "Also empty": "Also empty",
    }
    assert filled == 2, "only the two empty translations count as filled"
    assert 'type="unfinished"' not in written


def test_filling_is_idempotent(build_script: ModuleType, tmp_path: Path) -> None:
    catalog = tmp_path / "kimix_gui_en.ts"
    catalog.write_text(MIXED_CATALOG, encoding="utf-8")

    build_script.fill_identical(catalog)
    once = catalog.read_text(encoding="utf-8")
    assert build_script.fill_identical(catalog) == 0
    assert catalog.read_text(encoding="utf-8") == once


def test_plural_forms_are_filled_with_the_msgid(build_script: ModuleType, tmp_path: Path) -> None:
    catalog = tmp_path / "kimix_gui_en.ts"
    catalog.write_text(PLURAL_CATALOG, encoding="utf-8")

    filled = build_script.fill_identical(catalog)

    written = catalog.read_text(encoding="utf-8")
    root = ElementTree.fromstring(written)
    forms = [form.text for form in root.iter("numerusform")]
    # An empty form renders as an empty label instead of falling back to the msgid,
    # so every form has to carry text even though English needs no real translation.
    assert forms == ["%n file(s)", "%n file(s)"]
    assert filled == 2
    assert 'type="unfinished"' not in written


def test_the_repository_catalogs_are_parseable_and_complete(build_script: ModuleType) -> None:
    """The committed catalogs must stay loadable; ``lrelease`` is not forgiving."""
    catalog_dir = REPOSITORY / "src" / "kimix_gui" / "translations"
    for language in ("en", "zh_CN"):
        catalog = catalog_dir / f"kimix_gui_{language}.ts"
        assert build_script.catalog_source(language) == catalog
        assert build_script.catalog_output(language) == catalog.with_suffix(".qm")
        text = catalog.read_text(encoding="utf-8")
        assert _translations(text), f"{catalog.name} has no messages"
        assert 'type="unfinished"' not in text, f"{catalog.name} has untranslated entries"
