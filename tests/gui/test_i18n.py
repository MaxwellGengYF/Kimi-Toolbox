"""Pure-layer language resolution plus catalogue completeness.

The ``.ts`` sources are tracked; the compiled ``.qm`` catalogs are gitignored build
output, so their presence test points at the build script rather than at a commit.

Nothing here imports PySide6: ``resolve_language`` is the whole ``auto`` decision,
so it stays coverable with plain pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kimix_gui.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_PREFERENCES,
    SUPPORTED_LANGUAGES,
    language_for_locale,
    normalize_language_preference,
    resolve_language,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPOSITORY / "src" / "kimix_gui" / "translations"


def test_i18n_module_stays_free_of_qt() -> None:
    source = (REPOSITORY / "src" / "kimix_gui" / "i18n.py").read_text(encoding="utf-8")

    assert "PySide6" not in source


@pytest.mark.parametrize(
    ("system_locale", "expected"),
    [
        ("zh_CN", "zh_CN"),
        # Every Chinese variant folds into Simplified: no Traditional catalog yet,
        # and Simplified still beats English for a Chinese reader.
        ("zh_TW", "zh_CN"),
        ("zh_HK", "zh_CN"),
        ("zh", "zh_CN"),
        ("zh-Hans-CN", "zh_CN"),
        ("ZH_cn", "zh_CN"),
        ("en_US", "en"),
        ("en_GB", "en"),
        ("de_DE", "en"),
        ("ja_JP", "en"),
        ("C", "en"),
        ("", "en"),
        ("   ", "en"),
    ],
)
def test_auto_follows_the_system_locale(system_locale: str, expected: str) -> None:
    assert resolve_language("auto", system_locale) == expected
    assert language_for_locale(system_locale) == expected


@pytest.mark.parametrize("system_locale", ["zh_CN", "en_US", "de_DE", ""])
def test_an_explicit_language_ignores_the_system_locale(system_locale: str) -> None:
    assert resolve_language("en", system_locale) == "en"
    assert resolve_language("zh_CN", system_locale) == "zh_CN"


@pytest.mark.parametrize("preference", ["", "  ", "kl_GL", "zh_TW", "EN-GB", None, 7])
def test_unusable_preferences_resolve_to_english(preference: object) -> None:
    # A resolvable-looking but unsupported code (``zh_TW``) lands here too. The
    # picker cannot produce one; only a hand-edited file can, and
    # ``normalize_language_preference`` turns it into ``auto`` on the way in, so the
    # locale branch still gets its chance before this fallback applies.
    assert resolve_language(preference, "zh_CN") == "en"  # type: ignore[arg-type]
    assert DEFAULT_LANGUAGE == "en"


def test_surrounding_whitespace_does_not_hide_the_auto_preference() -> None:
    assert resolve_language(" auto ", "zh_CN") == "zh_CN"
    assert resolve_language(" zh_CN ", "en_US") == "zh_CN"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("en", "en"),
        ("zh_CN", "zh_CN"),
        ("zh-cn", "zh_CN"),
        # Unknown values behave like "never chosen", which is ``auto`` -- the
        # documented default of the setting, not an explicit English choice.
        ("zh_TW", "auto"),
        ("klingon", "auto"),
        ("", "auto"),
        (None, "auto"),
        (13, "auto"),
    ],
)
def test_stored_preferences_normalize_into_the_offered_choices(
    stored: object, expected: str
) -> None:
    normalized = normalize_language_preference(stored)

    assert normalized == expected
    assert normalized in LANGUAGE_PREFERENCES


def test_every_supported_language_has_a_committed_source_catalog() -> None:
    for language in SUPPORTED_LANGUAGES:
        assert (CATALOG_DIR / f"kimix_gui_{language}.ts").is_file()


def test_every_supported_language_has_a_compiled_catalog() -> None:
    """``.qm`` files are build output, so a fresh checkout has to run the script.

    Missing catalogs are silent at runtime (``tr()`` falls back to English), which is
    why this is a hard failure carrying the exact command instead of a skip.
    """

    for language in SUPPORTED_LANGUAGES:
        assert (CATALOG_DIR / f"kimix_gui_{language}.qm").is_file(), (
            f"kimix_gui_{language}.qm is missing; run "
            "uv run python scripts/gui/build_translations.py"
        )


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_no_translation_is_left_unfinished(language: str) -> None:
    """Untranslated entries silently fall back to English, so make them loud here.

    Applies to ``en`` too: ``scripts/gui/build_translations.py`` fills that catalog
    mechanically (an English translation of an English msgid is the msgid), so a
    leftover ``unfinished`` there means the build step was skipped.
    """

    catalog = (CATALOG_DIR / f"kimix_gui_{language}.ts").read_text(encoding="utf-8")

    assert 'type="unfinished"' not in catalog, (
        f"kimix_gui_{language}.ts has untranslated entries; translate them, then run "
        "uv run python scripts/gui/build_translations.py"
    )
