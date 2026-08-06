"""Builtin ``kimix_api`` skill renders exactly like ``skill-creator``.

``kimix_api`` (``kimi-cli/src/kimi_cli/skills/kimix_api/SKILL.md``) is shipped as a
builtin skill alongside ``skill-creator``. Because the builtin skills root is always
a discovery root, ``kimix_api`` is discovered from any working directory and rendered
by the same ``format_skills_for_prompt`` path as ``skill-creator`` — real frontmatter
description and absolute path — so the two entries in the system prompt have an
identical shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kaos.path import KaosPath

from kimi_cli.skill import (
    discover_skills_from_roots,
    format_skills_for_prompt,
    get_builtin_skills_dir,
    resolve_skills_roots,
)

BUILTIN_SKILL_NAMES = ("kimix_api", "skill-creator")
KIMIX_API_DESCRIPTION_PREFIX = "Guide for using KimiX API utilities"


def test_builtin_skills_dir_ships_kimix_api_and_skill_creator():
    """Both skills live side by side in the builtin skills root, with frontmatter
    ``name`` matching the directory name and a non-empty description."""
    root = get_builtin_skills_dir()
    for name in BUILTIN_SKILL_NAMES:
        md = root / name / "SKILL.md"
        assert md.is_file(), f"missing builtin skill: {md}"
        text = md.read_text(encoding="utf-8")
        assert f"name: {name}" in text
        assert "description:" in text


@pytest.mark.asyncio
async def test_kimix_api_discovered_as_builtin_and_rendered_like_skill_creator(
    tmp_path, monkeypatch
):
    """Discovery from an arbitrary work dir (no project skill root) finds
    ``kimix_api`` as a builtin skill and renders it with the same 3-line shape
    as ``skill-creator``: real frontmatter description, absolute path, and no
    generic fallback text."""
    import kimi_cli.skill as skill_mod

    # Isolate user/share roots so they cannot interfere; keep the real builtin root.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    monkeypatch.setattr(skill_mod, "_supports_builtin_skills", lambda: True)

    work_dir = tmp_path / "elsewhere"
    work_dir.mkdir(parents=True)

    scoped = await resolve_skills_roots(KaosPath.unsafe_from_local_path(work_dir))
    skills = await discover_skills_from_roots(scoped)
    by_name = {s.name: s for s in skills}

    kimix = by_name.get("kimix_api")
    assert kimix is not None
    assert kimix.scope == "builtin"
    assert kimix.description.startswith(KIMIX_API_DESCRIPTION_PREFIX)

    rendered = format_skills_for_prompt(skills)
    assert "### Built-in" in rendered
    assert "- kimix_api" in rendered
    assert "- skill-creator" in rendered
    # Identical entry shape (name / Path / Description triple) for both skills.
    for name in BUILTIN_SKILL_NAMES:
        assert f"- {name}\n  - Path:" in rendered
        assert f"  - Description:" in rendered
    # kimix_api keeps its real frontmatter description — no generic fallback text.
    assert "Project skill `kimix_api`" not in rendered
    # kimix_api's path points into the builtin skills root, like skill-creator's.
    assert str(get_builtin_skills_dir()) in rendered
