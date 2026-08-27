"""Keep the developer-documentation entry point and module map honest.

Deleting or adding a module otherwise leaves the architecture guide compiling, passing,
and wrong. Only table membership is checked: whether a row still describes what the
module does is a judgement no test can make.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE = REPOSITORY / "src" / "kimix_gui"
DOCS = REPOSITORY / "docs" / "gui"

# Entry points and package markers carry no responsibility worth a row.
_UNLISTED = {"__init__.py", "__main__.py"}

# Directories the table names as a whole, because their members share one job.
_LISTED_AS_DIRECTORIES = {"design", "components", "translations"}


def _documented_modules() -> set[str]:
    """The ``| `path` | description |`` rows of the layering table, as paths."""

    text = (DOCS / "architecture.md").read_text(encoding="utf-8")
    start = text.index("## 分层")
    end = text.index("## ", start + 1)
    return set(re.findall(r"^\| `([^`]+)` \|", text[start:end], re.MULTILINE))


def _actual_modules() -> set[str]:
    paths: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        if path.name in _UNLISTED:
            continue
        parts = path.relative_to(PACKAGE).parts
        collapsed = [i for i, part in enumerate(parts) if part in _LISTED_AS_DIRECTORIES]
        if collapsed:
            paths.add("/".join(parts[: collapsed[0] + 1]) + "/")
        else:
            paths.add("/".join(parts))
    return paths


def test_every_module_has_a_row() -> None:
    missing = sorted(_actual_modules() - _documented_modules())
    assert missing == [], f"add a row to the 分层 table in docs/gui/architecture.md for: {missing}"


def test_no_row_describes_a_module_that_is_gone() -> None:
    phantom = sorted(_documented_modules() - _actual_modules())
    assert phantom == [], f"the 分层 table in docs/gui/architecture.md still lists: {phantom}"


def test_every_topic_document_is_linked_from_the_index() -> None:
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^/)]+\.md)(?:#[^)]+)?\)", index))
    topics = {path.name for path in DOCS.glob("*.md")} - {"README.md"}
    assert linked == topics


def test_agents_points_to_the_documentation_index() -> None:
    agents = (REPOSITORY / "AGENTS.md").read_text(encoding="utf-8")
    assert "](docs/gui/README.md)" in agents
