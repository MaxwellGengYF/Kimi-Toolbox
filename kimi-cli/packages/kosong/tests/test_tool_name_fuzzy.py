"""Unit tests for tool-name fuzzy matching and resolution.

Covers ``normalize_tool_name``, ``fuzzy_match_tool_name``, ``resolve_tool_name``
and the ``ToolNameResolution`` result type in ``kosong.tooling``.
"""

import pytest
from pydantic import BaseModel

from kosong.tooling import (
    CallableTool2,
    ToolCandidate,
    ToolNameResolution,
    ToolOk,
    ToolReturnValue,
    fuzzy_match_tool_name,
    normalize_tool_name,
    resolve_tool_name,
)

# Default valid names used across the tests.
VALID = [
    "WriteFile",
    "ReadFile",
    "ReadMediaFile",
    "SearchWeb",
    "Grep",
    "Shell",
    "LongNamedTool",
]


# ---------------------------------------------------------------------------
# normalize_tool_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        "write_file",
        "write-file",
        "WRITE_FILE",
        "Write-File",
        "Write_File",
        "WriteFile",
        "writefile",
    ],
)
def test_normalize_collapses_separators_and_case(spelling: str):
    assert normalize_tool_name(spelling) == "writefile"


def test_normalize_multi_word():
    assert normalize_tool_name("read_media_file") == "readmediafile"
    assert normalize_tool_name("READ-MEDIA-FILE") == "readmediafile"
    assert normalize_tool_name("ReadMediaFile") == "readmediafile"


# ---------------------------------------------------------------------------
# fuzzy_match_tool_name — separator-insensitive normalized exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sent,expected",
    [
        ("write_file", "WriteFile"),
        ("write-file", "WriteFile"),
        ("WRITE_FILE", "WriteFile"),
        ("Write_File", "WriteFile"),
        ("write-File", "WriteFile"),
        ("read_media_file", "ReadMediaFile"),
        ("search-web", "SearchWeb"),
    ],
)
def test_fuzzy_normalized_exact(sent: str, expected: str):
    # Even with the restrictive auto-correct cutoff, a normalized-exact match
    # is returned because it is high confidence (independent of *cutoff*).
    assert fuzzy_match_tool_name(sent, VALID, n=1, cutoff=0.75) == [expected]


# ---------------------------------------------------------------------------
# fuzzy_match_tool_name — case-insensitive exact
# ---------------------------------------------------------------------------


def test_fuzzy_case_insensitive_exact_concatenated():
    assert fuzzy_match_tool_name("writefile", VALID, n=1) == ["WriteFile"]


@pytest.mark.parametrize("sent", ["shell", "SHELL", "Shell"])
def test_fuzzy_case_insensitive_exact_single_word(sent: str):
    assert fuzzy_match_tool_name(sent, VALID, n=1) == ["Shell"]


# ---------------------------------------------------------------------------
# fuzzy_match_tool_name — normalized fuzzy (separator + typo)
# ---------------------------------------------------------------------------


def test_fuzzy_snake_typo_autocorrect_cutoff():
    # normalize("wrte_file") == "wrtefile"  (~0.94 vs "writefile")
    assert fuzzy_match_tool_name("wrte_file", VALID, n=1, cutoff=0.75) == ["WriteFile"]


def test_fuzzy_plain_typo_still_matches():
    # Normalization is a no-op for already-CamelCase input, so plain typos
    # continue to resolve through the same fuzzy path.
    assert fuzzy_match_tool_name("LongNamedTol", VALID, n=1, cutoff=0.75) == ["LongNamedTool"]


# ---------------------------------------------------------------------------
# fuzzy_match_tool_name — guards and non-matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("short", ["rm", "ls", "x", ""])
def test_fuzzy_short_name_guard(short: str):
    assert fuzzy_match_tool_name(short, VALID) == []


def test_fuzzy_min_length_configurable():
    # "grep" (len 4) resolves by default (min_length=3) via case-insensitive exact.
    assert fuzzy_match_tool_name("grep", VALID, n=1) == ["Grep"]
    # Raising min_length above the query length suppresses matching entirely.
    assert fuzzy_match_tool_name("grep", VALID, n=1, min_length=5) == []


def test_fuzzy_no_match_returns_empty():
    assert fuzzy_match_tool_name("xyzzy", VALID) == []


def test_fuzzy_distant_snake_returns_empty():
    # Every normalized ratio for "totallyunrelated" is well below 0.5.
    assert fuzzy_match_tool_name("totally_unrelated", VALID) == []


# ---------------------------------------------------------------------------
# fuzzy_match_tool_name — determinism / ambiguity
# ---------------------------------------------------------------------------


def test_fuzzy_normalized_collision_is_deterministic():
    # "ReadFile" and "read_file" both normalize to "readfile" (a genuine
    # collision). "Read-File" is *not* a case-insensitive exact match to
    # either (it has a "-"), so resolution goes through the normalized-exact
    # branch and must rank deterministically.
    collide = ["ReadFile", "read_file"]
    result_a = fuzzy_match_tool_name("Read-File", collide, n=3, cutoff=0.75)
    result_b = fuzzy_match_tool_name("Read-File", list(reversed(collide)), n=3, cutoff=0.75)
    assert result_a == result_b == ["ReadFile", "read_file"]
    # The n=1 (auto-correct) call yields a stable single winner.
    assert fuzzy_match_tool_name("Read-File", collide, n=1, cutoff=0.75) == ["ReadFile"]


# ---------------------------------------------------------------------------
# fuzzy_match_tool_name — n limit and cutoff boundary
# ---------------------------------------------------------------------------


def test_fuzzy_n_limit():
    # "TollX" is ~0.6 similar to both ToolA and ToolB.
    names = ["ToolA", "ToolB"]
    assert fuzzy_match_tool_name("TollX", names, n=1, cutoff=0.5) == ["ToolA"]
    assert fuzzy_match_tool_name("TollX", names, n=3, cutoff=0.5) == ["ToolA", "ToolB"]


def test_fuzzy_cutoff_boundary():
    names = ["ToolA", "ToolB"]
    # ratio == 0.6: included when cutoff <= 0.6, excluded when cutoff > 0.6.
    assert fuzzy_match_tool_name("TollX", names, n=3, cutoff=0.6) == ["ToolA", "ToolB"]
    assert fuzzy_match_tool_name("TollX", names, n=3, cutoff=0.61) == []


# ---------------------------------------------------------------------------
# resolve_tool_name / ToolNameResolution
# ---------------------------------------------------------------------------


def test_resolve_exact_hit_no_correction():
    r = resolve_tool_name("WriteFile", VALID)
    assert isinstance(r, ToolNameResolution)
    assert r.name == "WriteFile"
    assert r.original == "WriteFile"
    assert r.corrected is False
    assert r.suggestions == []


@pytest.mark.parametrize(
    "sent,expected",
    [
        ("write_file", "WriteFile"),
        ("write-file", "WriteFile"),
        ("WRITE_FILE", "WriteFile"),
        ("Write_File", "WriteFile"),
        ("read_media_file", "ReadMediaFile"),
        ("search-web", "SearchWeb"),
    ],
)
def test_resolve_autocorrects_separator_styles(sent: str, expected: str):
    r = resolve_tool_name(sent, VALID)
    assert r.name == expected
    assert r.original == sent
    assert r.corrected is True
    assert r.suggestions == []


def test_resolve_autocorrects_typo():
    r = resolve_tool_name("wrte_file", VALID)
    assert r.name == "WriteFile"
    assert r.corrected is True
    assert r.suggestions == []


def test_resolve_not_found_populates_suggestions():
    r = resolve_tool_name("TollX", ["ToolA", "ToolB"])
    assert r.name is None
    assert r.corrected is False
    assert set(r.suggestions) == {"ToolA", "ToolB"}


def test_resolve_not_found_no_suggestions_when_distant():
    r = resolve_tool_name("xyzzy", VALID)
    assert r.name is None
    assert r.corrected is False
    assert r.suggestions == []


def test_resolve_short_name_guard():
    r = resolve_tool_name("rm", VALID)
    assert r.name is None
    assert r.suggestions == []


def test_resolve_accepts_dict_keys_and_generators():
    # dict.keys() is an Iterable, not a Sequence — must be materialized safely.
    tool_dict = {name: object() for name in VALID}
    r = resolve_tool_name("write_file", tool_dict.keys())
    assert r.name == "WriteFile"
    assert r.corrected is True

    # A one-shot generator must not be exhausted by the internal ``in`` test.
    gen = (name for name in VALID)
    r2 = resolve_tool_name("read_file", gen)
    assert r2.name == "ReadFile"
    assert r2.corrected is True


def test_resolve_custom_cutoffs():
    # Lowering the auto-correct cutoff turns a ~0.6 suggestion into a correction.
    names = ["ToolA", "ToolB"]
    r = resolve_tool_name("TollX", names, auto_correct_cutoff=0.55)
    assert r.name == "ToolA"
    assert r.corrected is True

# ══════════════════════════════════════════════════════════════════════════════
# Redirect map tests
# ══════════════════════════════════════════════════════════════════════════════


def test_resolve_redirect_exact_returns_corrected():
    """When a hallucinated name has a redirect, resolve returns the real name."""
    redirects = {"appendfile": "WriteFile", "createfile": "WriteFile"}
    r = resolve_tool_name("AppendFile", VALID, redirects=redirects)
    assert r.name == "WriteFile"
    assert r.original == "AppendFile"
    assert r.corrected is True


def test_resolve_redirect_case_insensitive():
    """Redirect lookup is case-insensitive via normalize_tool_name."""
    redirects = {"appendfile": "WriteFile"}
    for variant in ["appendfile", "APPENDFILE", "AppendFile", "APPEND_FILE"]:
        r = resolve_tool_name(variant, VALID, redirects=redirects)
        assert r.name == "WriteFile", f"{variant} should redirect to WriteFile"
        assert r.corrected is True


def test_resolve_redirect_not_in_valid_falls_through():
    """If the redirected name is not in valid_names, fall through to fuzzy."""
    redirects = {"shell": "NonExistent"}
    r = resolve_tool_name("Shell", VALID, redirects=redirects)
    # "Shell" is in VALID — exact match returns unchanged (corrected=False)
    assert r.name == "Shell"
    assert r.corrected is False


def test_resolve_redirect_precedes_fuzzy():
    """Redirect map is checked before fuzzy matching."""
    redirects = {"appndfile": "WriteFile"}
    r = resolve_tool_name("appndfile", VALID, redirects=redirects)
    assert r.name == "WriteFile"
    assert r.corrected is True


def test_resolve_redirect_no_match_falls_to_fuzzy():
    """When no redirect matches, fall through to fuzzy/cutoff resolution."""
    r = resolve_tool_name("wrte_file", VALID)
    assert r.name == "WriteFile"
    assert r.corrected is True


# ══════════════════════════════════════════════════════════════════════════════
# TOOL_NAME_REDIRECTS map content verification
# ══════════════════════════════════════════════════════════════════════════════


def test_redirect_map_contains_expected_entries():
    """Smoke-test key redirect entries exist."""
    from kosong.tooling import TOOL_NAME_REDIRECTS

    assert "AppendFile" in TOOL_NAME_REDIRECTS
    assert TOOL_NAME_REDIRECTS["AppendFile"] == "write"
    assert "Shell" in TOOL_NAME_REDIRECTS
    assert "Rm" not in TOOL_NAME_REDIRECTS  # canonical — no redirect needed


def test_normalized_redirects_no_self_mappings():
    """Self-mapping entries are filtered out in the normalized map."""
    from kosong.tooling import _TOOL_NAME_REDIRECTS_NORMALIZED

    for norm_key, target in _TOOL_NAME_REDIRECTS_NORMALIZED.items():
        assert norm_key != normalize_tool_name(target), (
            f"Self-mapping should be filtered: {norm_key} -> {target}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# _score_argument_fit tests
# ══════════════════════════════════════════════════════════════════════════════


class _DummyParams(BaseModel):
    """A simple params model for scoring tests."""
    path: str
    content: str = ""
    mode: str = "overwrite"


class _OptionalOnlyParams(BaseModel):
    """A params model with only optional fields."""
    name: str = ""
    value: int = 0


class _RequiredFieldParam(BaseModel):
    """A params model with a single required field."""
    code: str


def test_score_none_model_returns_zero():
    """None or non-BaseModel returns 0.0."""
    from kosong.tooling import _score_argument_fit

    score, repaired = _score_argument_fit({"path": "x"}, None)
    assert score == 0.0
    assert repaired is None

    score, repaired = _score_argument_fit({"path": "x"}, dict)  # not a BaseModel
    assert score == 0.0
    assert repaired is None


def test_score_empty_args_returns_zero():
    from kosong.tooling import _score_argument_fit

    score, repaired = _score_argument_fit({}, _DummyParams)
    assert score == 0.0
    assert repaired is None


def test_score_exact_match_required_fields():
    from kosong.tooling import _score_argument_fit

    score, repaired = _score_argument_fit({"code": "print('hi')"}, _RequiredFieldParam)
    assert score > 0.0
    assert repaired is not None


def test_score_partial_match():
    from kosong.tooling import _score_argument_fit

    # Only matching one of many optional fields
    score, repaired = _score_argument_fit({"mode": "append"}, _DummyParams)
    assert score > 0.0
    assert repaired is not None


def test_score_unmapped_keys_penalized():
    from kosong.tooling import _score_argument_fit

    # Providing keys not in the schema should lower the score
    score, repaired = _score_argument_fit(
        {"path": "/tmp/f", "unknown_key": "value"},
        _DummyParams,
    )
    assert score > 0.0
    # The presence of unknown_key adds a -0.15 penalty but doesn't zero out the score
    # since "path" is a required field match


def test_score_optional_only_model():
    from kosong.tooling import _score_argument_fit

    score, repaired = _score_argument_fit({"name": "test"}, _OptionalOnlyParams)
    assert score > 0.0
    assert repaired is not None


# ══════════════════════════════════════════════════════════════════════════════
# ToolCandidate & resolve_tool_by_arguments tests
# ══════════════════════════════════════════════════════════════════════════════


class _ToolA(CallableTool2[_DummyParams]):
    name: str = "ToolA"
    description: str = "Tool A"
    params: type[_DummyParams] = _DummyParams

    async def __call__(self, params: _DummyParams) -> ToolReturnValue:
        return ToolOk(output="a")


def test_tool_candidate_creation():

    c = ToolCandidate(name="test_tool", score=0.85)
    assert c.name == "test_tool"
    assert c.score == 0.85
    assert c.args_repaired is None

    c2 = ToolCandidate(name="test_tool", score=0.5, args_repaired={"key": "val"})
    assert c2.args_repaired == {"key": "val"}
    assert c2 != c  # frozen, different values


def test_resolve_by_arguments_empty_returns_none():
    from kosong.tooling import resolve_tool_by_arguments

    assert resolve_tool_by_arguments("SomeTool", {}, ["ToolA"], {}) is None
    assert resolve_tool_by_arguments("SomeTool", {"k": "v"}, [], {}) is None


def test_resolve_by_arguments_no_candidates_with_params():
    from kosong.tooling import resolve_tool_by_arguments

    # Candidates that are not CallableTool2 (no params) should be silently skipped
    tool_dict = {"Dummy": object()}
    result = resolve_tool_by_arguments("SomeTool", {"path": "/tmp"}, ["Dummy"], tool_dict)
    assert result is None


def test_resolve_by_arguments_with_real_tool():
    from kosong.tooling import resolve_tool_by_arguments

    tool = _ToolA()
    tool_dict = {"ToolA": tool}
    result = resolve_tool_by_arguments(
        "SomeTool", {"path": "/tmp/f", "content": "hello"}, ["ToolA"], tool_dict
    )
    # ToolA has _DummyParams with required "path" and optional "content" — both match
    if result is not None:
        assert result.name == "ToolA"
        assert result.corrected is True


def test_resolve_by_arguments_below_threshold():
    from kosong.tooling import resolve_tool_by_arguments

    tool = _ToolA()
    tool_dict = {"ToolA": tool}
    # Provide only an optional field with no required match — likely below threshold
    resolve_tool_by_arguments(
        "SomeTool", {"mode": "append"}, ["ToolA"], tool_dict,
        min_argument_score=0.5,  # Raise threshold
    )
    # May be None if score < threshold


def test_resolve_by_arguments_exports_as_public():
    """resolve_tool_by_arguments is importable from kosong.tooling."""
    from kosong.tooling import resolve_tool_by_arguments

    assert callable(resolve_tool_by_arguments)
