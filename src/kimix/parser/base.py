"""Base parser class and data models for source code comment parsers.

The canonical pure-Python reference implementations of the parser base class,
data models, and all language parsers live in
``bin/kimix_native/_parse_compat.py`` (the ``kimix_native`` shim); this module
re-exports them so there is exactly one copy of the parser logic.
"""

from __future__ import annotations

from kimix.native_loader import get_compat as _native_get_compat

# The canonical pure-Python implementation (the historical body of this
# module) lives in the kimix_native shim so there is exactly one copy of the
# parser logic.
_compat = _native_get_compat("_parse_compat")
if _compat is None:  # pragma: no cover - shim missing (unbundled install)
    raise ImportError(
        "kimix_native shim unavailable: the pure-Python comment parsers live "
        "in bin/kimix_native/_parse_compat.py and must be importable. Install "
        "the kimix package with its bundled shim or run from the repository "
        "checkout."
    )

Comment = _compat.Comment
ParseResult = _compat.ParseResult
BaseParser = _compat.BaseParser

__all__ = ["Comment", "ParseResult", "BaseParser", "native_parse_result"]


def native_parse_result(
    lang: str,
    app_language: str,
    source_code: str,
) -> ParseResult | None:
    """Route a parser invocation to the native kernel (kimix_native.parse).

    Returns an app-shaped :class:`ParseResult` (comments converted to this
    module's :class:`Comment`) when the native path is active, else None so
    the caller runs its original pure-Python body unchanged.

    Args:
        lang: native language key ("c", "python", "shell", "sql",
            "html", "lisp", "pascal").
        app_language: the language label the app parsers put into
            ``ParseResult.language`` (e.g. "C", "Python").
        source_code: source text to parse.
    """
    try:
        from kimix.native_loader import get_module, use_native
    except Exception:
        # kimix.native_loader unavailable (e.g. kimix-base's isolated
        # reference-test env loads this file into a synthetic package):
        # run the pure-Python body unchanged.
        return None

    if not use_native("PARSE"):
        return None
    mod = get_module("parse")
    if mod is None:
        return None
    result = mod.parse(lang, source_code)
    return ParseResult(
        language=app_language,
        comments=[
            Comment(
                content=c.content,
                line=c.line,
                column=c.column,
                kind=c.kind,
            )
            for c in result.comments
        ],
        code_without_comments=result.code_without_comments,
    )
