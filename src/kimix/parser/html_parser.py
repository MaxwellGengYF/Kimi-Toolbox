"""HTML/XML comment parser (<!-- ... --> and <? ... ?> processing instructions).

The parser implementation is the canonical pure-Python reference living in
``bin/kimix_native/_parse_compat.py`` (the ``kimix_native`` shim); this module
keeps only the native-acceleration gate, so the parser logic exists in exactly
one place."""

from __future__ import annotations

from kimix.native_loader import get_compat as _native_get_compat
from kimix.parser.base import ParseResult, native_parse_result

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

_CompatParser = _compat.HtmlParser


class HtmlParser(_CompatParser):
    """Parse HTML source code and extract comments."""

    def parse(self, source_code: str) -> ParseResult:  # noqa: C901
        # Native acceleration: kimix_native.parse.parse (html).
        native = native_parse_result("html", "HTML", source_code)
        if native is not None:
            return native
        return super().parse(source_code)
