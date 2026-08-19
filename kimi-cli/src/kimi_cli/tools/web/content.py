"""Content processing helpers for web extraction.

Ported from the Hermes project's ``tools/web_tools.py``: inline base64 image
placeholder conversion, best-effort full-text storage under cache/web,
deterministic head+tail truncation with a footer pointing at the stored full
text, and the per-page character budget resolver.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import regex as re

if TYPE_CHECKING:
    from kimi_cli.config import Config

# Default per-page char budget sent to the model. Override via
# ``web.extract_char_limit`` in config.
DEFAULT_EXTRACT_CHAR_LIMIT = 15000

# Hard ceiling on the full-text file written to cache/web. A multi-MB page
# would otherwise write unbounded bytes to disk on every extract; the model
# only ever sees ``char_limit`` anyway.
MAX_STORED_TEXT_CHARS = 2_000_000

__all__ = (
    "DEFAULT_EXTRACT_CHAR_LIMIT",
    "MAX_STORED_TEXT_CHARS",
    "convert_base64_images_to_links",
    "store_full_text",
    "truncate_with_footer",
    "get_extract_char_limit",
)


def convert_base64_images_to_links(text: str) -> str:
    """Replace inline base64 image blobs with labeled markdown placeholders.

    base64 image payloads are token bombs (a single inline PNG can be tens of
    thousands of characters), so we never send the raw bytes to the model. But
    we preserve the fact that an image was there, and its alt text, as an
    inspectable placeholder. Real (http/https) markdown image links are left
    untouched so the agent can fetch/analyze them later.

    Transformations:
      ``![alt](data:image/png;base64,AAAA...)``  -> ``[IMAGE: alt]``
      ``(data:image/png;base64,AAAA...)``        -> ``[IMAGE]``
      bare ``data:image/...;base64,AAAA...``     -> ``[IMAGE]``
    """
    # 1. Markdown image with base64 source -> keep alt text, drop the blob.
    def _md_repl(m: re.Match[str]) -> str:
        alt = (m.group("alt") or "").strip()
        return f"[IMAGE: {alt}]" if alt else "[IMAGE]"

    md_b64 = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)"
    )
    out = md_b64.sub(_md_repl, text)

    # 2. Parenthesised base64 (non-markdown) and 3. bare base64 -> [IMAGE].
    out = re.sub(r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", "[IMAGE]", out)
    out = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", out)
    return out


def _cache_web_dir() -> Path:
    """Resolve the cache/web directory used for full extracted-page storage."""
    from kimi_cli.config import get_share_dir

    return get_share_dir() / "cache" / "web"


def store_full_text(url: str, content: str) -> str | None:
    """Write the full extracted page to cache/web and return its absolute path.

    Best-effort: returns ``None`` on any failure so truncated content is still
    returned to the model.
    """
    try:
        import xxhash

        cache_dir = _cache_web_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        host = (urlparse(url).hostname or "page").replace(":", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"
        digest = xxhash.xxh64(url.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"{slug}-{digest}.md"
        # Bound the stored copy so a pathologically large page can't write
        # unbounded bytes to disk. If capped, append a marker so a reader of
        # the file knows it isn't the literal complete page.
        if len(content) > MAX_STORED_TEXT_CHARS:
            content = (
                content[:MAX_STORED_TEXT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_TEXT_CHARS:,} chars "
                f"of {len(content):,}; re-extract a more specific URL for the rest ...]"
            )
        path.write_text(content, encoding="utf-8")
        return str(path)
    except Exception:
        # Storage is best-effort; the truncated model text is still usable.
        return None


def truncate_with_footer(content: str, url: str, char_limit: int) -> tuple[str, bool]:
    """Return ``(model_text, was_truncated)`` for one page's clean content.

    Pages at or under ``char_limit`` are returned whole. Larger pages get a
    head+tail window (~75% head / ~25% tail) cut on a markdown line boundary
    where possible, plus an explicit footer telling the model exactly how much
    it is seeing, where the full text is stored, and which ``read_file`` call
    pages in the omitted middle. Deterministic — no model involvement.
    """
    if len(content) <= char_limit:
        return content, False

    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget

    head = content[:head_budget]
    tail = content[-tail_budget:]
    # Snap the head cut back to the last newline so we don't slice mid-line.
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    # Snap the tail cut forward to the next newline for the same reason.
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1 :]

    total = len(content)
    stored_path = store_full_text(url, content)

    footer_lines = [
        "",
        "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {total:,} total clean characters.",
    ]
    if stored_path:
        # The omitted middle begins right after the head we're showing. Give
        # the model a concrete starting line (head line count + 1) so its first
        # read_file lands in the gap instead of guessing <line>. read_file is
        # 1-indexed; +1 moves past the last head line we already showed.
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full text saved to: {stored_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{stored_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete page; "
            f"raise/lower offset to page through it)."
        )
    else:
        footer_lines.append(
            "Full text could not be stored; re-run web_extract on a more "
            "specific URL for the complete page."
        )
    footer_lines.append("─" * 29)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    model_text += "\n" + "\n".join(footer_lines)
    return model_text, True


def get_extract_char_limit(config: Config | None = None) -> int:
    """Resolve the per-page char budget from config, clamped to a sane range.

    When ``config`` is None the user config is loaded best-effort (matching the
    provider resolution helpers); on any failure the default budget is used.
    """
    if config is None:
        try:
            from kimi_cli.config import load_config

            config = load_config()
        except Exception:
            config = None
    if config is not None and config.web.extract_char_limit is not None:
        try:
            # Floor at 2k (below that the footer dominates) and cap at 500k so
            # a typo can't blow up context.
            return max(2000, min(int(config.web.extract_char_limit), 500_000))
        except (TypeError, ValueError):
            pass
    return DEFAULT_EXTRACT_CHAR_LIMIT
