"""kimix_native.tools - line hashing, string find, grep line scan, security.

Native implementations live in ``runtime_py.tools`` (compiled kernels, GIL
released). The ``_compat`` mirrors below replicate the reference algorithms
from the kimi-agent repo (hash_line.py::compute_line_hash, find_str.py::
find_in_file, grep_local.py backup_grep line scanning, security.py,
file/bash/safety.py, file/bash/output_enhance.py, background/utils.py,
grep_local.py newline kernels), so ``use_native("TOOLS") is False`` yields
bit-identical behavior.

Public API:
  - line_hash(line, seed=0) -> int            (xxh32 & 0xFF of the filtered line)
  - line_hashes(content, seed=0) -> list[int] (chained per-line hashes, ints)
  - compute_line_hashes(content) -> list[str] (reference-shaped: 2-char nibble
      strings, HASH_SEED chain; exactly the reference's cumulative hashes for
      LF/CRLF files)
  - find_in_file(content, needle, case_sensitive=False, path="") -> list[dict]
      reference-shaped {file, line, column, content} records (1-based)
  - scan_lines(content, pattern, case_insensitive=True) -> list[(line_index,
      byte_offset, line_len)]   (0-based line index; literal-pattern matcher)
  - scan_lines_cb(content, callback) -> list[(line_index, byte_offset,
      line_len)]   (per-line Python matcher; offsets stay native)

Security / shell-safety kernels (plan: commit 0582e09 "Study from hermes"):
  - redact_sensitive_output(output) -> str           (WIRED in security.py)
  - scrub_child_env(env) -> dict                     (WIRED in security.py)
  - validate_workdir(workdir) -> str|None            (SHIM-ONLY: exposed +
      parity-tested; the app keeps its Python body -- per-call boundary
      overhead exceeds the gain for a trivial char scan)
  - bounded_append(content, text, cap) -> (str, bool) (WIRED in
      background/utils.py)
  - command_detection_variants(command) -> list[str]  (shim-exposed; used
      internally by check_hardline_blocked)
  - detect_hardline_command(command) -> (bool, str|None) (shim-exposed)
  - check_hardline_blocked(command) -> (bool, str|None) (WIRED in
      file/bash/safety.py)
  - foreground_background_guidance(command) -> str|None (WIRED in
      file/bash/safety.py)
  - base_command_name(command) -> str                 (kernel; used by
      interpret_exit_code)
  - interpret_exit_code(command, exit_code) -> str|None (WIRED in
      file/bash/output_enhance.py)
  - annotate_failure(output, command, exit_code) -> str|None (WIRED in
      file/bash/output_enhance.py)
  - pattern_has_regex_newline(pattern) -> bool        (WIRED in grep_local.py)
  - multiline_pattern(pattern) -> str                 (WIRED in grep_local.py)

Native-path notes (documented deviations):
  - find_in_file and scan_lines fold ASCII A-Z only (the reference uses full
    Unicode str.lower()); content or patterns with any non-ASCII byte are
    routed to _compat.
  - All security/shell/grep-pattern kernels run natively ONLY on pure-ASCII
    input (str.isascii()); non-ASCII input routes to the verbatim _compat
    mirror (Python ``regex`` \\s/\\w/\\b and .lower()/.isalpha() are
    Unicode-aware; the native scanners use ASCII [ \t\n\r\f\v] /
    [A-Za-z0-9_] / ASCII word boundary, which is bit-exact on ASCII).
  - bounded_append / validate_workdir are pure string math: any str is fine
    natively (validate_workdir also accepts None).
  - line_hash/line_hashes are Unicode-exact (the kernel embeds the Unicode
    whitespace + alnum tables) and run natively on any UTF-8 input.
"""

from __future__ import annotations

import io
import json

import regex as re
import orjson

from . import _native, use_native
from .json import _compat_indent2
from .soul import _build_structure

NIBBLE_STR = "ZPMQVRWSNKTXJBYH"
HASH_SEED = 0


# ---------------------------------------------------------------------------
# _compat: pure-Python mirrors of the reference algorithms
# ---------------------------------------------------------------------------

def _py_xxh32(data: bytes, seed: int = 0) -> int:
    """Canonical XXH32 (public domain, Yann Collet) - pure-Python reference
    mirror of xxhash.xxh32 (kept for fallback + parity checks)."""
    PRIME1 = 0x9E3779B1
    PRIME2 = 0x85EBCA77
    PRIME3 = 0xC2B2AE3D
    PRIME4 = 0x27D4EB2F
    PRIME5 = 0x165667B1
    MASK = 0xFFFFFFFF

    def rotl(x, r):
        return ((x << r) | (x >> (32 - r))) & MASK

    def xround(acc, inp):
        acc = (acc + inp * PRIME2) & MASK
        acc = rotl(acc, 13)
        acc = (acc * PRIME1) & MASK
        return acc

    n = len(data)
    p = 0
    if n >= 16:
        limit = n - 16
        v1 = (seed + PRIME1 + PRIME2) & MASK
        v2 = (seed + PRIME2) & MASK
        v3 = seed & MASK
        v4 = (seed - PRIME1) & MASK
        while p <= limit:
            v1 = xround(v1, int.from_bytes(data[p : p + 4], "little"))
            v2 = xround(v2, int.from_bytes(data[p + 4 : p + 8], "little"))
            v3 = xround(v3, int.from_bytes(data[p + 8 : p + 12], "little"))
            v4 = xround(v4, int.from_bytes(data[p + 12 : p + 16], "little"))
            p += 16
        h = (rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)) & MASK
    else:
        h = (seed + PRIME5) & MASK
    h = (h + n) & MASK
    while p + 4 <= n:
        h = (h + int.from_bytes(data[p : p + 4], "little") * PRIME3) & MASK
        h = (rotl(h, 17) * PRIME4) & MASK
        p += 4
    while p < n:
        h = (h + data[p] * PRIME5) & MASK
        h = (rotl(h, 11) * PRIME1) & MASK
        p += 1
    h ^= h >> 15
    h = (h * PRIME2) & MASK
    h ^= h >> 13
    h = (h * PRIME3) & MASK
    h ^= h >> 16
    return h & MASK


def _compat_xxh32(data: bytes, seed: int = 0) -> int:
    """Canonical XXH32 via the xxhash package (bit-identical to _py_xxh32;
    the C implementation is ~100x faster on the line-hash hot path)."""
    try:
        import xxhash

        return xxhash.xxh32(data, seed).intdigest()
    except ImportError:  # pragma: no cover - xxhash is a declared dependency
        return _py_xxh32(data, seed)


def _compat_compute_line_hash(line_num: int, line: str, prev_hash: str | None) -> str:
    """Exact mirror of hash_line.py::compute_line_hash (41-69)."""
    if line.endswith("\r"):
        line = line[:-1]
    chars: list[str] = []
    has_significant = False
    for c in line:
        if not c.isspace():
            chars.append(c)
            if not has_significant and c.isalnum():
                has_significant = True
    if prev_hash is not None:
        seed = 0
        for c in prev_hash:
            seed = ((seed * 256) + ord(c)) & 0xFFFFFFFF
    elif has_significant:
        seed = HASH_SEED
    else:
        seed = line_num
    data = "".join(chars).encode("utf-8")
    h = _compat_xxh32(data, seed) & 0xFF
    return NIBBLE_STR[h >> 4] + NIBBLE_STR[h & 0x0F]


def _compat_compute_line_hashes(content: str) -> list[str]:
    """Mirror of the native kernel: split on '\n' ONLY (a trailing '\r' is
    stripped per line by _compat_compute_line_hash). str.splitlines() also
    splits on lone '\r', which would diverge from the kernel's line stream.
    """
    hashes: list[str] = []
    prev: str | None = None
    if not content:
        return hashes
    lines = content.split("\n")
    if lines and lines[-1] == "" and content.endswith("\n"):
        lines.pop()
    for i, line in enumerate(lines, 1):
        h = _compat_compute_line_hash(i, line, prev)
        hashes.append(h)
        prev = h
    return hashes


def _compat_find_in_file(content: str, needle: str, case_sensitive: bool) -> list[dict]:
    """Exact mirror of find_str.py::find_in_file (per-line, overlapping)."""
    results = []
    lines = content.splitlines(keepends=True)
    if not case_sensitive:
        search_lower = needle.lower()
    else:
        search_lower = needle
    for line_num, line in enumerate(lines, start=1):
        line_to_search = line if case_sensitive else line.lower()
        start = 0
        while True:
            idx = line_to_search.find(search_lower, start)
            if idx == -1:
                break
            results.append({
                "file": "",
                "line": line_num,
                "column": idx + 1,
                "content": line.rstrip("\n\r"),
            })
            start = idx + 1
    return results


def _compat_scan_lines(content: str, pattern: str, case_insensitive: bool) -> list[tuple[int, int, int]]:
    """backup_grep content-mode line scan with a literal-pattern matcher."""
    if not pattern:
        return []
    needle = pattern.lower() if case_insensitive else pattern
    hits: list[tuple[int, int, int]] = []
    start = 0
    line_index = 0
    n = len(content)
    while start < n:
        nl = content.find("\n", start)
        line_end = n if nl < 0 else nl
        line = content[start:line_end]
        test = line.lower() if case_insensitive else line
        if needle in test:
            hits.append((line_index, start, line_end - start))
        start = line_end + 1 if nl >= 0 else n
        line_index += 1
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def line_hash(line: str | bytes, seed: int = 0) -> int:
    """xxh32(non-whitespace(line without trailing CR), seed) & 0xFF."""
    if isinstance(line, (bytes, bytearray)):
        data = bytes(line)
    else:
        data = str(line).encode("utf-8", "surrogatepass")
    if use_native("TOOLS") and _native is not None:
        return _native.tools.line_hash(data, seed)
    # _compat fallback replicates the kernel contract (CR strip + Unicode
    # whitespace filter + xxh32).
    text = data.decode("utf-8", "surrogatepass")
    if text.endswith("\r"):
        text = text[:-1]
    filtered = "".join(c for c in text if not c.isspace())
    return _compat_xxh32(filtered.encode("utf-8", "surrogatepass"), seed) & 0xFF


def line_hashes(content: str | bytes, seed: int = 0) -> list[int]:
    """Chained per-line hashes (ints 0..255) with reference seed semantics."""
    if isinstance(content, (bytes, bytearray)):
        data = bytes(content)
    else:
        data = str(content).encode("utf-8", "surrogatepass")
    if use_native("TOOLS") and _native is not None:
        return _native.tools.line_hashes(data, seed)
    text = data.decode("utf-8", "surrogatepass")
    return [NIBBLE_STR.index(h[0]) * 16 + NIBBLE_STR.index(h[1])
            for h in _compat_compute_line_hashes(text)]


def compute_line_hashes(content: str) -> list[str]:
    """Reference-shaped cumulative hashes (2-char nibble strings)."""
    if use_native("TOOLS") and _native is not None:
        data = content.encode("utf-8", "surrogatepass")
        hs = _native.tools.line_hashes(data, HASH_SEED)
        return [NIBBLE_STR[h >> 4] + NIBBLE_STR[h & 0x0F] for h in hs]
    return _compat_compute_line_hashes(content)


def find_in_file(
    content: str, needle: str, case_sensitive: bool = False, path: str = ""
) -> list[dict]:
    """Reference-shaped records {file, line, column, content} (1-based)."""
    if use_native("TOOLS") and _native is not None and content.isascii() and needle.isascii():
        data = content.encode("utf-8", "surrogatepass")
        ndata = needle.encode("utf-8", "surrogatepass")
        matches = _native.tools.find_in_file(data, ndata, not case_sensitive)
        lines = data.split(b"\n")
        if data.endswith(b"\n"):
            lines = lines[:-1]
        results = []
        for line_index, col, length in matches:
            line_bytes = lines[line_index] if line_index < len(lines) else b""
            results.append({
                "file": path,
                "line": line_index + 1,
                "column": col + 1,
                "content": line_bytes.rstrip(b"\n\r").decode("utf-8", "surrogatepass"),
            })
        return results
    compat = _compat_find_in_file(content, needle, case_sensitive)
    for rec in compat:
        rec["file"] = path
    return compat


def scan_lines(
    content: str | bytes, pattern: str | bytes, case_insensitive: bool = True
) -> list[tuple[int, int, int]]:
    """(line_index 0-based, byte_offset, line_len) for matching lines."""
    if isinstance(content, (bytes, bytearray)):
        cdata = bytes(content)
    else:
        cdata = str(content).encode("utf-8", "surrogatepass")
    if isinstance(pattern, (bytes, bytearray)):
        pdata = bytes(pattern)
    else:
        pdata = str(pattern).encode("utf-8", "surrogatepass")
    if use_native("TOOLS") and _native is not None and cdata.isascii() and pdata.isascii():
        return _native.tools.scan_lines(cdata, pdata, case_insensitive)
    text = cdata.decode("utf-8", "surrogatepass")
    pat = pdata.decode("utf-8", "surrogatepass")
    return _compat_scan_lines(text, pat, case_insensitive)


def scan_lines_cb(
    content: str | bytes, callback
) -> list[tuple[int, int, int]]:
    """Per-line matcher callback; offsets computed natively (regex stays in
    Python). callback(line_bytes, line_index) -> bool."""
    if _native is None:
        raise RuntimeError("scan_lines_cb requires the native extension")
    if isinstance(content, (bytes, bytearray)):
        data = bytes(content)
    else:
        data = str(content).encode("utf-8", "surrogatepass")
    return _native.tools.scan_lines_cb(data, callback)

# ---------------------------------------------------------------------------
# Security / shell-safety kernels (plan: commit 0582e09 "Study from hermes")
# ---------------------------------------------------------------------------
# The _compat_* mirrors below are VERBATIM copies of the reference algorithms
# (src/kimix/tools/security.py, src/kimix/tools/file/bash/safety.py,
# src/kimix/tools/file/bash/output_enhance.py,
# src/kimix/tools/background/utils.py, kimi-cli grep_local.py); only the
# names changed (_compat_ prefix).  The public functions gate on
# use_native("TOOLS") and route non-ASCII input to the mirrors.

_REDACTED = "[REDACTED]"

# JSON Web Tokens (header.payload.signature, header starts "eyJ").
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}")
# PEM private keys (RSA / EC / OPENSSH / DSA / ENCRYPTED variants).
_PEM_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    r".*?"
    r"-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    re.DOTALL,
)
# GitHub classic tokens (ghp_ / gho_ / ghu_ / ghr_ / ghs_) and PATs.
_GITHUB_TOKEN_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")
_GITHUB_PAT_RE = re.compile(r"github_pat_[A-Za-z0-9_]{20,}")
# GitLab personal access tokens.
_GITLAB_TOKEN_RE = re.compile(r"glpat-[A-Za-z0-9_-]{15,}")
# AWS access key IDs.
_AWS_KEY_RE = re.compile(r"AKIA[0-9A-Z]{16}")
# Authorization / API key headers.
_AUTH_HEADER_RE = re.compile(
    r"(?i)(authorization|x-api-key|apikey|proxy-authorization)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
# URL userinfo (https://user:pass@host) — keep the scheme, mask credentials.
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@")
# password= / secret: / api_key= style assignments (min value length 6).
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key)"
    r"\s*[=:]\s*(['\"]?)[^\s'\";]{6,}\2"
)
# Generic high-entropy bearer tokens.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")


def _compat_mask_userinfo(match) -> str:
    """Reference security.py::_mask_userinfo (104-105)."""
    return match.group(1) + _REDACTED + "@"


def _compat_redact_sensitive_output(output: str) -> str:
    """Exact mirror of security.py::redact_sensitive_output (108-127)."""
    if not output:
        return output
    output = _URL_USERINFO_RE.sub(_compat_mask_userinfo, output)
    output = _JWT_RE.sub(_REDACTED, output)
    output = _PEM_RE.sub(_REDACTED, output)
    output = _GITHUB_PAT_RE.sub(_REDACTED, output)
    output = _GITHUB_TOKEN_RE.sub(_REDACTED, output)
    output = _GITLAB_TOKEN_RE.sub(_REDACTED, output)
    output = _AWS_KEY_RE.sub(_REDACTED, output)
    output = _AUTH_HEADER_RE.sub(_REDACTED, output)
    output = _ASSIGNMENT_RE.sub(_REDACTED, output)
    output = _BEARER_RE.sub(_REDACTED, output)
    return output


_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                      "AUTH", "DSN", "WEBHOOK", "CREDS", "BEARER", "APIKEY")

_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM", "TMP", "TEMP", "SHELL",
                      "LOGNAME", "XDG_", "PYTHON", "VIRTUAL_ENV", "CONDA", "KIMIX_", "PROCESSOR_",
                      "PROGRAMFILES", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH", "SYSTEM",
                      "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "OS", "COMPUTERNAME",
                      "USERPROFILE", "TZ", "PWD", "SHLVL", "SSH_", "GIT_", "UV_", "PIP_")


def _compat_scrub_child_env(env: dict[str, str]) -> dict[str, str]:
    """Exact mirror of security.py::scrub_child_env (39-65)."""
    if not env:
        return {}
    scrubbed: dict[str, str] = {}
    for name, value in env.items():
        upper = name.upper()
        if any(upper.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES):
            scrubbed[name] = value
        elif any(substring in upper for substring in _SECRET_SUBSTRINGS):
            continue
        else:
            scrubbed[name] = value
    return scrubbed


_WORKDIR_ALLOWED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 _.-\\/:~"
)


def _compat_validate_workdir(workdir: str | None) -> str | None:
    """Exact mirror of security.py::validate_workdir (139-151)."""
    if not workdir:
        return None
    for char in workdir:
        if char not in _WORKDIR_ALLOWED:
            return f"Invalid workdir: character {char!r} is not allowed."
    return None


def _compat_bounded_append(content: str, text: str, cap: int) -> tuple[str, bool]:
    """Mirror of background/utils.py::bounded_append (30-50) on the
    (content, text, cap) native contract: returns (new_content, truncated)."""
    buf = io.StringIO()
    buf.write(content)
    buf.write(text)
    if buf.tell() <= cap:
        return buf.getvalue(), False
    full = buf.getvalue()
    head_len = int(cap * 0.4)
    tail_len = cap - head_len
    head = full[:head_len]
    tail = full[-tail_len:] if tail_len else ""
    marker = f"\n[... (output truncated, keeping first {head_len} and last {tail_len} chars)]\n"
    return head + marker + tail, True


_LONG_RUNNING_PATTERNS = [
    r"\b(?:npm|pnpm|yarn|bun)\s+run\s+(?:dev|start|serve|watch)\b",
    r"\bnext\s+dev\b",
    r"\bvite\b",
    r"\bnodemon\b",
    r"\buvicorn\b",
    r"\bgunicorn\b",
    r"\bpython\s+-m\s+http\.server\b",
    r"\bdocker\s+compose\s+up\b",
    r"\bdocker-compose\s+up\b",
    r"&\s*$",
    r"\bnohup\b",
    r"\bsetsid\b",
]

_FG_BG_HINT = (
    "Long-running process detected. Consider mode='send' (background) + "
    "TaskOutput to avoid blocking on timeout."
)


# Matches a regex ``\n`` escape (odd number of backslashes before ``n``).
# Even backslashes, e.g. ``\\n``, mean a literal backslash+n search.
_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")


def _compat_pattern_has_regex_newline(pattern: str) -> bool:
    """Exact mirror of grep_local.py::_pattern_has_regex_newline (86-95)."""
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


def _compat_multiline_pattern(pattern: str) -> str:
    """Exact mirror of grep_local.py::_multiline_pattern (98-114)."""
    if "\n" not in pattern and not _REGEX_NEWLINE_ESCAPE_RE.search(pattern):
        return pattern
    # Normalize explicit CRLF in the pattern, then rewrite real newlines and
    # regex ``\n`` escapes to ``\r?\n``.  Lambdas keep the replacement text
    # literal (re.sub would otherwise interpret ``\r``/``\n`` escapes).
    p = pattern.replace("\r\n", "\n")
    p = _REGEX_NEWLINE_ESCAPE_RE.sub(lambda _m: r"\r?\n", p)
    return p.replace("\n", r"\r?\n")


def _compat_command_detection_variants(command: str) -> list[str]:
    """Exact mirror of safety.py::command_detection_variants (24-46)."""
    if not command or not command.strip():
        return []
    collapsed = " ".join(command.split())
    deobfuscated = re.sub(r"[\\'\"]", "", collapsed).lower()
    lowered = collapsed.lower()
    variants: list[str] = []
    for variant in (collapsed, deobfuscated, lowered):
        if variant and variant not in variants:
            variants.append(variant)
    return variants or [collapsed]


def _compat_segment_tokens(text: str, start: int) -> list[str]:
    """Exact mirror of safety.py::_segment_tokens (49-57)."""
    tail = text[start:]
    tail = re.split(r";|\|\||&&|\||\n", tail, maxsplit=1)[0]
    return tail.split()


def _compat_looks_like_flag(token: str) -> bool:
    """Exact mirror of safety.py::_looks_like_flag (60-67)."""
    if token.startswith("-") and len(token) > 1:
        return True
    if token.startswith("/") and len(token) > 1 and token[1:].isalpha():
        return True
    return False


def _compat_collect_flags(tokens: list[str]) -> set[str]:
    """Exact mirror of safety.py::_collect_flags (70-86)."""
    flags: set[str] = set()
    for token in tokens:
        if not _compat_looks_like_flag(token):
            continue
        core = token.lstrip("-/")
        if not core:
            continue
        if "recursive" in core:
            flags.add("r")
        if "force" in core:
            flags.add("f")
        for char in core:
            if char in "rfsq":
                flags.add(char)
    return flags


def _compat_rm_target_is_protected(target: str) -> bool:
    """Exact mirror of safety.py::_rm_target_is_protected (89-107)."""
    t = target.strip().strip("\"'").lower()
    t = t.replace("${home}", "$home")
    if t.rstrip("/\\") in ("~", "$home"):
        return True
    # Windows drive root, optionally with trailing separator and/or glob.
    if re.match(r"^[a-z]:[\\/]?(?:[\\/]?\*)?$", t):
        return True
    if t.startswith("/"):
        parts = [p for p in t.split("/") if p not in ("", ".", "..")]
        if not parts or parts == ["*"]:
            return True
    return False


def _compat_detect_recursive_delete(text: str) -> str | None:
    """Exact mirror of safety.py::_detect_recursive_delete (110-126)."""
    for match in re.finditer(r"\b(rm|rmdir|del)(?:\.exe)?\b", text):
        command_word = match.group(1)
        tokens = _compat_segment_tokens(text, match.end())
        flags = _compat_collect_flags(tokens)
        if command_word == "rm" and not ({"r", "f"} & flags):
            continue
        if command_word == "rmdir" and not ({"r", "s"} & flags):
            continue
        if command_word == "del" and not ({"r", "f", "s"} & flags):
            continue
        targets = [t for t in tokens if not _compat_looks_like_flag(t)]
        for target in targets:
            if _compat_rm_target_is_protected(target):
                return f"Recursive delete of protected root/home (`{target}`)"
    return None


def _compat_detect_hardline_command(command: str) -> tuple[bool, str | None]:
    """Exact mirror of safety.py::detect_hardline_command (129-179)."""
    if not command or not command.strip():
        return False, None
    text = " ".join(command.split()).lower()

    # 1. Recursive delete of root / home / Windows drive root.
    desc = _compat_detect_recursive_delete(text)
    if desc is not None:
        return True, desc

    # 2. Disk formatting (mkfs.* formats devices).
    if re.search(r"\bmkfs(?:\.\w+)?\b", text):
        return True, "Disk formatting command (`mkfs`) is blocked"

    # 3. dd writing to a raw device (of=/dev/sd*, nvme*, disk*, rdisk*).
    if re.search(r"\bdd\b", text) and re.search(
        r"\bof=/dev/(?:sd|nvme|disk|rdisk)[a-z0-9]*", text
    ):
        return True, "`dd` writing to a raw device is blocked"

    # 4. System power commands: shutdown / reboot / poweroff / halt.
    first = text.split()[0] if text.split() else ""
    if first in ("shutdown", "reboot", "poweroff", "halt"):
        return True, f"System `{first}` command is blocked"

    # 5. Fork bomb: `:(){ :|:& };:`
    if re.search(r":\(\)\{", text) and re.search(r":\|:&", text):
        return True, "Fork bomb pattern detected"

    # 6. kill targeting PID 1 (or $PPID — kills the parent shell).
    for match in re.finditer(r"\bkill(?:\.exe)?\b", text):
        tokens = _compat_segment_tokens(text, match.end())
        targets = [t for t in tokens if not _compat_looks_like_flag(t)]
        for target in targets:
            if target == "1" or target == "$ppid":
                return True, "`kill` targeting PID 1 (or `$PPID`) is blocked"

    # 7. Windows: format <drive>: and del /f /s /q <drive>:\*.
    for match in re.finditer(r"\bformat(?:\.exe)?\b", text):
        tokens = _compat_segment_tokens(text, match.end())
        for target in tokens:
            if re.match(r"^[a-z]:[\\/]?$", target):
                return True, "Windows `format` on a drive is blocked"

    return False, None


def _compat_check_hardline_blocked(command: str) -> tuple[bool, str | None]:
    """Exact mirror of safety.py::check_hardline_blocked (182-193)."""
    for variant in _compat_command_detection_variants(command):
        blocked, desc = _compat_detect_hardline_command(variant)
        if blocked:
            return True, desc
    return False, None


def _compat_strip_quoted(text: str) -> str:
    """Exact mirror of safety.py::_strip_quoted (222-225)."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", text)


def _compat_foreground_background_guidance(command: str) -> str | None:
    """Exact mirror of safety.py::foreground_background_guidance (228-241)."""
    if not command or not command.strip():
        return None
    stripped = _compat_strip_quoted(command)
    text = " ".join(stripped.split())
    if any(re.search(pattern, text) for pattern in _LONG_RUNNING_PATTERNS):
        return _FG_BG_HINT
    return None


def _compat_base_command_name(command: str) -> str:
    """Exact mirror of output_enhance.py::_base_command_name (28-41)."""
    last_segment = command.strip().split("&&")[-1].split("||")[-1]
    last_segment = last_segment.split("|")[-1].split(";")[-1].strip()
    for word in last_segment.split():
        if "=" in word and not word.startswith("-"):
            continue
        stem = word.split("/")[-1]
        return stem[:-4] if stem.lower().endswith(".exe") else stem
    return ""


def _compat_interpret_exit_code(command: str, exit_code: int | None) -> str | None:
    """Exact mirror of output_enhance.py::interpret_exit_code (44-75)."""
    if exit_code is None or exit_code == 0:
        return None
    name = _compat_base_command_name(command).lower()
    code = exit_code

    if name in ("grep", "egrep", "fgrep", "rg", "ag", "ack") and code == 1:
        return "No matches found (not an error)"
    if name in ("diff", "colordiff") and code == 1:
        return "Files differ (expected, not an error)"
    if name == "find" and code == 1:
        return "Some directories were inaccessible (partial results may still be valid)"
    if name in ("test", "[") and code == 1:
        return "Condition evaluated to false (expected, not an error)"
    if name == "curl":
        notes = {
            6: "Could not resolve host (DNS failure)",
            7: "Failed to connect to host",
            22: "HTTP error (server returned an error status)",
            28: "Connection timed out",
        }
        if code in notes:
            return notes[code]
    if name == "git" and code == 1:
        return "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"
    return None


def _compat_annotate_failure(output: str, command: str, exit_code: int | None) -> str | None:
    """Exact mirror of output_enhance.py::annotate_failure (78-114)."""
    if not output:
        return None
    sample = output[:4000]
    lowered = sample.lower()

    if (
        "command not found" in lowered
        or "not recognized as an internal or external command" in lowered
    ):
        return (
            "The command was not found. Check it is installed and on PATH "
            "(use `which <cmd>` / `Get-Command <cmd>`)."
        )
    if "no such file or directory" in lowered:
        return (
            "A file or directory referenced by the command does not exist. "
            "Verify the path with `Glob`/ReadFile."
        )
    module_match = re.search(
        r"modulenotfounderror:\s*no module named '([^']+)'", sample, re.IGNORECASE
    )
    if module_match:
        missing = module_match.group(1)
        return (
            f"Python module {missing} is missing. Install it "
            f"(e.g. `pip install {missing}`) or check the environment."
        )
    if "permission denied" in lowered:
        return "Permission denied. Check file permissions (ls -la) or ownership."
    return None


# ---------------------------------------------------------------------------
# Public gated API (security / shell-safety kernels)
# ---------------------------------------------------------------------------

def redact_sensitive_output(output: str) -> str:
    """Mask credentials with ``[REDACTED]`` (10 chained redactions)."""
    if not output:
        return output
    if use_native("TOOLS") and _native is not None and output.isascii():
        return _native.tools.redact_sensitive_output(output)
    return _compat_redact_sensitive_output(output)


def scrub_child_env(env: dict[str, str]) -> dict[str, str]:
    """Copy of *env* with credential-looking variables removed (order kept)."""
    if use_native("TOOLS") and _native is not None and all(k.isascii() for k in env):
        return _native.tools.scrub_child_env(env)
    return _compat_scrub_child_env(env)


def validate_workdir(workdir: str | None) -> str | None:
    """SHIM-ONLY: None when *workdir* is safe, else the reference error message.
    Accepts any str (and None) natively -- pure string math."""
    if use_native("TOOLS") and _native is not None:
        return _native.tools.validate_workdir(workdir)
    return _compat_validate_workdir(workdir)


def bounded_append(content: str, text: str, cap: int) -> tuple[str, bool]:
    """Append *text* to *content*, bounding retained output to *cap* chars;
    returns (new_content, truncated).  Any str is fine natively."""
    if use_native("TOOLS") and _native is not None:
        return _native.tools.bounded_append(content, text, cap)
    return _compat_bounded_append(content, text, cap)


def command_detection_variants(command: str) -> list[str]:
    """Deobfuscation variants used to defeat quoting tricks (at most 3)."""
    if use_native("TOOLS") and _native is not None and command.isascii():
        return _native.tools.command_detection_variants(command)
    return _compat_command_detection_variants(command)


def detect_hardline_command(command: str) -> tuple[bool, str | None]:
    """(True, description) when *command* matches a hardline pattern."""
    if use_native("TOOLS") and _native is not None and command.isascii():
        return _native.tools.detect_hardline_command(command)
    return _compat_detect_hardline_command(command)


def check_hardline_blocked(command: str) -> tuple[bool, str | None]:
    """Single entry point: detector over every deobfuscation variant."""
    if use_native("TOOLS") and _native is not None and command.isascii():
        return _native.tools.check_hardline_blocked(command)
    return _compat_check_hardline_blocked(command)


def foreground_background_guidance(command: str) -> str | None:
    """Hint when *command* looks long-lived, else None (quotes ignored)."""
    if use_native("TOOLS") and _native is not None and command.isascii():
        return _native.tools.foreground_background_guidance(command)
    return _compat_foreground_background_guidance(command)


def base_command_name(command: str) -> str:
    """First non-assignment command word, directory-stripped, .exe removed."""
    if use_native("TOOLS") and _native is not None and command.isascii():
        return _native.tools.base_command_name(command)
    return _compat_base_command_name(command)


def interpret_exit_code(command: str, exit_code: int | None) -> str | None:
    """Explain a non-zero exit code for well-known commands, else None."""
    if use_native("TOOLS") and _native is not None and command.isascii():
        return _native.tools.interpret_exit_code(command, exit_code)
    return _compat_interpret_exit_code(command, exit_code)


def annotate_failure(output: str, command: str, exit_code: int | None) -> str | None:
    """Single actionable hint for common failure signatures, else None."""
    if use_native("TOOLS") and _native is not None and output.isascii():
        return _native.tools.annotate_failure(output, command, exit_code)
    return _compat_annotate_failure(output, command, exit_code)


def pattern_has_regex_newline(pattern: str) -> bool:
    """True when a search regex tries to match a newline (literal or escape)."""
    if use_native("TOOLS") and _native is not None and pattern.isascii():
        return _native.tools.pattern_has_regex_newline(pattern)
    return _compat_pattern_has_regex_newline(pattern)


def multiline_pattern(pattern: str) -> str:
    """Rewrite newline constructs so the pattern also matches CRLF."""
    if use_native("TOOLS") and _native is not None and pattern.isascii():
        return _native.tools.multiline_pattern(pattern)
    return _compat_multiline_pattern(pattern)

# ---------------------------------------------------------------------------
# Plan 016: session export markdown builder.
# ---------------------------------------------------------------------------

NL = chr(10)


def _exp_shorten(text: str, width: int) -> str:
    """kimi_cli/utils/string.py::shorten (whitespace-normalize + truncate)."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    cut = width - 1  # placeholder "..." (U+2026)
    if cut <= 0:
        return text[:width]
    space = text.rfind(" ", 0, cut + 1)
    if space > 0:
        cut = space
    return text[:cut].rstrip() + chr(0x2026)


def _exp_part_md(part: dict, stringify: bool = False) -> str:
    ptype = part.get("type")
    if ptype == "text":
        return part.get("text", "")
    if ptype == "think":
        think = part.get("think", "")
        if stringify:
            return "[think]"
        if not think.strip():
            return ""
        return "<details><summary>Thinking</summary>" + NL + NL + think + NL + NL + "</details>"
    if ptype == "image_url":
        return "[image]"
    if ptype == "audio_url":
        if stringify:
            aid = (part.get("audio_url") or {}).get("id")
            return "[audio:" + str(aid) + "]" if aid else "[audio]"
        return "[audio]"
    if ptype == "video_url":
        return "[video]"
    return "[" + ptype + "]"


def _exp_message_stringify(msg: dict) -> str:
    return "".join(_exp_part_md(p, stringify=True) for p in (msg.get("content") or []))


def _exp_internal_user(msg: dict) -> bool:
    content = msg.get("content") or []
    if msg.get("role") != "user" or len(content) != 1:
        return False
    part = content[0]
    text = part.get("text", "") if part.get("type") == "text" else ""
    return (
        text.strip().startswith("<system>CHECKPOINT")
        or text.strip().startswith("<system-reminder>")
        or text.lstrip().startswith("<notification ")
    )


_EXP_HINT_KEYS = ("path", "file_path", "command", "query", "url", "name", "pattern")


def _loads(s: str | bytes):
    """orjson-fast parse (str or bytes); stdlib fallback keeps lone-surrogate
    parity."""
    try:
        return orjson.loads(s)
    except ValueError:
        return json.loads(s)


def _exp_extract_hint(args_json: str) -> str:
    try:
        parsed = _loads(args_json)
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    for key in _EXP_HINT_KEYS:
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            return _exp_shorten(val, 60)
    for val in parsed.values():
        if isinstance(val, str) and 0 < len(val) <= 80:
            return _exp_shorten(val, 60)
    return ""


def _exp_format_tool_call_md(tc: dict) -> str:
    args_raw = (tc.get("function") or {}).get("arguments") or "{}"
    hint = _exp_extract_hint(args_raw)
    title = "#### Tool Call: " + str((tc.get("function") or {}).get("name", ""))
    if hint:
        title += " (`" + hint + "`)"
    try:
        parsed = _loads(args_raw)
        args_formatted = _compat_indent2(parsed).decode("utf-8", "surrogatepass")
    except ValueError:
        args_formatted = args_raw
    return (title + NL + "<!-- call_id: " + str(tc.get("id", "")) + " -->"
            + NL + "```json" + NL + args_formatted + NL + "```")


def _exp_format_tool_result_md(msg: dict, tool_name: str, hint: str) -> str:
    call_id = msg.get("tool_call_id") or "unknown"
    result_parts = []
    for part in msg.get("content") or []:
        text = _exp_part_md(part)
        if text.strip():
            result_parts.append(text)
    result_text = NL.join(result_parts)
    summary = "Tool Result: " + str(tool_name)
    if hint:
        summary += " (`" + str(hint) + "`)"
    return ("<details><summary>" + summary + "</summary>" + NL + NL
            + "<!-- call_id: " + call_id + " -->" + NL
            + result_text + NL + NL + "</details>")


def _exp_group_turns(history: list) -> list:
    turns = []
    current = []
    for msg in history:
        if _exp_internal_user(msg):
            continue
        if msg.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def _exp_format_turn_md(messages: list, turn_number: int) -> str:
    lines = ["## Turn " + str(turn_number), ""]
    tool_call_info = {}
    assistant_header_written = False
    for msg in messages:
        if _exp_internal_user(msg):
            continue
        if msg.get("role") == "user":
            lines.append("### User")
            lines.append("")
            for part in msg.get("content") or []:
                text = _exp_part_md(part)
                if text.strip():
                    lines.append(text)
                    lines.append("")
        elif msg.get("role") == "assistant":
            if not assistant_header_written:
                lines.append("### Assistant")
                lines.append("")
                assistant_header_written = True
            for part in msg.get("content") or []:
                text = _exp_part_md(part)
                if text.strip():
                    lines.append(text)
                    lines.append("")
            for tc in msg.get("tool_calls") or []:
                args_raw = (tc.get("function") or {}).get("arguments") or "{}"
                hint = _exp_extract_hint(args_raw)
                tool_call_info[tc.get("id", "")] = (
                    (tc.get("function") or {}).get("name", ""),
                    hint,
                )
                lines.append(_exp_format_tool_call_md(tc))
                lines.append("")
        elif msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id") or ""
            name, hint = tool_call_info.get(tc_id, ("unknown", ""))
            lines.append(_exp_format_tool_result_md(msg, name, hint))
            lines.append("")
        elif msg.get("role") in ("system", "developer"):
            lines.append("### " + str(msg.get("role")).capitalize())
            lines.append("")
            for part in msg.get("content") or []:
                text = _exp_part_md(part)
                if text.strip():
                    lines.append(text)
                    lines.append("")
    return NL.join(lines)


def _compat_build_export_markdown(history: list, opts: dict) -> bytes:
    token_count = int(opts.get("token_count", 0))
    lines = ["---",
             "session_id: " + str(opts.get("session_id", "")),
             "exported_at: " + str(opts.get("exported_at", "")),
             "work_dir: " + str(opts.get("work_dir", "")),
             "message_count: " + str(len(history)),
             "token_count: " + str(token_count),
             "---", "", "# Kimi Session Export", ""]

    turns = _exp_group_turns(history)

    topic = ""
    for msg in history:
        if msg.get("role") == "user" and not _exp_internal_user(msg):
            topic = _exp_shorten(_exp_message_stringify(msg), 80)
            break
    n_tool_calls = sum(len(m.get("tool_calls") or []) for m in history)

    comma_tokens = format(token_count, ",")
    overview = ["## Overview", "",
                ("- **Topic**: " + topic) if topic else "- **Topic**: (empty)",
                "- **Conversation**: " + str(len(turns)) + " turns | "
                + str(n_tool_calls) + " tool calls | " + comma_tokens + " tokens",
                "", "---"]
    lines.append(NL.join(overview))
    lines.append("")
    for idx, turn_messages in enumerate(turns):
        lines.append(_exp_format_turn_md(turn_messages, idx + 1))
    return NL.join(lines).encode("utf-8", "surrogatepass")


def build_export_markdown(history: list, opts: dict) -> bytes:
    """Full session export markdown (export.py build_export_markdown)."""
    if use_native("TOOLS") and _native is not None:
        buf, structure = _build_structure(history)
        return bytes(_native.tools.build_export_markdown(buf, structure, dict(opts)))
    return _compat_build_export_markdown(history, opts)
