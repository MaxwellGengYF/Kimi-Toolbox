"""Tests for the kimix_base native runtime download+unpack logic in install.py.

Covers the archive naming rule (``kimix_base-<platform>-<arch>-<version>.zip``),
platform/arch detection, download, zip extraction (standard-library ``zipfile``,
mirroring the ripgrep/rtk installers), staging into ``bin/``, the
``KIMIX_NATIVE=1`` verification, and the delete-the-downloaded-archive-on-
success contract.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

import install as install_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_BIN = REPO_ROOT / "bin"

# The staged artifact name is platform-dependent (runtime_py.pyd on Windows /
# runtime_py.so on Linux & macOS) — mirrors install_mod._native_files().
_NATIVE_FILE = "runtime_py.pyd" if sys.platform == "win32" else "runtime_py.so"


# ---------------------------------------------------------------------------
# naming rule / URL / platform detection
# ---------------------------------------------------------------------------


def test_native_files_is_platform_aware(monkeypatch):
    """The staged artifact name follows the platform (mirrors kimix-base
    publish.py): .pyd on Windows, .so on Linux & macOS."""
    monkeypatch.setattr(install_mod.sys, "platform", "win32")
    assert install_mod._native_files() == ("runtime_py.pyd",)
    assert install_mod.KIMIX_BASE_NATIVE_FILES == ("runtime_py.pyd",)
    monkeypatch.setattr(install_mod.sys, "platform", "linux")
    assert install_mod._native_files() == ("runtime_py.so",)
    monkeypatch.setattr(install_mod.sys, "platform", "darwin")
    assert install_mod._native_files() == ("runtime_py.so",)


def test_archive_name_rule():
    version = install_mod.KIMIX_BASE_VERSION
    assert (
        install_mod._kimix_base_archive_name("windows", "x64")
        == f"kimix_base-windows-x64-{version}.zip"
    )
    assert (
        install_mod._kimix_base_archive_name("linux", "arm64", "0.3.0")
        == "kimix_base-linux-arm64-0.3.0.zip"
    )
    assert (
        install_mod._kimix_base_archive_name("macos", "x64")
        == f"kimix_base-macos-x64-{version}.zip"
    )


def test_download_url():
    version = install_mod.KIMIX_BASE_VERSION
    url = install_mod._kimix_base_download_url("windows", "x64")
    assert url == (
        "https://github.com/Sikao-Engine/KimiX-native/releases/download/Release/"
        f"kimix_base-windows-x64-{version}.zip"
    )


@pytest.mark.parametrize(
    ("platform_name", "machine", "expected"),
    [
        ("win32", "AMD64", ("windows", "x64")),
        ("win32", "x86_64", ("windows", "x64")),
        ("linux", "x86_64", ("linux", "x64")),
        ("linux", "aarch64", ("linux", "arm64")),
        ("darwin", "arm64", ("macos", "arm64")),
        ("linux", "x86", ("linux", "x86")),
    ],
)
def test_platform_arch_detection(monkeypatch, platform_name, machine, expected):
    monkeypatch.setattr(install_mod.sys, "platform", platform_name)
    monkeypatch.setattr(install_mod.platform, "machine", lambda: machine)
    assert install_mod._kimix_base_platform_arch() == expected


def test_platform_arch_detection_unsupported(monkeypatch):
    monkeypatch.setattr(install_mod.sys, "platform", "win32")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "mips")
    with pytest.raises(RuntimeError, match="Unsupported architecture"):
        install_mod._kimix_base_platform_arch()
    monkeypatch.setattr(install_mod.sys, "platform", "os2")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match="Unsupported platform"):
        install_mod._kimix_base_platform_arch()


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def test_download_file_file_uri(tmp_path):
    src = tmp_path / "asset.zip"
    src.write_bytes(b"zip-bytes")
    dest = tmp_path / "out.zip"
    install_mod._download_file(src.as_uri(), dest)
    assert dest.read_bytes() == b"zip-bytes"


def test_download_file_retries_then_succeeds(tmp_path, monkeypatch):
    """Transient URLErrors are retried; a later attempt succeeds."""
    src = tmp_path / "asset.zip"
    src.write_bytes(b"zip-bytes")
    dest = tmp_path / "out.zip"
    calls = {"n": 0}
    real_once = install_mod._download_once

    def flaky(url, dest_path, timeout=300.0, extra_headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("winerror 10061 conn refused")
        return real_once(url, dest_path, timeout, extra_headers)

    monkeypatch.setattr(install_mod, "_download_once", flaky)
    install_mod._download_file(src.as_uri(), dest)
    assert calls["n"] == 2
    assert dest.read_bytes() == b"zip-bytes"


def test_download_file_exhausts_retries_and_fallback_fails(tmp_path, monkeypatch):
    """All primary attempts fail AND the fallback download fails -> raises."""
    dest = tmp_path / "out.zip"
    fallback_url = "https://release-assets.githubusercontent.com/direct.zip"

    def always_fail(url, dest_path, timeout=300.0, extra_headers=None):
        if url == fallback_url:
            raise urllib.error.URLError("fallback failed")
        raise urllib.error.URLError("winerror 10060 timed out")

    monkeypatch.setattr(install_mod, "_download_once", always_fail)
    monkeypatch.setattr(install_mod, "_kimix_base_fallback_url", lambda name: fallback_url)
    with pytest.raises(urllib.error.URLError, match="fallback failed"):
        install_mod._download_file(
            "https://github.com/Sikao-Engine/KimiX-native/releases/download/Release/"
            "kimix_base-linux-x64-1.0.0.zip",
            dest,
        )
    assert not dest.exists()


def test_download_file_fallback_succeeds(tmp_path, monkeypatch):
    """The API-resolved direct CDN fallback completes the download."""
    src = tmp_path / "asset.zip"
    src.write_bytes(b"zip-bytes")
    dest = tmp_path / "out.zip"
    fallback_url = "https://api.github.com/repos/Sikao-Engine/KimiX-native/releases/assets/534870525"
    primary_url = (
        "https://github.com/Sikao-Engine/KimiX-native/releases/download/Release/"
        "kimix_base-linux-x64-1.0.0.zip"
    )
    fallback_headers: dict | None = None

    def dispatch(url, dest_path, timeout=300.0, extra_headers=None):
        nonlocal fallback_headers
        if url == fallback_url:
            fallback_headers = extra_headers
            dest_path.write_bytes(b"zip-bytes")
            return
        if url == primary_url:
            raise urllib.error.URLError("winerror 10060 timed out")
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(install_mod, "_download_once", dispatch)
    monkeypatch.setattr(install_mod, "_kimix_base_fallback_url", lambda name: fallback_url)
    install_mod._download_file(primary_url, dest)
    assert dest.read_bytes() == b"zip-bytes"
    # The GitHub API asset endpoint must be opened as an octet stream so it
    # redirects straight to the CDN blob host instead of returning JSON.
    assert fallback_headers == {"Accept": "application/octet-stream"}


def test_download_file_raises_when_fallback_unavailable(tmp_path, monkeypatch):
    """If no fallback URL can be resolved, the last URLError is raised."""
    dest = tmp_path / "out.zip"

    def always_fail(url, dest_path, timeout=300.0, extra_headers=None):
        raise urllib.error.URLError("winerror 10060 timed out")

    monkeypatch.setattr(install_mod, "_download_once", always_fail)
    monkeypatch.setattr(install_mod, "_kimix_base_fallback_url", lambda name: None)
    with pytest.raises(urllib.error.URLError, match="winerror 10060"):
        install_mod._download_file("https://example.com/asset.zip", dest)
    assert not dest.exists()


def test_download_file_http_error_not_retried(tmp_path, monkeypatch):
    """An authoritative HTTP error (e.g. 404) propagates immediately."""
    dest = tmp_path / "out.zip"
    calls = {"n": 0}

    def http_error(url, dest_path, timeout=300.0, extra_headers=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(install_mod, "_download_once", http_error)
    with pytest.raises(urllib.error.HTTPError):
        install_mod._download_file("https://github.com/x/y.zip", dest)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# GitHub API fallback URL resolution
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    """Minimal file-like response for urlopen monkeypatching."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fallback_url_resolves_asset(monkeypatch, tmp_path):
    """_kimix_base_fallback_url returns the matching asset's API URL."""
    asset_url = "https://api.github.com/repos/Sikao-Engine/KimiX-native/releases/assets/534870525"
    release = {
        "tag_name": "Release",
        "assets": [
            {"name": "kimix_base-windows-x64-1.0.0.zip",
             "url": "https://api.github.com/.../assets/1",
             "browser_download_url": "https://github.com/.../windows.zip"},
            {"name": "kimix_base-linux-x64-1.0.0.zip",
             "url": asset_url,
             "browser_download_url": "https://github.com/.../linux.zip"},
        ],
    }

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(json.dumps(release).encode("utf-8"))

    monkeypatch.setattr(install_mod.urllib.request, "urlopen", fake_urlopen)
    assert (
        install_mod._kimix_base_fallback_url("kimix_base-linux-x64-1.0.0.zip") == asset_url
    )


def test_fallback_url_missing_asset_returns_none(monkeypatch):
    """A release without the requested asset yields None (no exception)."""
    release = {"tag_name": "Release", "assets": []}

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(json.dumps(release).encode("utf-8"))

    monkeypatch.setattr(install_mod.urllib.request, "urlopen", fake_urlopen)
    assert install_mod._kimix_base_fallback_url("kimix_base-linux-x64-1.0.0.zip") is None


def test_fallback_url_api_error_returns_none(monkeypatch):
    """A failed GitHub API lookup yields None (no exception)."""

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("api down")

    monkeypatch.setattr(install_mod.urllib.request, "urlopen", fake_urlopen)
    assert install_mod._kimix_base_fallback_url("kimix_base-linux-x64-1.0.0.zip") is None


# ---------------------------------------------------------------------------
# helpers: build a local kimix_base-style release archive
# ---------------------------------------------------------------------------


def _make_release_archive(dest: Path) -> Path:
    """Build a kimix_base-style zip archive with the platform's extension."""
    archive = dest / f"kimix_base-windows-x64-{install_mod.KIMIX_BASE_VERSION}.zip"
    payload = dest / "payload"
    payload.mkdir(parents=True, exist_ok=True)
    (payload / _NATIVE_FILE).write_bytes(b"# fake runtime_py extension\n")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(payload / _NATIVE_FILE, arcname=_NATIVE_FILE)
    return archive


# ---------------------------------------------------------------------------
# zip extraction
# ---------------------------------------------------------------------------


def test_extract_zip_extracts_members(tmp_path):
    archive = _make_release_archive(tmp_path)
    out = tmp_path / "out"
    install_mod._extract_zip(archive, out)
    assert (out / _NATIVE_FILE).read_bytes() == b"# fake runtime_py extension\n"


def test_extract_zip_rejects_bad_archive(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"this is not a zip archive")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="not a valid zip archive"):
        install_mod._extract_zip(archive, out)


# ---------------------------------------------------------------------------
# staging into bin/
# ---------------------------------------------------------------------------


def test_stage_native_files_copies_artifacts_and_deps(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / _NATIVE_FILE).write_bytes(b"pyd")
    (src / "vcruntime140.dll").write_bytes(b"dep")
    (src / "notes.txt").write_bytes(b"skip")
    dest = tmp_path / "dest"
    copied = install_mod._stage_native_files(src, dest)
    assert sorted(copied) == sorted([_NATIVE_FILE, "vcruntime140.dll"])
    assert sorted(p.name for p in dest.iterdir()) == sorted(copied)
    assert not (dest / "notes.txt").exists()


def test_stage_native_files_finds_nested_pyd_dir(tmp_path):
    src = tmp_path / "src" / "release"
    src.mkdir(parents=True)
    (src / _NATIVE_FILE).write_bytes(b"pyd")
    dest = tmp_path / "dest"
    copied = install_mod._stage_native_files(tmp_path / "src", dest)
    assert sorted(copied) == [_NATIVE_FILE]
    assert (dest / _NATIVE_FILE).read_bytes() == b"pyd"


def test_stage_native_files_missing_artifacts_raises(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "other.dll").write_bytes(b"dll")  # native extension missing
    with pytest.raises(RuntimeError, match="does not contain"):
        install_mod._stage_native_files(src, tmp_path / "dest")


def test_stage_native_files_locked_file_reports(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / _NATIVE_FILE).write_bytes(b"pyd")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / _NATIVE_FILE).write_bytes(b"old")
    real_copy2 = shutil.copy2

    def locked_copy2(src_file, dst_file, **kwargs):
        if Path(dst_file).name == _NATIVE_FILE:
            raise PermissionError(32, "in use")
        return real_copy2(src_file, dst_file, **kwargs)

    monkeypatch.setattr(install_mod.shutil, "copy2", locked_copy2)
    with pytest.raises(RuntimeError, match="in use by a running process"):
        install_mod._stage_native_files(src, dest)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def test_verify_native_binaries_missing_files(tmp_path):
    ok, msg = install_mod._verify_native_binaries(tmp_path)
    assert ok is False
    assert "missing" in msg


def test_verify_native_binaries_rejects_broken_pyd(tmp_path):
    shim = tmp_path / "kimix_native"
    shim.mkdir()
    (shim / "__init__.py").write_text(
        "import os\n"
        "_native = None\n"
        "if os.environ.get('KIMIX_NATIVE', 'auto') != '0':\n"
        "    try:\n"
        "        import runtime_py as _native\n"
        "    except ImportError:\n"
        "        if os.environ.get('KIMIX_NATIVE') == '1':\n"
        "            raise\n"
        "        _native = None\n"
        "def use_native(kernel):\n"
        "    return _native is not None\n"
        "def version():\n"
        "    return 'broken'\n"
    )
    (tmp_path / _NATIVE_FILE).write_bytes(b"this is not a real extension")
    ok, msg = install_mod._verify_native_binaries(tmp_path)
    assert ok is False
    assert msg  # a diagnostic is returned


def test_verify_native_binaries_repo_bin(tmp_path):
    """The real repo bin/ (staged runtime) must verify when present."""
    if not (REPO_BIN / _NATIVE_FILE).is_file():
        pytest.skip("native runtime not staged in repo bin/")
    ok, version = install_mod._verify_native_binaries(REPO_BIN)
    assert ok is True
    assert install_mod.KIMIX_BASE_VERSION in version


# ---------------------------------------------------------------------------
# end-to-end: download -> extract -> stage -> verify -> delete zip
# ---------------------------------------------------------------------------


def _install_env(tmp_path, monkeypatch, archive: Path):
    """Point the installer at a local file:// archive and a temp bin dir."""
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    stage_bin = tmp_path / "bin"
    shutil.copytree(REPO_BIN, stage_bin)  # carries the tracked kimix_native shim
    monkeypatch.setattr(install_mod.tempfile, "mkdtemp", lambda prefix="": str(download_dir))
    monkeypatch.setattr(install_mod, "_kimix_base_platform_arch", lambda: ("windows", "x64"))
    monkeypatch.setattr(
        install_mod, "_kimix_base_download_url", lambda os_name, arch: archive.as_uri()
    )
    return download_dir, stage_bin


def test_install_kimix_native_success_deletes_archive(tmp_path, monkeypatch):
    archive = _make_release_archive(tmp_path)
    download_dir, stage_bin = _install_env(tmp_path, monkeypatch, archive)
    # The fake payload is not a real extension; verification itself is covered
    # by the dedicated tests above, so simulate a successful check here.
    monkeypatch.setattr(
        install_mod,
        "_verify_native_binaries",
        lambda d: (True, f"kimix-runtime {install_mod.KIMIX_BASE_VERSION}"),
    )
    ok = install_mod._install_kimix_native(bin_dir=stage_bin, force=True)
    assert ok is True
    assert (stage_bin / _NATIVE_FILE).is_file()
    # The downloaded archive must be deleted on success.
    assert not (download_dir / archive.name).exists()


def test_install_kimix_native_failure_keeps_archive(tmp_path, monkeypatch):
    archive = _make_release_archive(tmp_path)
    download_dir, stage_bin = _install_env(tmp_path, monkeypatch, archive)
    monkeypatch.setattr(
        install_mod, "_verify_native_binaries", lambda d: (False, "boom")
    )
    ok = install_mod._install_kimix_native(bin_dir=stage_bin, force=True)
    assert ok is False
    # The downloaded archive must be KEPT on failure (delete only on success).
    assert (download_dir / archive.name).exists()


def test_install_kimix_native_already_installed_skips(tmp_path, monkeypatch):
    download_dir, stage_bin = _install_env(tmp_path, monkeypatch, tmp_path / "nope.zip")
    monkeypatch.setattr(
        install_mod,
        "_verify_native_binaries",
        lambda d: (True, f"kimix-runtime {install_mod.KIMIX_BASE_VERSION}"),
    )
    ok = install_mod._install_kimix_native(bin_dir=stage_bin)
    assert ok is True  # skip path: no download attempted
    assert list(download_dir.iterdir()) == []


def test_sync_kimix_native_version_writes_config(tmp_path, monkeypatch):
    """_sync_kimix_native_version writes the version to the repo-root config file."""
    version_path = tmp_path / "KIMIX_NATIVE_VERSION"
    monkeypatch.setattr(
        install_mod, "_kimix_native_version_path", lambda: version_path
    )
    assert install_mod._sync_kimix_native_version("0.3.0") is True
    assert version_path.read_text(encoding="utf-8").strip() == "0.3.0"
