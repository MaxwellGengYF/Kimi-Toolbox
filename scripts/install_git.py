"""
Install Git for Windows silently.

Strategy:
Uses PortableGit self-extracting archive -- no installer, fully portable.

Fallback strategies (in priority order):
1. Official installer from GitHub releases
2. Chocolatey (if already available)
3. Scoop (if already available)

Usage:
    python install_git.py                          # default install
    python install_git.py --version 2.47.0         # pin version
    python install_git.py --dir "D:\\Git"          # custom dir
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ============================================================
# Global configuration -- change these to pin version / path
# ============================================================
GIT_VERSION: str = "2.55.0"
"""Git for Windows version to install when using the direct-download strategy."""

INSTALL_DIR = Path.home() / ".kimi" / "git"
"""Custom install directory.
   ``None`` means use the installer default (usually ``C:\\Program Files\\Git``).
   Example: ``r"D:\\Program Files\\Git"``
"""

# URL pattern for GitHub releases.  The tag name uses a ``.windows.N``
# suffix where N is an incrementing release counter.
# Reference: https://github.com/git-for-windows/git/releases
_DOWNLOAD_URL = (
    "https://github.com/git-for-windows/git/releases/download/"
    "v{version}.windows.{release}/Git-{version}-64-bit.exe"
)

# The `.windows.N` release counter for the current GIT_VERSION.
# Check https://github.com/git-for-windows/git/releases for the correct value.
_GIT_WINDOWS_RELEASE: int = 1

# Inno Setup silent-install flags used by the official installer.
# /VERYSILENT  - no window at all
# /NORESTART   - don't reboot after install
# /NOCANCEL    - user can't cancel
# /SP-         - skip "about to install" page
# /CLOSEAPPLICATIONS - close apps that might lock files
# /RESTARTAPPLICATIONS - restart those apps afterwards
_INNO_FLAGS = [
    "/VERYSILENT",
    "/NORESTART",
    "/NOCANCEL",
    "/SP-",
    "/CLOSEAPPLICATIONS",
    "/RESTARTAPPLICATIONS",
]

# Components to include (matching a typical dev setup).
# icons             - Start-menu icons
# ext\reg\shellhere - "Git Bash Here" right-click menu
# assoc             - associate .git* files
# assoc_sh          - associate .sh files
_INNO_COMPONENTS = r"icons,ext\reg\shellhere,assoc,assoc_sh"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return sys.platform == "win32"


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result (stdout/stderr captured as text)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# strategy implementations
# ---------------------------------------------------------------------------

def _try_direct_download(
    version: str = GIT_VERSION,
    install_dir: str | None = INSTALL_DIR,
) -> bool:
    """Download the official Git installer and run it silently."""
    url = _DOWNLOAD_URL.format(version=version, release=_GIT_WINDOWS_RELEASE)
    installer = Path(tempfile.gettempdir()) / f"Git-{version}-64-bit.exe"

    # --- download ---
    try:
        print(f"Downloading Git {version} ...")
        _download_file(url, installer)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False

    # --- install ---
    args = [str(installer), *_INNO_FLAGS, f"/COMPONENTS={_INNO_COMPONENTS}"]
    if install_dir:
        args.append(f'/DIR="{install_dir}"')

    try:
        print("Running silent installer ...")
        _run(args, timeout=900)
        # The installer may return non-zero for "reboot needed" warnings;
        # treat any result as success and verify afterwards.
    except subprocess.TimeoutExpired:
        print("Installer timed out.")
    except Exception as exc:
        print(f"Installer error: {exc}")

    # --- add to PATH ---
    target = Path(install_dir) if install_dir else Path(r"C:\Program Files\Git")
    _ensure_in_user_path(str(target / "bin"))
    _ensure_in_user_path(str(target / "cmd"))

    # --- clean up ---
    installer.unlink(missing_ok=True)

    return _git_found()


def _try_choco() -> bool:
    """Install Git via Chocolatey (if already on the machine)."""
    if not shutil.which("choco"):
        return False
    try:
        result = _run(["choco", "install", "git", "-y"])
        return result.returncode == 0
    except Exception:
        return False


def _try_scoop() -> bool:
    """Install Git via Scoop (if already on the machine)."""
    if not shutil.which("scoop"):
        return False
    try:
        result = _run(["scoop", "install", "git"])
        return result.returncode == 0
    except Exception:
        return False


_PORTABLE_DOWNLOAD_URL = (
    "https://github.com/git-for-windows/git/releases/download/"
    "v{version}.windows.{release}/PortableGit-{version}-64-bit.7z.exe"
)


def _try_portable(
    version: str = GIT_VERSION,
    install_dir: str | None = INSTALL_DIR,
) -> bool:
    """Download PortableGit self-extracting archive and extract it to *install_dir*.

    Unlike the installer, PortableGit always extracts to the given directory
    regardless of any existing Git installation.  This is the recommended
    strategy for installing into the KIMI share directory.
    """
    url = _PORTABLE_DOWNLOAD_URL.format(
        version=version, release=_GIT_WINDOWS_RELEASE
    )
    archive = Path(tempfile.gettempdir()) / f"PortableGit-{version}-64-bit.7z.exe"

    # --- download ---
    try:
        print(f"Downloading PortableGit {version} ...")
        _download_file(url, archive)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False

    # --- extract ---
    target = Path(install_dir) if install_dir else Path.home() / ".kimi" / "git"
    target.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Extracting to {target} ...")
        _run([str(archive), "-o" + str(target), "-y"], timeout=300)
    except subprocess.TimeoutExpired:
        print("Extraction timed out.")
    except Exception as exc:
        print(f"Extraction error: {exc}")
    finally:
        archive.unlink(missing_ok=True)

    # --- verify ---
    bash_exe = target / "bin" / "bash.exe"
    git_exe = target / "bin" / "git.exe"
    ok = bash_exe.exists() and git_exe.exists()
    if ok:
        # Add to user PATH if not already there
        _ensure_in_user_path(str(target / "bin"))
        _ensure_in_user_path(str(target / "cmd"))
    return ok


def _ensure_in_user_path(dirpath: str) -> None:
    """Add *dirpath* to the current user's PATH environment variable (persistent).

    Updates both the registry (for new processes) and the current process's
    ``os.environ`` so that ``shutil.which`` picks it up immediately.
    """
    import winreg

    # --- current process (immediate) ---
    current_path = os.environ.get("PATH", "")
    current_entries = [p.strip() for p in current_path.split(";") if p.strip()]
    if dirpath not in current_entries:
        current_entries.append(dirpath)
        os.environ["PATH"] = ";".join(current_entries)

    # --- registry (persistent) ---
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
    except FileNotFoundError:
        return

    try:
        path_val, _ = winreg.QueryValueEx(key, "Path")
    except FileNotFoundError:
        path_val = ""

    entries = [p.strip() for p in path_val.split(";") if p.strip()]
    if dirpath in entries:
        winreg.CloseKey(key)
        return

    entries.append(dirpath)
    new_path = ";".join(entries)
    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
    winreg.CloseKey(key)


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, with a progress indicator."""
    import urllib.request

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, int(block_num * block_size * 100 / total_size))
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), _report)
    print()  # newline after progress


def _git_found(install_dir: str | None = None) -> bool:
    """Return ``True`` if ``git.exe`` is available.

    When *install_dir* is given, checks that directory first
    (both ``bin/git.exe`` and ``cmd/git.exe``).  Always falls back to
    checking PATH via ``shutil.which``.
    """
    if install_dir:
        base = Path(install_dir)
        if (base / "bin" / "git.exe").exists():
            return True
        if (base / "cmd" / "git.exe").exists():
            return True
    # Fall back to checking actual PATH
    return shutil.which("git") is not None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def install_git(
    version: str = GIT_VERSION,
    install_dir: str | None = INSTALL_DIR,
) -> bool:
    """Install Git for Windows using the best available strategy.

    Parameters
    ----------
    version:
        Git version string (only used for the direct-download strategy).
    install_dir:
        Custom install directory.  ``None`` to accept the default.

    Returns
    -------
    ``True`` if Git is available on PATH after execution, ``False`` otherwise.
    """
    if not _is_windows():
        print("install_git: this script only supports Windows.", file=sys.stderr)
        return False

    # Already installed in the target directory?  Nothing to do.
    if _git_found(install_dir):
        where = f"at {install_dir}" if install_dir else "on PATH"
        print(f"Git is already installed {where}.")
        return True

    # Attempt strategies in priority order.
    strategies: list[tuple[str, object]] = [
        ("portable", lambda: _try_portable(version, install_dir)),
        ("direct download", lambda: _try_direct_download(version, install_dir)),
        ("chocolatey", _try_choco),
        ("scoop", _try_scoop),
    ]

    for name, fn in strategies:
        print(f"Trying {name} ...")
        try:
            ok = fn()  # type: ignore[operator]
        except Exception as exc:
            print(f"  {name} raised: {exc}")
            ok = False
        if ok and _git_found(install_dir):
            print(f"Git installed successfully via {name}.")
            return True
        print(f"  {name} did not succeed.")

    print("All installation strategies failed.", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Install Git for Windows silently.",
    )
    parser.add_argument(
        "--version",
        default=GIT_VERSION,
        help=f"Git version (default: {GIT_VERSION})",
    )
    parser.add_argument(
        "--dir",
        dest="install_dir",
        default=INSTALL_DIR,
        help="Custom install directory (default: installer default)",
    )
    args = parser.parse_args()

    success = install_git(version=args.version, install_dir=args.install_dir)
    sys.exit(0 if success else 1)
