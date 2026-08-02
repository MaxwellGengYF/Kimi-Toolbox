"""
Install Git cross-platform (Windows, Linux, macOS) into a directory that is
already on PATH.

Why a directory on PATH?
    Git used to be installed into ``~/.kimi/git`` and then manually added to
    the user PATH.  Now the target directory is always a directory that is
    already present in PATH (e.g. ``/usr/bin`` on Unix or ``C:\\Windows`` on
    Windows), so the binary is immediately visible and no hidden KIMI
    directory is created.  The caller is told exactly where git was installed.

Strategies (in priority order):
- Windows:
    1. PortableGit self-extracting archive extracted into a PATH directory
    2. Official installer from GitHub releases (adds dirs to user PATH)
    3. Chocolatey (if already available)
    4. Scoop (if already available)
- Linux / macOS:
    1. System package manager (brew, apt-get, apt, dnf, yum, pacman, apk,
       zypper, port, conda) -- installs git into a system binary directory
       that is already on PATH (e.g. ``/usr/bin``).

Usage:
    python install_git.py                          # default install
    python install_git.py --version 2.55.0         # pin version
    python install_git.py --dir "D:\\Tools"        # custom dir (must be on PATH)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ============================================================
# Global configuration -- change these to pin version
# ============================================================
GIT_VERSION: str = "2.55.0"
"""Git version to install when using the download-based strategies."""

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

_PORTABLE_DOWNLOAD_URL = (
    "https://github.com/git-for-windows/git/releases/download/"
    "v{version}.windows.{release}/PortableGit-{version}-64-bit.7z.exe"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return sys.platform == "win32"


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result (stdout/stderr captured as text)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _is_writable_dir(path: Path) -> bool:
    """Best-effort check that *path* is a directory we can write into."""
    try:
        if not path.is_dir():
            return False
        probe = path / f".kimi-git-write-test-{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _preferred_bin_dirs() -> list[Path]:
    """Standard system directories that are normally already on PATH."""
    if _is_windows():
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        return [Path(system_root), Path(r"C:\Windows")]
    return [Path("/usr/local/bin"), Path("/opt/homebrew/bin"), Path("/usr/bin"), Path("/bin")]


def _path_entries() -> list[Path]:
    """Directories currently listed in the PATH environment variable."""
    entries: list[Path] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if entry:
            entries.append(Path(entry))
    return entries


def _pick_path_dir() -> Path:
    """Pick a directory already on PATH to install git into.

    Preference order:
    1. Standard system directories (``/usr/bin``, ``C:\\Windows``, ...) that
       are on PATH and writable;
    2. Any other writable directory already on PATH;
    3. The first PATH entry as a last resort (installation may still fail).

    The old KIMI share dir (``~/.kimi/git``) is never considered.
    """
    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in _preferred_bin_dirs() + _path_entries():
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    for candidate in candidates:
        if _is_writable_dir(candidate):
            return candidate
    if candidates:
        return candidates[0]
    return Path.home()


# ---------------------------------------------------------------------------
# strategy implementations (Windows)
# ---------------------------------------------------------------------------

def _try_direct_download(
    version: str = GIT_VERSION,
    install_dir: str | Path | None = None,
) -> bool:
    """Download the official Git installer and run it silently (Windows)."""
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
    if install_dir is not None:
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
    target = Path(install_dir) if install_dir is not None else _pick_path_dir()
    _ensure_in_user_path(str(target / "bin"))
    _ensure_in_user_path(str(target / "cmd"))

    # --- clean up ---
    installer.unlink(missing_ok=True)

    return shutil.which("git") is not None


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


def _try_portable(
    version: str = GIT_VERSION,
    install_dir: str | Path | None = None,
) -> bool:
    """Download PortableGit and extract it into *install_dir* (Windows only).

    Unlike the installer, PortableGit always extracts to the given directory
    regardless of any existing Git installation.  *install_dir* defaults to a
    writable directory that is already on PATH (e.g. ``C:\\Windows``).
    """
    url = _PORTABLE_DOWNLOAD_URL.format(
        version=version, release=_GIT_WINDOWS_RELEASE
    )
    archive = Path(tempfile.gettempdir()) / f"PortableGit-{version}-64-bit.7z.exe"
    target = Path(install_dir) if install_dir is not None else _pick_path_dir()

    # --- download ---
    try:
        print(f"Downloading PortableGit {version} ...")
        _download_file(url, archive)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False

    # --- extract ---
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
        # Make the extracted bin/cmd dirs visible on PATH (current process +
        # persistent user PATH), then verify.
        _ensure_in_user_path(str(target / "bin"))
        _ensure_in_user_path(str(target / "cmd"))
    return ok


def _ensure_in_user_path(dirpath: str) -> None:
    """Add *dirpath* to the current user's PATH environment variable (persistent).

    Updates both the registry (Windows, for new processes) and the current
    process's ``os.environ`` so that ``shutil.which`` picks it up immediately.
    """
    # --- current process (immediate) ---
    current_path = os.environ.get("PATH", "")
    current_entries = [p.strip() for p in current_path.split(os.pathsep) if p.strip()]
    if dirpath not in current_entries:
        current_entries.append(dirpath)
        os.environ["PATH"] = os.pathsep.join(current_entries)

    # --- registry (persistent, Windows only) ---
    if not _is_windows():
        return

    import winreg

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
# strategy implementations (Linux / macOS)
# ---------------------------------------------------------------------------

def _package_manager_commands() -> list[tuple[str, list[str], bool]]:
    """Return ``(name, command, needs_sudo)`` triples for installing git.

    System package managers install git into a binary directory that is
    already on PATH (``/usr/bin``, ``/usr/local/bin``, ``/opt/homebrew/bin``).
    """
    if sys.platform == "darwin":
        return [
            ("homebrew", ["brew", "install", "git"], False),
            ("macports", ["port", "install", "git"], True),
            ("conda", ["conda", "install", "-y", "git"], False),
        ]
    if sys.platform.startswith("linux"):
        return [
            ("apt-get", ["apt-get", "install", "-y", "git"], True),
            ("apt", ["apt", "install", "-y", "git"], True),
            ("dnf", ["dnf", "install", "-y", "git"], True),
            ("yum", ["yum", "install", "-y", "git"], True),
            ("pacman", ["pacman", "-S", "--noconfirm", "git"], True),
            ("apk", ["apk", "add", "--no-cache", "git"], True),
            ("zypper", ["zypper", "--non-interactive", "install", "git"], True),
            ("conda", ["conda", "install", "-y", "git"], False),
        ]
    return []


def _need_sudo() -> bool:
    """Return ``True`` when the current user is not root (Unix only)."""
    if _is_windows():
        return False
    try:
        return os.geteuid() != 0
    except (AttributeError, OSError):
        return True


def _try_package_manager() -> bool:
    """Install git via the system package manager (Linux/macOS)."""
    if _is_windows():
        return False
    for name, cmd, needs_sudo in _package_manager_commands():
        if not shutil.which(cmd[0]):
            continue
        if needs_sudo and _need_sudo():
            if not shutil.which("sudo"):
                continue
            full = ["sudo", "-n", *cmd]
        else:
            full = cmd
        print(f"Installing git via {name} ...")
        try:
            result = _run(full, timeout=900)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            continue
        if result.returncode == 0 and shutil.which("git"):
            return True
    return False


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


def _git_found(install_dir: str | Path | None = None) -> bool:
    """Return ``True`` if ``git`` is available in *install_dir* or on PATH.

    Presence check only -- the version is deliberately never checked.
    """
    if install_dir:
        base = Path(install_dir)
        git_name = "git.exe" if _is_windows() else "git"
        if (base / "bin" / git_name).exists():
            return True
        if (base / "cmd" / git_name).exists():
            return True
    return shutil.which("git") is not None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def install_git(
    version: str = GIT_VERSION,
    install_dir: str | Path | None = None,
) -> str | None:
    """Install Git into a directory that is already on PATH.

    Parameters
    ----------
    version:
        Git version string (only used for the download-based strategies).
    install_dir:
        Target directory.  It must already be on PATH.  ``None`` (default)
        auto-picks a writable directory that is already on PATH
        (e.g. ``/usr/bin`` on Unix or ``C:\\Windows`` on Windows).  The old
        KIMI share dir (``~/.kimi/git``) is never used.

    Returns
    -------
    The directory where Git is installed (always a directory on PATH), or
    ``None`` if installation failed.
    """
    # Already on PATH?  Nothing to do -- presence check only, no version check.
    if shutil.which("git"):
        git_path = Path(shutil.which("git"))
        print(f"Git is already installed at {git_path.parent}, skipping.")
        return str(git_path.parent)

    if install_dir is not None and _git_found(install_dir):
        print(f"Git is already installed at {install_dir}.")
        return str(Path(install_dir))

    target = Path(install_dir) if install_dir is not None else _pick_path_dir()
    if install_dir is None:
        print(f"Git will be installed into {target} (already on PATH).")
    else:
        print(f"Git will be installed into {target}.")

    if _is_windows():
        strategies: list[tuple[str, object]] = [
            ("portable", lambda: _try_portable(version, target)),
            ("direct download", lambda: _try_direct_download(version, target)),
            ("chocolatey", _try_choco),
            ("scoop", _try_scoop),
        ]
    else:
        strategies = [
            ("package manager", _try_package_manager),
        ]

    for name, fn in strategies:
        print(f"Trying {name} ...")
        try:
            ok = fn()  # type: ignore[operator]
        except Exception as exc:
            print(f"  {name} raised: {exc}")
            ok = False
        if ok and (shutil.which("git") or _git_found(target)):
            git_path = shutil.which("git")
            location = Path(git_path).parent if git_path else target
            print(f"Git installed successfully via {name}.")
            return str(location)
        print(f"  {name} did not succeed.")

    print("All installation strategies failed.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Install Git cross-platform into a directory already on PATH.",
    )
    parser.add_argument(
        "--version",
        default=GIT_VERSION,
        help=f"Git version (default: {GIT_VERSION})",
    )
    parser.add_argument(
        "--dir",
        dest="install_dir",
        default=None,
        help=(
            "Custom install directory.  Must already be on PATH "
            "(default: auto-pick a writable directory on PATH, "
            "e.g. /usr/bin or C:\\Windows)"
        ),
    )
    args = parser.parse_args()

    result = install_git(version=args.version, install_dir=args.install_dir)
    sys.exit(0 if result else 1)
