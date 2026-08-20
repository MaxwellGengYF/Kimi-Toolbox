#!/usr/bin/env python3
"""Install script for the project using uv."""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def _get_rg_version() -> str:
    """Return the expected ripgrep version from kimi_cli._ripgrep_common."""
    try:
        from kimi_cli._ripgrep_common import RG_VERSION
        return RG_VERSION
    except ImportError:
        pass
    # If a partial kimi_cli is already cached in sys.modules (e.g. from a global
    # Python environment that has an unrelated kimi_cli), Python will not re-import
    # it even after adding the source path.  Clean up before retrying.
    for key in list(sys.modules):
        if key == "kimi_cli" or key.startswith("kimi_cli."):
            del sys.modules[key]
    src_path = Path(__file__).resolve().parent / "kimi-cli" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from kimi_cli._ripgrep_common import RG_VERSION
    return RG_VERSION


def _get_rtk_version() -> str:
    """Return the expected rtk version from kimi_cli._rtk_common."""
    try:
        from kimi_cli._rtk_common import RTK_VERSION
        return RTK_VERSION
    except ImportError:
        pass
    # Clean up any partial kimi_cli in sys.modules (same rationale as above).
    for key in list(sys.modules):
        if key == "kimi_cli" or key.startswith("kimi_cli."):
            del sys.modules[key]
    src_path = Path(__file__).resolve().parent / "kimi-cli" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from kimi_cli._rtk_common import RTK_VERSION
    return RTK_VERSION


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# Set to True when the script is invoked with ``-y``/``--yes``: every yes/no
# prompt is then answered affirmatively without user input.
_ASSUME_YES = False


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask the user a yes/no question.

    When the script was invoked with ``-y``/``--yes`` (or in non-interactive
    environments such as CI pipelines) the affirmative answer is returned
    immediately so the script does not hang.
    """
    if _ASSUME_YES:
        return True
    if not sys.stdin.isatty():
        return default

    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default

    if not answer:
        return default
    return answer in ("y", "yes")


def run_command(cmd: list[str], description: str) -> bool:
    print(f"\n▶ {description} ...")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n❌ Command failed: {' '.join(cmd)}")
            return False
        print(f"✅ {description} completed.")
        return True
    except Exception as e:
        print(f"\n❌ Error running command: {' '.join(cmd)}")
        print(f"   Details: {e}")
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _install_coreutils() -> tuple[bool, bool]:
    """Check existence and install coreutils (Windows only).

    Returns (was_installed, should_restart_shell).
    """
    if sys.platform != "win32":
        return False, False

    if command_exists("cat.exe"):
        print("✅ Coreutils is already installed, skipping.")
        return False, False

    if not _ask_yes_no("Microsoft Coreutils was not found. Install Coreutils?"):
        print("⏭️  Skipping Coreutils installation.")
        return False, False

    coreutils_script = Path(__file__).parent / "scripts" / "install_coreutils.py"
    if not coreutils_script.exists():
        print(f"⚠️  install_coreutils.py not found at {coreutils_script}, skipping.")
        return False, False

    try:
        scripts_dir = str(coreutils_script.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import install_coreutils  # type: ignore[import-untyped]

        print("\n▶ Installing coreutils ...")
        result = install_coreutils.install_coreutils()
        if result:
            print(f"✅ Coreutils installed at {result}.")
            return True, True
        else:
            print("⚠️  Coreutils installation was not successful (non-fatal).")
            return False, False
    except Exception as e:
        print(f"⚠️  Could not install coreutils: {e}")
        return False, False

def _install_git() -> tuple[bool, bool]:
    """Prompt for and install Git if needed (cross-platform).

    If ``git`` is already on PATH it is skipped without any version check.
    Otherwise ``scripts/install_git.py`` installs it into a directory that
    is already on PATH (e.g. ``/usr/bin`` or ``C:\\Windows``) and reports
    where it was installed.

    Returns (was_installed, should_restart_shell).
    """
    if command_exists("git"):
        print("✅ Git is already installed, skipping.")
        return False, False

    if not _ask_yes_no("Git was not found. Install Git?"):
        print("⏭️  Skipping Git installation.")
        return False, False

    git_script = Path(__file__).parent / "scripts" / "install_git.py"
    if not git_script.exists():
        print(f"⚠️  install_git.py not found at {git_script}, skipping.")
        return False, False

    try:
        scripts_dir = str(git_script.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import install_git

        print("\n▶ Installing Git ...")
        result = install_git.install_git()
        if result:
            print(f"✅ Git installed at {result}.")
            return True, True
        else:
            print("⚠️  Git installation was not successful (non-fatal).")
            return False, False
    except Exception as e:
        print(f"⚠️  Could not install Git: {e}")
        return False, False


def _shared_bin_path(bin_name: str) -> Path:
    """Return the expected shared bin path for *bin_name*."""
    if share_dir := os.getenv("KIMI_SHARE_DIR"):
        return Path(share_dir) / "bin" / bin_name
    return Path.home() / ".kimi" / "bin" / bin_name


def _get_installed_binary_version(binary: Path, timeout: float = 5.0) -> str | None:
    """Run ``binary --version`` and extract a semver version string."""
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = result.stdout or result.stderr
        if not output:
            return None
        # Match first occurrence of X.Y.Z
        m = re.search(r"(\d+\.\d+\.\d+)", output)
        return m.group(1) if m else None
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _install_ripgrep() -> tuple[bool, bool]:
    """Prompt for and install ripgrep if needed (cross-platform).

    Returns (was_installed, should_restart_shell).
    """
    bin_name = "rg.exe" if sys.platform == "win32" else "rg"
    share_bin = _shared_bin_path(bin_name)

    if share_bin.is_file():
        expected_version = _get_rg_version()
        installed_version = _get_installed_binary_version(share_bin)
        if installed_version == expected_version:
            print(f"✅ Ripgrep {expected_version} is already installed in shared bin, skipping.")
            return False, False
        else:
            installed_str = installed_version or "unknown"
            print(f"⚠️  Ripgrep found but version is {installed_str}, expected {expected_version}.")
            if not _ask_yes_no("Reinstall Ripgrep to the expected version?"):
                print("⏭️  Skipping Ripgrep reinstall.")
                return False, False
            # Remove the stale binary so the install function actually replaces it.
        try:
            share_bin.unlink()
        except OSError:
            pass

        # Fall through to the download/install block below

    if not _ask_yes_no("Ripgrep was not found. Install Ripgrep?"):
        print("⏭️  Skipping Ripgrep installation.")
        return False, False

    rg_script = Path(__file__).parent / "scripts" / "install_ripgrep.py"
    if not rg_script.exists():
        print(f"⚠️  install_ripgrep.py not found at {rg_script}, skipping.")
        return False, False

    try:
        scripts_dir = str(rg_script.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import install_ripgrep

        print("\n▶ Installing Ripgrep ...")
        result = install_ripgrep.install_ripgrep()
        if result:
            expected_version = _get_rg_version()
            print(f"✅ Ripgrep installed at {result} (version {expected_version}).")
            return True, True
        else:
            print("⚠️  Ripgrep installation was not successful (non-fatal).")
            return False, False
    except Exception as e:
        print(f"⚠️  Could not install Ripgrep: {e}")
        return False, False


def _install_rtk() -> tuple[bool, bool]:
    """Prompt for and install rtk if needed (cross-platform).

    Returns (was_installed, should_restart_shell).
    """
    bin_name = "rtk.exe" if sys.platform == "win32" else "rtk"
    share_bin = _shared_bin_path(bin_name)

    if share_bin.is_file():
        expected_version = _get_rtk_version()
        installed_version = _get_installed_binary_version(share_bin)
        if installed_version == expected_version:
            print(f"✅ rtk {expected_version} is already installed in shared bin, skipping.")
            return False, False
        else:
            installed_str = installed_version or "unknown"
            print(f"⚠️  rtk found but version is {installed_str}, expected {expected_version}.")
            if not _ask_yes_no("Reinstall rtk to the expected version?"):
                print("⏭️  Skipping rtk reinstall.")
                return False, False
            # Remove the stale binary so the install function actually replaces it.
        try:
            share_bin.unlink()
        except OSError:
            pass

        # Fall through to the download/install block below

    if not _ask_yes_no("rtk (reasoning toolkit) was not found. Install rtk?"):
        print("⏭️  Skipping rtk installation.")
        return False, False

    rtk_script = Path(__file__).parent / "scripts" / "install_rtk.py"
    if not rtk_script.exists():
        print(f"⚠️  install_rtk.py not found at {rtk_script}, skipping.")
        return False, False

    try:
        scripts_dir = str(rtk_script.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import install_rtk

        print("\n▶ Installing rtk ...")
        result = install_rtk.install_rtk()
        if result:
            expected_version = _get_rtk_version()
            print(f"✅ rtk installed at {result} (version {expected_version}).")
            return True, True
        else:
            print("⚠️  rtk installation was not successful (non-fatal).")
            return False, False
    except Exception as e:
        print(f"⚠️  Could not install rtk: {e}")
        return False, False


# ---------------------------------------------------------------------------
# kimix_base native runtime (download + unpack into bin/)
# ---------------------------------------------------------------------------

# Release metadata for the kimix_base native runtime (runtime_py.pyd on
# Windows / runtime_py.so on Linux & macOS).
# The release asset naming rule is:
#   kimix_base-<platform>-<arch>-<version>.zip
# e.g. kimix_base-windows-x64-0.1.0.zip under
# https://github.com/Sikao-Engine/KimiX-native/releases/download/Release/...

KIMIX_BASE_VERSION = (Path(__file__).parent / 'KIMIX_NATIVE_VERSION').read_text(encoding='utf-8', errors='replace').strip()
KIMIX_BASE_RELEASE_URL = (
    "https://github.com/Sikao-Engine/KimiX-native/releases/download/Release"
)


def _native_files() -> tuple[str, ...]:
    """Compiled-extension file name(s) shipped by kimix_base for this platform.

    Mirrors kimix-base publish.py: Windows ships the CPython extension under
    its native ``.pyd`` name; Linux/macOS must use the ``.so`` suffix (CPython
    on those platforms only imports ``*.so`` modules).
    """
    if sys.platform == "win32":
        return ("runtime_py.pyd",)
    return ("runtime_py.so",)


KIMIX_BASE_NATIVE_FILES = _native_files()


def _kimix_native_version_path() -> Path:
    """Return the path to the repo-root ``KIMIX_NATIVE_VERSION`` config file.

    The shim and loader fallback version strings read this file so the
    pure-Python fallback reports the same version as the compiled runtime.
    """
    return Path(__file__).resolve().parent / "KIMIX_NATIVE_VERSION"


def _sync_kimix_native_version(version: str) -> bool:
    """Write *version* to the repo-root version config file.

    Keeping this file in sync with ``KIMIX_BASE_VERSION`` lets runtime code
    report a consistent version when the native extension is unavailable.
    The file is only touched when the version actually changes.
    """
    path = _kimix_native_version_path()
    try:
        current = path.read_text(encoding="utf-8").strip() if path.exists() else None
        if current == version:
            return True
        path.write_text(f"{version}\n", encoding="utf-8")
        print(f"✅ Synced kimix-native version file: {path.name} -> {version}")
        return True
    except OSError as exc:
        print(f"⚠️  Could not write {path}: {exc}")
        return False


def _kimix_base_platform_arch() -> tuple[str, str]:
    """Detect ``(platform, arch)`` for the kimix_base archive name rule.

    Supported platforms: windows / macos / linux.
    Supported architectures: x64 / x86 / arm64.
    """
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine in ("x86", "i386", "i686"):
        arch = "x86"
    else:
        raise RuntimeError(f"Unsupported architecture for kimix_base: {machine}")

    if sys.platform == "win32":
        os_name = "windows"
    elif sys.platform == "darwin":
        os_name = "macos"
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        raise RuntimeError(f"Unsupported platform for kimix_base: {sys.platform}")
    return os_name, arch


def _kimix_base_archive_name(
    os_name: str, arch: str, version: str = KIMIX_BASE_VERSION
) -> str:
    """Return the release archive file name (``kimix_base-<platform>-<arch>-<version>.zip``)."""
    return f"kimix_base-{os_name}-{arch}-{version}.zip"


def _kimix_base_download_url(
    os_name: str, arch: str, version: str = KIMIX_BASE_VERSION
) -> str:
    """Return the full download URL for the kimix_base release archive."""
    return f"{KIMIX_BASE_RELEASE_URL}/{_kimix_base_archive_name(os_name, arch, version)}"


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress indicator.

    Raises urllib.error.HTTPError / urllib.error.URLError on failure.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "kimi-agent-install/1.0"}
    )
    with urllib.request.urlopen(request, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, int(downloaded * 100 / total))
                    sys.stdout.write(f"\r  {pct}%")
                    sys.stdout.flush()
        print()  # newline after the progress line


def _extract_zip(archive: Path, dest: Path) -> None:
    """Extract a ``.zip`` archive into *dest* (created if missing).

    Uses the standard-library ``zipfile`` module — the same dependency-free
    approach as the ripgrep/rtk installers
    (``kimi_cli._ripgrep_common._extract_rg_archive``) — so no third-party
    package is required.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(path=dest)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Failed to extract {archive.name}: not a valid zip archive"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to extract {archive.name}") from exc


def _unlink_with_retry(path: Path, attempts: int = 5, delay: float = 1.0) -> None:
    """Delete *path*, retrying on transient Windows file locks (e.g. AV scans)."""
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def _stage_native_files(src_dir: Path, dest_dir: Path) -> list[str]:
    """Copy the native artifacts found under *src_dir* into *dest_dir*.

    The release archive may place the compiled extension (``runtime_py.pyd``
    on Windows / ``runtime_py.so`` on Linux & macOS) at the archive root or
    inside a sub-directory, so the directory holding it is located by walking
    *src_dir*. The extension and any sibling ``*.dll`` runtime dependencies are
    then copied into *dest_dir* (the same layout contract as
    tools/sync_native.py: the extension at the root of ``bin/``). Returns the
    list of copied file names.
    """
    native_files = KIMIX_BASE_NATIVE_FILES
    pyd_dirs = sorted(
        d for d in src_dir.rglob("*") if d.is_dir() and (d / native_files[0]).is_file()
    )
    source = pyd_dirs[0] if pyd_dirs else src_dir
    if not (source / native_files[0]).is_file():
        raise RuntimeError(
            f"archive does not contain {' or '.join(native_files)} (checked {source})"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    locked: list[str] = []
    for name in sorted(os.listdir(source)):
        path = source / name
        if not path.is_file():
            continue
        if name in KIMIX_BASE_NATIVE_FILES or name.lower().endswith(".dll"):
            try:
                shutil.copy2(path, dest_dir / name)
                copied.append(name)
            except OSError:
                locked.append(name)
    if locked:
        raise RuntimeError(
            "cannot overwrite "
            + ", ".join(locked)
            + f" in {dest_dir}: the file(s) are in use by a running process "
            "(e.g. a kimix/agent session that loaded the native runtime from bin/). "
            "Close that process and re-run."
        )
    return copied


def _verify_native_binaries(bin_dir: Path) -> tuple[bool, str]:
    """Verify the staged native runtime in *bin_dir*.

    Checks that the compiled extension (``runtime_py.pyd`` on Windows /
    ``runtime_py.so`` on Linux & macOS) exists and that the ``kimix_native``
    shim can import it. ``KIMIX_NATIVE=1`` is forced so a broken/mismatched
    extension is an error instead of a silent pure-Python fallback.
    Returns ``(ok, version_or_error)``.
    """
    for name in KIMIX_BASE_NATIVE_FILES:
        if not (bin_dir / name).is_file():
            return False, f"missing {name}"
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(bin_dir)!r});"
        "import kimix_native;"
        "assert kimix_native._native is not None, 'native extension failed to load';"
        "print(kimix_native.version())"
    )
    env = dict(os.environ)
    env["KIMIX_NATIVE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return False, f"verification command failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return False, (detail[-1] if detail else f"exit code {result.returncode}")
    return True, result.stdout.strip()


def _install_kimix_native(bin_dir: Path | None = None, force: bool = False) -> bool:
    """Download the kimix_base native runtime and unpack it into ``bin/``.

    The archive name follows the rule
    ``kimix_base-<platform>-<arch>-<version>.zip`` (e.g.
    ``kimix_base-windows-x64-0.1.0.zip``). The ``.zip`` is downloaded to a
    temporary location, extracted, staged into *bin_dir* (default
    ``<repo>\bin`` — the same layout as the compiled extension
    ``runtime_py.pyd`` / ``runtime_py.so``), verified by importing the
    extension, and the downloaded archive is deleted on success.

    When the staged runtime already loads and its version matches
    ``KIMIX_BASE_VERSION`` the download is skipped; a version mismatch prompts
    a reinstall. Pass ``force=True`` to always re-download and re-install.

    Returns True when the runtime is installed (or was already installed and
    verified); False when skipped or failed (non-fatal, matching the other
    optional binary installers in this script).
    """
    if bin_dir is None:
        bin_dir = Path(__file__).resolve().parent / "bin"

    already, installed_version = _verify_native_binaries(bin_dir)
    if already and not force:
        if KIMIX_BASE_VERSION in installed_version:
            print("✅ kimix_base native runtime is already installed and verified, skipping.")
            return True
        print(
            f"⚠️  kimix_base native runtime found but version is {installed_version!r}, "
            f"expected {KIMIX_BASE_VERSION}."
        )
        if not _ask_yes_no("Reinstall kimix_base native runtime to the expected version?"):
            print("⏭️  Skipping kimix_base reinstall.")
            return False

    if not _ask_yes_no(
        f"kimix_base native runtime ({' or '.join(_native_files())}) was not "
        "found/verified. Download and install it?"
    ):
        print("⏭️  Skipping kimix_base native runtime installation.")
        return False

    try:
        os_name, arch = _kimix_base_platform_arch()
    except RuntimeError as exc:
        print(f"⚠️  {exc}")
        return False

    archive_name = _kimix_base_archive_name(os_name, arch)
    url = _kimix_base_download_url(os_name, arch)
    print(f"\n▶ Downloading kimix_base native runtime {KIMIX_BASE_VERSION} ({os_name}-{arch}) ...")
    print(f"   {url}")

    tmpdir: Path | None = None
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="kimi-kimix-base-"))
        archive_path = tmpdir / archive_name
        extract_dir = tmpdir / "extract"

        _download_file(url, archive_path)
        print(f"\n▶ Extracting {archive_name} ...")
        _extract_zip(archive_path, extract_dir)
        copied = _stage_native_files(extract_dir, bin_dir)
        print(f"   staged into {bin_dir}: {', '.join(copied)}")

        ok, message = _verify_native_binaries(bin_dir)
        if not ok:
            raise RuntimeError(f"verification failed: {message}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"❌ Failed to download kimix_base native runtime: {exc}")
        if tmpdir is not None and archive_path.exists():
            print(f"   Downloaded archive kept at {archive_path} for debugging.")
        return False
    except (RuntimeError, OSError) as exc:
        print(f"❌ Failed to install kimix_base native runtime: {exc}")
        if tmpdir is not None and archive_path.exists():
            print(f"   Downloaded archive kept at {archive_path} for debugging.")
        return False

    # Success: delete the downloaded archive (only on success), then clean up.
    _unlink_with_retry(archive_path)
    print(f"🗑️  Removed downloaded archive {archive_path.name}.")
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"✅ kimix_base native runtime {KIMIX_BASE_VERSION} installed and verified: {message}")
    return True


def main(argv: list[str] | None = None) -> int:
    global _ASSUME_YES

    parser = argparse.ArgumentParser(
        prog="install.py",
        description="Install script for the project using uv.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to all prompts.",
    )
    args = parser.parse_args(argv)
    if args.yes:
        _ASSUME_YES = True
        print("🟢 '-y' detected: automatically accepting all prompts.")

    # 1. Check if python or uv exists
    has_python = command_exists("python") or command_exists("python3")
    has_uv = command_exists("uv")

    if not has_python and not has_uv:
        print(
            "❌ Neither 'python' nor 'uv' was found in your environment.\n"
            "   Please install Python (https://python.org) or uv (https://docs.astral.sh/uv) manually,\n"
            "   then re-run this script."
        )
        return 1

    if not has_uv:
        print(
            "⚠️  'uv' is not installed. Attempting to proceed anyway...\n"
            "   For best results, consider installing uv: https://docs.astral.sh/uv"
        )

    # 2. Optional binary installations (before uv sync so they are available)
    coreutils_installed, cu_restart = _install_coreutils()
    rg_installed, rg_restart = _install_ripgrep()
    rtk_installed, rtk_restart = _install_rtk()
    git_installed, git_restart = _install_git()

    any_binary_installed = coreutils_installed or git_installed or rg_installed or rtk_installed
    needs_restart = cu_restart or git_restart or rg_restart or rtk_restart

    if any_binary_installed and needs_restart:
        print(
            "\n💡 One or more tools were freshly installed, which may have modified your PATH.\n"
            "   Please **restart your current shell/CLI process** before using these tools,\n"
            "   so that the updated PATH environment variable is loaded."
        )

    # 3. Delete uv.lock file
    lock_file = Path("uv.lock")
    if lock_file.exists():
        if _ask_yes_no(f"Remove {lock_file}?"):
            print(f"\n🗑️  Removing {lock_file} ...")
            try:
                lock_file.unlink()
                print(f"✅ Removed {lock_file}.")
            except OSError as e:
                print(f"⚠️  Could not remove {lock_file}: {e}")
        else:
            print(f"⏭️  Keeping {lock_file}.")

    # 4. Run uv sync
    if _ask_yes_no("Sync dependencies with uv?"):
        if not run_command(["uv", "sync"], "Syncing dependencies with uv"):
            print(
                "\n💔 Oops! Something went wrong while syncing dependencies.\n"
                "   Please check the error messages above and try again.\n"
                "   If the issue persists, you may need to install dependencies manually."
            )
            return 1
    else:
        print("⏭️  Skipping uv sync. Dependencies may be out of date.")

    # 5. Keep the runtime version config in sync with KIMIX_BASE_VERSION so
    #    the Python fallback reports the same version as the compiled runtime.
    _sync_kimix_native_version(KIMIX_BASE_VERSION)

    # 6. Install the kimix_base native runtime (download + unpack into bin/)
    _install_kimix_native()

    # 7. Run uv tool install -e .
    if _ask_yes_no("Install tool in editable mode?"):
        if not run_command(["uv", "tool", "install", "-e", "."], "Installing tool in editable mode"):
            print(
                "\n💔 Oops! Something went wrong while installing the tool.\n"
                "   Please check the error messages above and try again.\n"
                "   If the issue persists, you may need to install the tool manually."
            )
            return 1
    else:
        print("⏭️  Skipping uv tool install. The tool may not be available on PATH.")

    print("\n🎉 All done! The project has been installed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
