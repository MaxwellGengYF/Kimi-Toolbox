#!/usr/bin/env python3
"""Install script for the project using uv."""

import os
import shutil
import subprocess
import re
import sys
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


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask the user a yes/no question.

    In non-interactive environments (e.g. CI pipelines) the *default*
    value is returned immediately so the script does not hang.
    """
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


def main() -> int:
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

    # 5. Run uv tool install -e .
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
