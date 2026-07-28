from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kimix.ui.printing import print_error, print_info, print_warning
from kimix.utils.windows_env import refresh_env_from_registry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4096
DEFAULT_FE_PORT = 5173

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = REPO_ROOT / "src" / "app"


def _resolve_uv() -> str:
    """Return the `uv` executable path, or 'uv' if not found."""
    uv = shutil.which("uv")
    return uv if uv else "uv"


def _resolve_npm() -> str:
    """Return `npm` (or `npm.cmd` on Windows)."""
    return "npm.cmd" if sys.platform == "win32" else "npm"


def _can_run_node(npm: str) -> tuple[bool, str]:
    """Check that `npm`/`node` are callable.

    Returns (ok, message). `message` is empty when ok is True.
    """
    try:
        subprocess.run(
            [npm, "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return False, f"`{npm}` not found on PATH. Install Node.js and npm to run the frontend."
    except subprocess.TimeoutExpired:
        return False, f"`{npm} --version` timed out; Node.js/npm may be misconfigured."

    node = "node.exe" if sys.platform == "win32" else "node"
    if shutil.which(node) is None:
        return False, f"`{node}` not found on PATH. Node.js is required to run the frontend."

    return True, ""


def _node_modules_present() -> bool:
    """Return True if src/app/node_modules exists."""
    return (APP_DIR / "node_modules").is_dir()


def _build_frontend(npm: str) -> bool:
    """Run `npm run build` in the frontend directory. Returns True on success."""
    print_info("[gui] Building frontend...")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=APP_DIR,
        check=False,
    )
    return result.returncode == 0


def _find_available_port(host: str, preferred: int, max_attempts: int = 100) -> int:
    """Find an available port starting from ``preferred``.

    Tries binding to ``preferred``, ``preferred + 1``, etc.
    Returns the first available port or raises RuntimeError.
    """
    for offset in range(max_attempts):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available port found after {max_attempts} attempts "
        f"starting from {preferred}"
    )


def run_gui(args: Any) -> None:
    """Launch the Kimix backend and (optionally) the Vite frontend dev server."""
    original_cwd = os.getcwd()
    host = getattr(args, "host", DEFAULT_HOST)
    port = getattr(args, "port", DEFAULT_PORT)
    original_port = port
    port = _find_available_port(host, port)
    if port != original_port:
        print_warning(f"[gui] Port {original_port} in use, shifting to {port}")
    fe_port = getattr(args, "fe_port", DEFAULT_FE_PORT)
    build_first = getattr(args, "build", False)
    no_fe = getattr(args, "no_fe", False)
    config = getattr(args, "config", None)

    uv = _resolve_uv()
    npm = _resolve_npm()

    # ── Decide whether to launch the frontend ───────────────────────
    if no_fe:
        if build_first:
            print_warning("[gui] --build is ignored when --no-fe is set.")
        print_info("[gui] --no-fe set; frontend will be skipped.")
        run_frontend = False
    else:
        npm_ok, npm_msg = _can_run_node(npm)
        if not npm_ok:
            print_error(
                f"[gui] {npm_msg}\n"
                "        Run `kimix gui --no-fe` to start only the backend,\n"
                "        or install Node.js/npm and run `npm install` in src/app."
            )
            sys.exit(1)

        if not _node_modules_present():
            # Try to install automatically
            print_info("[gui] node_modules not found; running `npm install`...")

            # On Windows, refresh PATH from registry so newly installed Node.js/npm is visible
            if sys.platform == "win32":
                refresh_env_from_registry()

            result = subprocess.run(
                [npm, "install"],
                cwd=APP_DIR,
                check=False,
            )
            if result.returncode != 0:
                print_error(
                    "[gui] `npm install` failed.\n"
                    "        Run `cd src/app && npm install`, then try again.\n"
                    "        Or use `kimix gui --no-fe` to start only the backend."
                )
                sys.exit(1)

            # Refresh PATH again after install (Node.js may add to PATH via install script)
            if sys.platform == "win32":
                refresh_env_from_registry()

            # Verify that node_modules now exists
            if not _node_modules_present():
                print_error(
                    "[gui] `npm install` completed but node_modules is still missing.\n"
                    "        Run `cd src/app && npm install`, then try again.\n"
                    "        Or use `kimix gui --no-fe` to start only the backend."
                )
                sys.exit(1)

        run_frontend = True

    # ── Optional: build frontend first ──────────────────────────────
    if build_first and run_frontend:
        if not _build_frontend(npm):
            print_error("[gui] Frontend build failed.")
            sys.exit(1)

    # ── Backend ─────────────────────────────────────────────────────
    be_cmd = [
        uv, "run", "kimix", "serve",
        "--host", host,
        "--port", str(port),
    ]
    if config:
        be_cmd.append(f'--config={config}')
        print(config)
    print_info(f"[gui] Starting backend: {' '.join(be_cmd)}")
    be_proc = subprocess.Popen(
        be_cmd,
        cwd=original_cwd,
    )

    # ── Frontend ────────────────────────────────────────────────────
    fe_proc: subprocess.Popen | None = None
    if run_frontend:
        fe_cmd = [
            npm, "run", "dev",
            "--", "--host", host, "--port", str(fe_port),
        ]
        print_info(f"[gui] Starting frontend: {' '.join(fe_cmd)}")
        fe_proc = subprocess.Popen(
            fe_cmd,
            cwd=APP_DIR,
        )

    # ── Print banner ────────────────────────────────────────────────
    time.sleep(2.0)
    print()
    print("=" * 64)
    if run_frontend:
        print("  Kimix GUI — Backend + Frontend")
    else:
        print("  Kimix GUI — Backend only (frontend skipped)")
    print("=" * 64)
    print(f"  Backend      : http://{host}:{port}")
    print(f"  API Docs     : http://{host}:{port}/docs")
    if run_frontend:
        print(f"  Frontend     : http://{host}:{fe_port}")
    print("  Backend mode : DUMMY (stub session manager)")
    print("=" * 64)
    print("  Commands (frontend): /new /abort /status /sessions /messages")
    print("                       /clear /compact /export /exit")
    print("  Press Ctrl+C to stop.")
    print()

    # ── Shutdown handler ────────────────────────────────────────────
    procs: list[subprocess.Popen] = [p for p in (be_proc, fe_proc) if p is not None]

    def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
        print("\n[gui] Shutting down...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 2.0
        for p in procs:
            remaining = deadline - time.time()
            if remaining > 0 and p.poll() is None:
                try:
                    p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
        for p in procs:
            if p.poll() is None:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Wait for running processes ──────────────────────────────────
    try:
        be_proc.wait()
        if fe_proc is not None:
            fe_proc.wait()
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)
