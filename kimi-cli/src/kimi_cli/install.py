"""rtk binary installation and discovery.

Downloads the rtk binary from GitHub releases and installs it to the shared bin directory.
Mirrors the same pattern used for ripgrep installation in kimi_cli.tools.file.grep_local.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import kimi_cli
from kimi_cli._rtk_common import (
    RTK_BASE_URL,
    RTK_VERSION,
    _detect_rtk_target,
    _extract_rtk_archive,
    _rtk_archive_name,
    _rtk_binary_name,
    _rtk_download_url,
)
from kimi_cli.share import get_share_dir
from kimi_cli.utils.logging import logger

_RTK_DOWNLOAD_LOCK = asyncio.Lock()


def _find_existing_rtk(bin_name: str) -> Path | None:
    """Find rtk binary: share dir, bundled deps, then PATH."""
    share_bin = get_share_dir() / "bin" / bin_name
    if share_bin.is_file():
        return share_bin

    assert kimi_cli.__file__ is not None
    local_dep = Path(kimi_cli.__file__).parent / "deps" / "bin" / bin_name
    if local_dep.is_file():
        return local_dep

    system_rtk = shutil.which("rtk")
    if system_rtk:
        return Path(system_rtk)

    return None


async def _download_and_install_rtk(bin_name: str) -> Path:
    target = _detect_rtk_target()

    filename = _rtk_archive_name(target)
    url = _rtk_download_url(RTK_VERSION, target)
    logger.info("Downloading rtk from {url}", url=url)

    share_bin_dir = get_share_dir() / "bin"
    share_bin_dir.mkdir(parents=True, exist_ok=True)
    destination = share_bin_dir / bin_name

    import aiohttp

    from kimi_cli.utils.aiohttp import new_client_session

    download_timeout = aiohttp.ClientTimeout(total=600, sock_read=60, sock_connect=15)
    async with new_client_session(timeout=download_timeout) as session:
        with tempfile.TemporaryDirectory(prefix="kimi-rtk-") as tmpdir:
            archive_path = Path(tmpdir) / filename

            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    with open(archive_path, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1024 * 64):
                            if chunk:
                                fh.write(chunk)
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise RuntimeError("Failed to download rtk binary") from exc

            _extract_rtk_archive(archive_path, destination, target, bin_name)

    logger.info("Installed rtk to {destination}", destination=destination)
    return destination


async def ensure_rtk_path() -> str:
    bin_name = _rtk_binary_name()
    existing = _find_existing_rtk(bin_name)
    if existing:
        return str(existing)

    async with _RTK_DOWNLOAD_LOCK:
        existing = _find_existing_rtk(bin_name)
        if existing:
            return str(existing)

        downloaded = await _download_and_install_rtk(bin_name)
        return str(downloaded)


__all__ = [
    "RTK_VERSION",
    "RTK_BASE_URL",
    "_rtk_binary_name",
    "_find_existing_rtk",
    "_detect_rtk_target",
    "_download_and_install_rtk",
    "ensure_rtk_path",
]
