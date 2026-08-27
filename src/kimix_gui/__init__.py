"""PySide6 desktop GUI for the public Kimi Agent SDK."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kimix_gui.app import KimixGuiApp
    from kimix_gui.backend import SessionOptions

try:
    __version__ = version("kimix")
except PackageNotFoundError:  # Source imported without installing the distribution.
    __version__ = "0+unknown"

__all__ = ["KimixGuiApp", "SessionOptions"]


def __getattr__(name: str) -> Any:
    """Load public GUI objects only when the optional Qt client is requested."""

    if name == "KimixGuiApp":
        from kimix_gui.app import KimixGuiApp

        return KimixGuiApp
    if name == "SessionOptions":
        from kimix_gui.backend import SessionOptions

        return SessionOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
