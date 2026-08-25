from enum import StrEnum


class FileOpsWindow:
    """File operations window."""

    pass


class FileActions(StrEnum):
    READ = "read file"
    EDIT = "edit file"
    EDIT_OUTSIDE = "edit file outside working directory"


from .glob import Glob  # noqa: E402
from .grep_local import Grep  # noqa: E402
from .hash_line import HashEdit, HashLine, HashRead  # noqa: E402
from .read import ReadFile  # noqa: E402
from .read_media import ReadMediaFile  # noqa: E402
from .edit import ApplyPatchFile, EditFile  # noqa: E402
from .write import WriteFile  # noqa: E402

__all__ = (
    "ReadFile",
    "ReadMediaFile",
    "Glob",
    "Grep",
    "WriteFile",
    "EditFile",
    "ApplyPatchFile",
    "HashLine",
    "HashRead",
    "HashEdit",
)
