"""File modification time tracker.

Detects external file modifications between read and write operations.

Behavior
--------
- ``clean_file`` is called after a successful read. It records the file's
  current mtime as the *read baseline* and drops any write baseline, so a
  fresh read legitimizes a subsequent write.
- ``mark_dirty`` is called before a write/edit. It allows the write only when:
  * the file is missing (creating it is safe), or
  * the current mtime is not newer than the last read baseline (no external
    modification since the file was read), and
  * no write happened since the last read (no write-after-write without an
    intervening read).
  Otherwise it returns ``False`` and the caller must refuse the write.
"""

from pathlib import Path

from kimi_cli.utils.path import kaos_path_from_user_input


class FileMTime:
    """Track file modification times to detect external changes.

    Two baselines are kept per file:
    - ``_read_times``: mtime recorded by the last successful read
      (``clean_file``). A current mtime strictly newer than this baseline means
      the file changed externally since it was read.
    - ``_times``: mtime recorded by the last write (``mark_dirty`` when the
      write was allowed). A write performed with no intervening read is refused
      even if nothing changed externally.
    """

    def __init__(self) -> None:
        self._times: dict[str, float] = {}
        self._read_times: dict[str, float] = {}

    def _resolve(self, path: str) -> str:
        """Resolve *path* to a canonical absolute path for use as a dict key.

        Both relative and absolute paths resolve to the same key.
        """
        try:
            p = kaos_path_from_user_input(path)
            return str(p.canonical())
        except Exception:
            return str(Path(path).resolve())

    def mark_dirty(self, path: str) -> bool:
        """Check whether *path* is safe to write now.

        Returns ``True`` when the write may proceed:
        - the file does not exist (it will be created), or
        - the file was read (``clean_file``) and its mtime has not changed
          since that read, and no write has been performed since that read.

        Returns ``False`` (and the caller must refuse the write) when:
        - the file was modified externally since the last read, or
        - a write was already performed since the last read (write-after-write
          without an intervening read).
        """
        key = self._resolve(path)
        try:
            current_mtime = Path(key).stat().st_mtime
        except (FileNotFoundError, OSError):
            # File doesn't exist yet — safe to create.
            self._times[key] = 0.0
            self._read_times.pop(key, None)
            return True

        read_t = self._read_times.get(key)
        if read_t is not None and current_mtime > read_t:
            # File changed externally since it was last read.
            return False

        write_t = self._times.get(key)
        if write_t is not None and current_mtime >= write_t:
            # A write already happened since the last read (no intervening read).
            return False

        self._times[key] = current_mtime
        return True

    def clean_file(self, path: str) -> None:
        """Record a successful read of *path*.

        Stores the file's current mtime as the read baseline and removes any
        write baseline, so a subsequent write is legitimate. If the file is
        missing, both baselines are dropped.
        """
        key = self._resolve(path)
        try:
            current_mtime = Path(key).stat().st_mtime
        except (FileNotFoundError, OSError):
            self._read_times.pop(key, None)
            self._times.pop(key, None)
            return
        self._read_times[key] = current_mtime
        self._times.pop(key, None)
