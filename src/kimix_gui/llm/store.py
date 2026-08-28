"""Versioned persistence for exact, secret-free LLM selections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import orjson
from kimi_cli.metadata import WorkDirMeta
from kimi_cli.share import get_share_dir

from kimix_gui.kimi_workdir import resolve_kimi_work_dir
from kimix_gui.llm.axes import AXIS_THINKING_EFFORT
from kimix_gui.llm.domain import (
    PROBLEM_INVALID_SESSION_SELECTION,
    ChatGPTTarget,
    LLMProblem,
    LLMSelection,
    ProviderFileTarget,
    ProviderTarget,
)
from kimix_gui.llm.parameters import ParameterAssignment
from kimix_gui.preferences import (
    InterfacePreferences,
    parse_interface_preferences,
    serialize_interface_preferences,
)

STORE_FILENAME = "kimix-gui.json"
SelectionFormat = Literal["legacy", "variant", "parameters"]


def default_store_file() -> Path:
    """Return the shared GUI metadata file."""

    return get_share_dir() / STORE_FILENAME


def session_selection_file(work_dir: Path, session_id: str) -> Path:
    """Return the GUI metadata file inside one Kimi session."""

    if not session_id or Path(session_id).name != session_id:
        raise ValueError(f"Invalid session id: {session_id!r}")
    resolved = resolve_kimi_work_dir(work_dir)
    return WorkDirMeta(path=str(resolved)).sessions_dir / session_id / STORE_FILENAME


@dataclass(frozen=True, slots=True)
class StoredSelection:
    """One decoded selection, including an isolated corrupt-entry problem."""

    selection: LLMSelection | None = None
    problem: LLMProblem | None = None


SessionSelectionFileResolver = Callable[[Path, str], Path]


class KimixGuiConfigStore:
    """Persist interface preferences, Provider files and exact LLM selections."""

    VERSION = 6
    SESSION_VERSION = 4

    def __init__(
        self,
        store_file: Path | None = None,
        *,
        session_file_resolver: SessionSelectionFileResolver = session_selection_file,
    ) -> None:
        self.store_file = store_file or default_store_file()
        self._session_file_resolver = session_file_resolver
        self._data, migrated = self._load()
        if migrated:
            self._save()

    @property
    def interface(self) -> InterfacePreferences:
        return parse_interface_preferences(self._data["interface"])

    def set_interface(self, preferences: InterfacePreferences) -> None:
        self._data["interface"] = serialize_interface_preferences(preferences)
        self._save()

    def default_selection_for(self, work_dir: Path) -> LLMSelection | None:
        entry = self._work_dir_entry(work_dir, create=False)
        if not entry:
            return None
        return self._selection_from_value(entry.get("default_llm"), format="parameters")

    def set_default(self, work_dir: Path, selection: LLMSelection) -> None:
        self._remember_provider_file(selection.target)
        entry = self._work_dir_entry(work_dir, create=True)
        entry["default_llm"] = self._selection_value(selection)
        self._save()

    def provider_files(self) -> tuple[ProviderFileTarget, ...]:
        return tuple(
            ProviderFileTarget(path)
            for value in self._provider_file_values()
            if (path := self._path_from_value(value)) is not None
        )

    def add_provider_file(self, target: ProviderFileTarget | Path) -> None:
        resolved_target = (
            target if isinstance(target, ProviderFileTarget) else ProviderFileTarget(target)
        )
        self._remember_provider_file(resolved_target)
        self._save()

    def remove_provider_file(self, path: Path) -> None:
        values = self._provider_file_values()
        path_text = self._path_text(path)
        if path_text in values:
            values.remove(path_text)
            self._save()

    def provider_targets_for(self, work_dir: Path) -> tuple[ProviderFileTarget, ...]:
        targets = list(self.provider_files())
        default = self.default_selection_for(work_dir)
        if default is not None and isinstance(default.target, ProviderFileTarget):
            targets.append(default.target)
        deduplicated: dict[str, ProviderFileTarget] = {}
        for target in targets:
            deduplicated[self._target_storage_key(target)] = target
        return tuple(deduplicated.values())

    def session_selection_for(self, work_dir: Path, session_id: str) -> StoredSelection:
        metadata_file = self._session_file_resolver(work_dir, session_id)
        try:
            data = orjson.loads(metadata_file.read_bytes())
        except FileNotFoundError:
            return StoredSelection()
        except (OSError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
            return StoredSelection(
                problem=LLMProblem(
                    PROBLEM_INVALID_SESSION_SELECTION,
                    metadata_file,
                    str(exc),
                )
            )
        selection, migrated = self._session_selection_from_data(data)
        if selection is None:
            return StoredSelection(
                problem=LLMProblem(PROBLEM_INVALID_SESSION_SELECTION, metadata_file)
            )
        if migrated:
            self._write_session_selection(work_dir, session_id, selection)
        return StoredSelection(selection=selection)

    def set_session(self, work_dir: Path, session_id: str, selection: LLMSelection) -> None:
        self._remember_provider_file(selection.target)
        self._write_session_selection(work_dir, session_id, selection)
        self._save()

    def clear_session(self, work_dir: Path, session_id: str) -> None:
        try:
            self._session_file_resolver(work_dir, session_id).unlink()
        except FileNotFoundError:
            pass

    @classmethod
    def _session_selection_from_data(
        cls,
        data: object,
    ) -> tuple[LLMSelection | None, bool]:
        if not isinstance(data, dict):
            return None, False
        version = data.get("version")
        if version == 1:
            return cls._selection_from_value(data.get("config"), format="legacy"), True
        if version == 2:
            return cls._selection_from_value(data.get("llm"), format="legacy"), True
        if version == 3:
            return cls._selection_from_value(data.get("llm"), format="variant"), True
        if version == cls.SESSION_VERSION:
            return cls._selection_from_value(data.get("llm"), format="parameters"), False
        return None, False

    def _write_session_selection(
        self,
        work_dir: Path,
        session_id: str,
        selection: LLMSelection,
    ) -> None:
        self._write_json(
            self._session_file_resolver(work_dir, session_id),
            {"version": self.SESSION_VERSION, "llm": self._selection_value(selection)},
        )

    def _work_dir_entry(self, work_dir: Path, *, create: bool) -> dict[str, Any]:
        work_dirs: dict[str, Any] = self._data["work_dirs"]
        key = self._path_text(work_dir)
        entry = work_dirs.get(key)
        if isinstance(entry, dict):
            return entry
        if not create:
            return {}
        fresh: dict[str, Any] = {}
        work_dirs[key] = fresh
        return fresh

    def _provider_file_values(self) -> list[Any]:
        values: list[Any] = self._data["provider_files"]
        return values

    def _remember_provider_file(self, target: ProviderTarget) -> None:
        if not isinstance(target, ProviderFileTarget):
            return
        values = self._provider_file_values()
        path_text = self._path_text(target.path)
        if path_text not in values:
            values.append(path_text)

    @staticmethod
    def _target_storage_key(target: ProviderFileTarget) -> str:
        return f"{target.path}\0{target.model_override or ''}"

    @staticmethod
    def _path_text(path: Path) -> str:
        return str(path.expanduser().resolve(strict=False))

    @staticmethod
    def _path_from_value(value: object) -> Path | None:
        if isinstance(value, str) and value:
            return Path(value).expanduser().resolve(strict=False)
        return None

    @classmethod
    def _target_from_value(cls, value: object) -> ProviderTarget | None:
        if not isinstance(value, dict):
            return None
        kind = value.get("kind")
        if not isinstance(kind, str):
            return None
        if kind in {"provider_file", "config_file"}:
            path = cls._path_from_value(value.get("path"))
            override = value.get("model_override")
            if path is None or (override is not None and not isinstance(override, str)):
                return None
            return ProviderFileTarget(path, override or None)
        if kind == "chatgpt":
            model = value.get("model")
            if isinstance(model, str) and model.strip():
                return ChatGPTTarget(model)
        return None

    @classmethod
    def _selection_from_value(
        cls,
        value: object,
        *,
        format: SelectionFormat,
    ) -> LLMSelection | None:
        if format == "legacy":
            target = cls._legacy_target_from_value(value)
            if target is None:
                return None
            return LLMSelection(target, pinned=not isinstance(target, ChatGPTTarget))
        if not isinstance(value, dict):
            return None
        target = cls._target_from_value(value.get("target"))
        if target is None:
            return None
        if format == "variant":
            migrated = cls._selection_from_variant(target, value.get("variant"))
            return migrated
        parameters = value.get("parameters")
        pinned = value.get("pinned")
        if not isinstance(parameters, dict) or not isinstance(pinned, bool):
            return None
        try:
            assignment = ParameterAssignment(parameters)
        except ValueError:
            return None
        return LLMSelection(target, assignment, pinned)

    @staticmethod
    def _selection_from_variant(
        target: ProviderTarget,
        value: object,
    ) -> LLMSelection | None:
        if not isinstance(value, dict):
            return None
        kind = value.get("kind")
        if not isinstance(kind, str):
            return None
        variant_value = value.get("value")
        if kind in {"configured", "provider_default"} and variant_value is None:
            return LLMSelection(target)
        if kind == "legacy_default" and variant_value is None:
            return LLMSelection(target, pinned=False)
        if kind != "reasoning_effort" or not isinstance(variant_value, str):
            return None
        try:
            parameters = ParameterAssignment({AXIS_THINKING_EFFORT: variant_value})
        except ValueError:
            return None
        return LLMSelection(target, parameters)

    @classmethod
    def _legacy_target_from_value(cls, value: object) -> ProviderTarget | None:
        if isinstance(value, str):
            path = cls._path_from_value(value)
            return ProviderFileTarget(path) if path is not None else None
        return cls._target_from_value(value)

    @classmethod
    def _selection_value(cls, selection: LLMSelection) -> dict[str, Any]:
        return {
            "target": cls._target_value(selection.target),
            "parameters": dict(selection.parameters.entries),
            "pinned": selection.pinned,
        }

    @classmethod
    def _target_value(cls, target: ProviderTarget) -> dict[str, Any]:
        if isinstance(target, ChatGPTTarget):
            return {"kind": target.kind, "model": target.model}
        if isinstance(target, ProviderFileTarget):
            value: dict[str, Any] = {
                "kind": target.kind,
                "path": cls._path_text(target.path),
            }
            if target.model_override is not None:
                value["model_override"] = target.model_override
            return value
        raise TypeError(f"Unsupported persisted Provider target: {target!r}")

    def _load(self) -> tuple[dict[str, Any], bool]:
        try:
            raw = orjson.loads(self.store_file.read_bytes())
        except OSError, orjson.JSONDecodeError, TypeError, ValueError:
            return self._empty_data(), False
        if not isinstance(raw, dict):
            return self._empty_data(), False
        version = raw.get("version")
        if not isinstance(version, int) or version not in {
            3,
            4,
            5,
            self.VERSION,
        }:
            return self._empty_data(), False
        selection_format: SelectionFormat
        if version in {3, 4}:
            selection_format = "legacy"
        elif version == 5:
            selection_format = "variant"
        else:
            selection_format = "parameters"
        source_files = raw.get("configs") if selection_format == "legacy" else raw.get("provider_files")
        source_work_dirs = raw.get("work_dirs")
        data = self._empty_data()
        data["interface"] = serialize_interface_preferences(
            parse_interface_preferences(raw.get("interface"))
        )
        if isinstance(source_files, list):
            data["provider_files"] = list(
                dict.fromkeys(
                    self._path_text(path)
                    for value in source_files
                    if (path := self._path_from_value(value)) is not None
                )
            )
        if isinstance(source_work_dirs, dict):
            for work_dir, value in source_work_dirs.items():
                if not isinstance(work_dir, str) or not isinstance(value, dict):
                    continue
                selection_value = (
                    value.get("default")
                    if selection_format == "legacy"
                    else value.get("default_llm")
                )
                selection = self._selection_from_value(
                    selection_value,
                    format=selection_format,
                )
                if selection is not None:
                    data["work_dirs"][work_dir] = {
                        "default_llm": self._selection_value(selection)
                    }
        return data, version != self.VERSION

    def _empty_data(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "interface": serialize_interface_preferences(InterfacePreferences()),
            "provider_files": [],
            "work_dirs": {},
        }

    def _save(self) -> None:
        self._write_json(self.store_file, self._data)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
        temporary.write_bytes(payload)
        temporary.replace(path)
