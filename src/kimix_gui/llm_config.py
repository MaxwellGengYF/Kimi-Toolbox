"""Provider validation and shared GUI configuration persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import orjson
from kimi_cli.metadata import WorkDirMeta
from kimi_cli.share import get_share_dir

import kimix
from kimi_agent_sdk import Config
from kimix.utils.config import (
    _create_config,
    _inherit_sub_provider_defaults,
    _pick_main_from_sub_providers,
)
from kimix_gui.kimi_workdir import resolve_kimi_work_dir
from kimix_gui.preferences import (
    InterfacePreferences,
    parse_interface_preferences,
    serialize_interface_preferences,
)

#: Why a provider config cannot be used. Machine-readable values: the Qt layer maps
#: them onto translated sentences, because these problems *are* shown to the user
#: (``qt/settings_dialog.py`` puts them in ``#settings-error``, and ``app.py`` raises
#: them as a toast). Never compare display strings to decide anything.
CONFIG_NOT_JSON = "not_json"
CONFIG_FILE_MISSING = "file_missing"
CONFIG_INVALID_JSON = "invalid_json"
CONFIG_NOT_AN_OBJECT = "not_an_object"
CONFIG_INVALID = "invalid_config"
CONFIG_UNAVAILABLE = "unavailable"
CONFIG_INVALID_SESSION_REFERENCE = "invalid_session_reference"

#: English one-liners for logs, tracebacks and ``str(exception)``. Deliberately *not*
#: the UI text: the UI reads :attr:`ConfigProblem.kind` and translates it. Keeping an
#: English spelling here means a traceback and an SDK log still agree on wording.
_PROBLEM_TEXT: dict[str, str] = {
    CONFIG_NOT_JSON: "Kimix configuration must be a JSON file: {path}",
    CONFIG_FILE_MISSING: "Configuration file does not exist: {path}",
    CONFIG_INVALID_JSON: "Invalid JSON configuration {path}: {reason}",
    CONFIG_NOT_AN_OBJECT: "Kimix configuration must contain a JSON object: {path}",
    CONFIG_INVALID: "Invalid Kimix configuration {path}: {reason}",
    CONFIG_UNAVAILABLE: "Configuration file is unavailable: {path}",
    CONFIG_INVALID_SESSION_REFERENCE: "Invalid session configuration reference: {path}",
}


@dataclass(frozen=True, slots=True)
class ConfigProblem:
    """A structured reason a provider config is unusable, plus the data to show.

    ``kind`` is one of the ``CONFIG_*`` constants, ``path`` is the file involved and
    ``reason`` carries the underlying parser or OS message when there is one (that
    part comes from outside this project and is not ours to translate).
    """

    kind: str
    path: Path
    reason: str = ""

    def __str__(self) -> str:
        template = _PROBLEM_TEXT.get(self.kind, "{path}")
        return template.format(path=self.path, reason=self.reason)


class LLMConfigError(ValueError):
    """Raised when an LLM configuration cannot be loaded safely.

    Carries :attr:`problem` so the Qt layer can phrase it in the user's language; the
    exception's own ``str()`` stays English for tracebacks.
    """

    def __init__(self, problem: ConfigProblem) -> None:
        super().__init__(str(problem))
        self.problem = problem


def default_config_path() -> Path:
    """Return Kimix's package-level default provider JSON path."""

    package_file = kimix.__file__
    if package_file is None:
        return Path("default_config.json").resolve(strict=False)
    return (Path(package_file).resolve().parent / "default_config.json").resolve(strict=False)


@dataclass(frozen=True, slots=True)
class LLMConfigReference:
    """A reloadable config path plus a redacted summary for display."""

    path: Path
    model_name: str
    provider_type: str
    base_url: str
    credential: str
    inspected: bool = True
    """Whether the provider JSON was inspected successfully.

    ``False`` marks a placeholder built by :func:`unavailable_config_reference`.
    Availability logic reads this flag; ``provider_type`` / ``base_url`` /
    ``credential`` are display strings and must never be compared against.
    """

    file_format: str = "Unknown"
    display_name: str | None = None
    model_override: str | None = None
    max_context_size: int | None = None
    max_tokens: int | None = None
    capabilities: tuple[str, ...] = ()
    thinking_effort: str | None = None
    show_thinking_stream: bool | None = None
    error: ConfigProblem | None = None
    """Why this reference is unusable, or ``None`` when it loaded cleanly.

    Structured rather than a sentence: it is displayed, so the Qt layer has to phrase
    it, and ``config_file_available`` only ever asks whether it is ``None``.
    """

    @property
    def label(self) -> str:
        return self.display_name or self.model_name or self.path.name

    @property
    def source(self) -> ConfigFileSource:
        return ConfigFileSource(self.path, self.model_override)


@dataclass(frozen=True, slots=True, init=False)
class ConfigFileSource:
    """A provider JSON plus an optional model override."""

    kind: Literal["config_file"]
    path: Path
    model_override: str | None = None

    def __init__(
        self,
        path: Path,
        model_override: str | None = None,
        *,
        kind: Literal["config_file"] = "config_file",
    ) -> None:
        if kind != "config_file":
            raise ValueError(f"Invalid config source kind: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path.expanduser().resolve(strict=False))
        object.__setattr__(self, "model_override", model_override)


@dataclass(frozen=True, slots=True, init=False)
class ChatGPTSource:
    """A model served by the connected ChatGPT Codex subscription."""

    kind: Literal["chatgpt"]
    model: str

    def __init__(
        self,
        model: str,
        *,
        kind: Literal["chatgpt"] = "chatgpt",
    ) -> None:
        if kind != "chatgpt":
            raise ValueError(f"Invalid ChatGPT source kind: {kind}")
        normalized = model.strip()
        if not normalized:
            raise ValueError("ChatGPT model must not be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "model", normalized)


LLMSource = ConfigFileSource | ChatGPTSource


@dataclass(frozen=True, slots=True)
class ChatGPTModelReference:
    """Secret-free metadata for one built-in ChatGPT Codex model."""

    model_name: str
    display_name: str | None = None
    max_context_size: int = 272_000
    max_tokens: int | None = 128_000
    capabilities: tuple[str, ...] = ("thinking",)
    supported_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    input_modalities: tuple[str, ...] = ("text", "image")
    priority: int = 10_000
    available: bool = False
    stale: bool = True
    problem_code: str | None = "login_required"

    @property
    def label(self) -> str:
        return self.display_name or self.model_name

    @property
    def source(self) -> ChatGPTSource:
        return ChatGPTSource(self.model_name)

    @property
    def provider_type(self) -> str:
        return "openai-codex"

    @property
    def base_url(self) -> str:
        return "https://chatgpt.com/backend-api/codex"

    @property
    def credential(self) -> str:
        return "oauth-managed"

    @property
    def file_format(self) -> str:
        return "builtin"


LLMReference = LLMConfigReference | ChatGPTModelReference


def llm_reference_available(reference: LLMReference) -> bool:
    """Return whether a reference can currently create a session."""

    if isinstance(reference, ChatGPTModelReference):
        return reference.available
    return config_file_available(reference)


def reference_from_source(source: LLMSource) -> LLMReference:
    """Build a display reference without reading OAuth credentials."""

    if isinstance(source, ChatGPTSource):
        return ChatGPTModelReference(model_name=source.model)
    try:
        return inspect_llm_config(source.path, model_override=source.model_override)
    except LLMConfigError as exc:
        return replace(
            unavailable_config_reference(source.path, model_override=source.model_override),
            error=exc.problem,
        )


def inspect_llm_config(
    config_file: Path,
    *,
    model_override: str | None = None,
) -> LLMConfigReference:
    """Validate a Kimix provider JSON and return a secret-free summary."""

    path, provider_dict = _load_kimix_provider_json(config_file)
    preview_dict = dict(provider_dict)
    preview_dict.pop("env", None)
    config = _build_sdk_config(preview_dict, path)
    model = config.model
    provider = config.provider
    if model is None or provider is None:
        # The SDK accepted the file but left the halves this summary reads. Callers
        # handle ``LLMConfigError`` (the settings dialog renders the problem); an
        # AssertionError escaped them, and under ``-O`` it became an AttributeError.
        missing = "model" if model is None else "provider"
        raise LLMConfigError(ConfigProblem(CONFIG_INVALID, path, f"no {missing} in config"))
    model_name = model_override or model.model
    profile_name = provider_dict.get("name")
    display_name = str(profile_name) if profile_name else model.display_name
    provider_type = str(provider.type)
    base_url = _redact_url(provider.base_url) if provider.base_url else "Not set in file"
    if provider_dict.get("oauth"):
        credential = "OAuth"
    elif provider_dict.get("api_key"):
        credential = "API key configured"
    elif _has_environment_credential(provider_dict):
        credential = "Environment"
    else:
        credential = "Not stored in file"

    return LLMConfigReference(
        path=path,
        model_name=model_name,
        provider_type=provider_type,
        base_url=base_url,
        credential=credential,
        file_format=_config_format(path),
        display_name=display_name,
        model_override=model_override,
        max_context_size=model.max_context_size,
        max_tokens=config.max_tokens,
        capabilities=tuple(sorted(str(item) for item in (model.capabilities or ()))),
        thinking_effort=(
            str(provider_dict["thinking_effort"])
            if provider_dict.get("thinking_effort") is not None
            else None
        ),
        show_thinking_stream=(
            bool(provider_dict["show_thinking_stream"])
            if provider_dict.get("show_thinking_stream") is not None
            else None
        ),
    )


def load_kimix_sdk_config(config_file: Path) -> Config:
    """Load a Kimix flat provider JSON as the SDK Config used for a session."""

    path, provider_dict = _load_kimix_provider_json(config_file)
    return _build_sdk_config(provider_dict, path)


def load_kimix_provider_dict(config_file: Path) -> dict[str, Any]:
    """Load the normalized provider mapping consumed by Kimix session factories."""

    _path, provider_dict = _load_kimix_provider_json(config_file)
    return provider_dict


def _load_kimix_provider_json(config_file: Path) -> tuple[Path, dict[str, Any]]:
    path = config_file.expanduser().resolve(strict=False)
    if path.suffix.lower() != ".json":
        raise LLMConfigError(ConfigProblem(CONFIG_NOT_JSON, path))
    if not path.is_file():
        raise LLMConfigError(ConfigProblem(CONFIG_FILE_MISSING, path))
    try:
        loaded = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise LLMConfigError(ConfigProblem(CONFIG_INVALID_JSON, path, str(exc))) from exc
    if not isinstance(loaded, dict):
        raise LLMConfigError(ConfigProblem(CONFIG_NOT_AN_OBJECT, path))

    provider_dict = deepcopy(loaded)
    sub_provider = provider_dict.pop("sub_provider", None)
    sub_providers = provider_dict.pop("sub_providers", None)
    _inherit_sub_provider_defaults(provider_dict, sub_provider, sub_providers)
    _pick_main_from_sub_providers(provider_dict, sub_provider, sub_providers)
    return path, provider_dict


def _build_sdk_config(provider_dict: dict[str, Any], path: Path) -> Config:
    try:
        config, _normalized = _create_config(provider_dict)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise LLMConfigError(ConfigProblem(CONFIG_INVALID, path, str(exc))) from exc
    return config


def _has_environment_credential(provider_dict: dict[str, Any]) -> bool:
    environment = provider_dict.get("env")
    if isinstance(environment, dict) and any(
        key in environment for key in ("KIMI_API_KEY", "KIMIX_API_KEY")
    ):
        return True
    return bool(os.environ.get("KIMI_API_KEY") or os.environ.get("KIMIX_API_KEY"))


def _redact_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "Configured endpoint"


def unavailable_config_reference(
    path: Path,
    *,
    model_override: str | None = None,
) -> LLMConfigReference:
    """Build a displayable reference when startup configuration is invalid."""

    resolved = path.expanduser().resolve(strict=False)
    return LLMConfigReference(
        path=resolved,
        model_name=model_override or "Configuration unavailable",
        # Display-only placeholders; ``inspected=False`` is the machine-readable part.
        provider_type="Unavailable",
        base_url="Unavailable",
        credential="Unavailable",
        inspected=False,
        file_format="JSON",
        model_override=model_override,
        error=ConfigProblem(CONFIG_UNAVAILABLE, resolved),
    )


def _config_format(path: Path) -> str:
    return "JSON" if path.suffix.lower() == ".json" else "Unsupported"


def config_file_available(reference: LLMConfigReference) -> bool:
    """Return whether a reference can be loaded without an external missing file."""

    return reference.error is None and reference.inspected and reference.path.is_file()


SessionConfigFileResolver = Callable[[Path, str], Path]

STORE_FILENAME = "kimix-gui.json"


def default_store_file() -> Path:
    """Return the share-dir file that stores global GUI configuration."""

    return get_share_dir() / STORE_FILENAME


def session_config_file(work_dir: Path, session_id: str) -> Path:
    """Return the config reference file inside a Kimi session."""

    if not session_id or Path(session_id).name != session_id:
        raise ValueError(f"Invalid session id: {session_id!r}")
    resolved = resolve_kimi_work_dir(work_dir)
    return WorkDirMeta(path=str(resolved)).sessions_dir / session_id / STORE_FILENAME


class KimixGuiConfigStore:
    """Persist GUI preferences, global defaults, and session-local references."""

    VERSION = 4
    SESSION_VERSION = 2

    def __init__(
        self,
        store_file: Path | None = None,
        *,
        session_file_resolver: SessionConfigFileResolver = session_config_file,
    ) -> None:
        self.store_file = store_file or default_store_file()
        self._session_file_resolver = session_file_resolver
        self._data = self._load()

    @property
    def interface(self) -> InterfacePreferences:
        return parse_interface_preferences(self._data["interface"])

    def set_interface(self, preferences: InterfacePreferences) -> None:
        self._data["interface"] = serialize_interface_preferences(preferences)
        self._save()

    def default_source_for(self, work_dir: Path) -> LLMSource | None:
        entry = self._work_dir_entry(work_dir, create=False)
        return self._source_from_value(entry.get("default")) if entry else None

    def default_for(self, work_dir: Path) -> LLMReference | None:
        source = self.default_source_for(work_dir)
        return reference_from_source(source) if source is not None else None

    def set_default(self, work_dir: Path, reference: LLMReference) -> None:
        self._add_reference_config(reference)
        self._work_dir_entry(work_dir, create=True)["default"] = self._source_value(
            reference.source
        )
        self._save()

    def configs(self) -> list[LLMConfigReference]:
        configs = self._config_paths()
        return [
            reference
            for value in configs
            if (reference := self._reference_from_value(value)) is not None
        ]

    def add_config(self, reference: LLMConfigReference) -> None:
        self._add_config(reference)
        self._save()

    def remove_config(self, path: Path) -> None:
        configs = self._config_paths()
        path_text = self._path_text(path)
        if path_text in configs:
            configs.remove(path_text)
            self._save()

    def session_source_for(self, work_dir: Path, session_id: str) -> LLMSource | None:
        metadata_file = self._session_file_resolver(work_dir, session_id)
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except OSError, TypeError, ValueError:
            return None
        return self._session_source_from_data(data)

    def session_for(self, work_dir: Path, session_id: str) -> LLMReference | None:
        metadata_file = self._session_file_resolver(work_dir, session_id)
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            return self._invalid_session_reference(metadata_file, str(exc))
        source = self._session_source_from_data(data)
        if source is None:
            return self._invalid_session_reference(metadata_file)
        return reference_from_source(source)

    @classmethod
    def _session_source_from_data(cls, data: object) -> LLMSource | None:
        if not isinstance(data, dict):
            return None
        version = data.get("version")
        if version == 1:
            return cls._source_from_value(data.get("config"))
        if version == cls.SESSION_VERSION:
            return cls._source_from_value(data.get("llm"))
        return None

    def set_session(
        self,
        work_dir: Path,
        session_id: str,
        reference: LLMReference,
    ) -> None:
        self._add_reference_config(reference)
        self._write_session_source(work_dir, session_id, reference.source)
        self._save()

    def clear_session(self, work_dir: Path, session_id: str) -> None:
        try:
            self._session_file_resolver(work_dir, session_id).unlink()
        except FileNotFoundError:
            pass

    def references_for(self, work_dir: Path) -> list[LLMConfigReference]:
        entry = self._work_dir_entry(work_dir, create=False)
        references = self.configs()
        if entry:
            default = self._reference_from_value(entry.get("default"))
            if default is not None:
                references.append(default)
        default_path = default_config_path()
        if default_path.is_file():
            try:
                references.append(inspect_llm_config(default_path))
            except LLMConfigError:
                pass
        deduplicated: dict[Path, LLMConfigReference] = {}
        for reference in references:
            path = reference.path.expanduser().resolve(strict=False)
            deduplicated[path] = reference
        return list(deduplicated.values())

    def _add_config(self, reference: LLMConfigReference) -> None:
        configs = self._config_paths()
        path = self._path_text(reference.path)
        if path not in configs:
            configs.append(path)

    def _add_reference_config(self, reference: LLMReference) -> None:
        if isinstance(reference, LLMConfigReference):
            self._add_config(reference)

    @staticmethod
    def _path_text(path: Path) -> str:
        return str(path.expanduser().resolve(strict=False))

    @staticmethod
    def _path_from_value(value: object) -> Path | None:
        if isinstance(value, str) and value:
            return Path(value).expanduser().resolve(strict=False)
        return None

    @classmethod
    def _reference_from_value(cls, value: object) -> LLMConfigReference | None:
        source = cls._source_from_value(value)
        if not isinstance(source, ConfigFileSource):
            return None
        try:
            return inspect_llm_config(source.path, model_override=source.model_override)
        except LLMConfigError as exc:
            return replace(
                unavailable_config_reference(
                    source.path,
                    model_override=source.model_override,
                ),
                error=exc.problem,
            )

    @classmethod
    def _source_from_value(cls, value: object) -> LLMSource | None:
        if isinstance(value, str):
            path = cls._path_from_value(value)
            return ConfigFileSource(path) if path is not None else None
        if not isinstance(value, dict):
            return None
        kind = value.get("kind")
        if kind == "config_file":
            path = cls._path_from_value(value.get("path"))
            override = value.get("model_override")
            if path is None or (override is not None and not isinstance(override, str)):
                return None
            return ConfigFileSource(path, override or None)
        if kind == "chatgpt":
            model = value.get("model")
            if isinstance(model, str) and model.strip():
                return ChatGPTSource(model)
        return None

    @classmethod
    def _source_value(cls, source: LLMSource) -> dict[str, Any]:
        if isinstance(source, ChatGPTSource):
            return {"kind": source.kind, "model": source.model}
        value: dict[str, Any] = {"kind": source.kind, "path": cls._path_text(source.path)}
        if source.model_override is not None:
            value["model_override"] = source.model_override
        return value

    @staticmethod
    def _invalid_session_reference(path: Path, reason: str = "") -> LLMConfigReference:
        return replace(
            unavailable_config_reference(path),
            error=ConfigProblem(CONFIG_INVALID_SESSION_REFERENCE, path, reason),
        )

    def _write_session_source(
        self,
        work_dir: Path,
        session_id: str,
        source: LLMSource,
    ) -> None:
        self._write_json(
            self._session_file_resolver(work_dir, session_id),
            {"version": self.SESSION_VERSION, "llm": self._source_value(source)},
        )

    def _work_dir_entry(self, work_dir: Path, *, create: bool) -> dict[str, Any]:
        work_dirs: dict[str, Any] = self._data["work_dirs"]
        key = str(work_dir.expanduser().resolve(strict=False))
        entry = work_dirs.get(key)
        if isinstance(entry, dict):
            return entry
        # ``_load`` validates the two top-level containers but not one work dir's entry.
        # Reading already treated a corrupt entry as absent while writing asserted on it,
        # so a hand-edited metadata file crashed the save path and survived the read one.
        if not create:
            return {}
        fresh: dict[str, Any] = {}
        work_dirs[key] = fresh
        return fresh

    def _config_paths(self) -> list[Any]:
        """The stored config paths. ``_load`` rejects a file whose ``configs`` is not a
        list, falling back to ``_empty_data``, so this key always holds one."""

        paths: list[Any] = self._data["configs"]
        return paths

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.store_file.read_text(encoding="utf-8"))
        except OSError, ValueError, TypeError:
            return self._empty_data()
        if not isinstance(data, dict) or data.get("version") not in (3, self.VERSION):
            return self._empty_data()
        if not isinstance(data.get("configs"), list):
            return self._empty_data()
        if not isinstance(data.get("work_dirs"), dict):
            return self._empty_data()
        data["interface"] = serialize_interface_preferences(
            parse_interface_preferences(data.get("interface"))
        )
        for entry in data["work_dirs"].values():
            if not isinstance(entry, dict) or "default" not in entry:
                continue
            source = self._source_from_value(entry.get("default"))
            if source is None:
                entry.pop("default", None)
            else:
                entry["default"] = self._source_value(source)
        data["version"] = self.VERSION
        return data

    def _empty_data(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "interface": serialize_interface_preferences(InterfacePreferences()),
            "configs": [],
            "work_dirs": {},
        }

    def _save(self) -> None:
        self._write_json(self.store_file, self._data)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
