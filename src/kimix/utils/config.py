from __future__ import annotations

from typing import Any
import os
import sys
from pathlib import Path
import orjson
import kimix.base as base
from kimi_cli.config import BackgroundConfig, LoopControl, SecretStr, NotificationConfig, MCPConfig, OAuthRef, OpenAISettings  # type: ignore[attr-defined]
from kimi_agent_sdk import Config
from . import _globals


# ── Provider validation constants ──────────────────────────────────────────

_REQUIRED_PROVIDER_KEYS: tuple[str, ...] = ("type", "max_context_size", "model", "url")


# Priority for picking a sub_provider as the main provider when root lacks 'model'.
# Earlier values in the tuple have higher priority.
_SUB_PROVIDER_PICK_PRIORITY: tuple[str | None, ...] = (None, "sub_agent", "planner")


# Keys that identify a specific provider entry and must never be inherited
# from the top-level config into sub-providers.
_SUB_PROVIDER_NO_INHERIT_KEYS: frozenset[str] = frozenset({"model", "model_name", "name", "role"})


# ── Sub-provider normalization ─────────────────────────────────────────────


def _inherit_sub_provider_defaults(
    config_data: dict[str, Any],
    sub_provider: Any,
    sub_providers: Any,
) -> None:
    """Fill missing keys in each sub-provider entry from the top-level config.

    Any key present in ``config_data`` (except identity keys such as ``model``
    and ``role``) is copied into a sub-provider entry only when that entry does
    not already define the key. Mutates the raw sub-provider entries in place
    so that later normalization/validation sees the inherited values.
    """
    if not config_data:
        return
    defaults = {
        k: v for k, v in config_data.items() if k not in _SUB_PROVIDER_NO_INHERIT_KEYS
    }
    if not defaults:
        return

    def _fill(entry: Any) -> None:
        if isinstance(entry, dict):
            for key, value in defaults.items():
                entry.setdefault(key, value)

    for src in (sub_provider, sub_providers):
        if isinstance(src, dict):
            _fill(src)
        elif isinstance(src, list):
            for entry in src:
                _fill(entry)


def _normalize_sub_providers(
    sub_provider: Any,
    sub_providers: Any,
) -> list[dict[str, Any]]:
    """Normalize ``sub_provider`` and ``sub_providers`` into a flat list.

    Each entry is validated to be a dict with required provider keys.
    Missing or empty ``role`` defaults to ``sub_agent``.
    """
    from kimix.base import print_debug, print_warning

    raw_entries: list[Any] = []
    for src in (sub_provider, sub_providers):
        if src is None:
            continue
        if isinstance(src, dict):
            raw_entries.append(src)
        elif isinstance(src, list):
            raw_entries.extend(src)
        else:
            print_warning(f"Ignoring invalid sub_provider value of type {type(src).__name__}")

    normalized: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            print_warning("Ignoring invalid sub_provider entry: expected dict")
            continue
        missing = [k for k in _REQUIRED_PROVIDER_KEYS if k not in entry]
        if missing:
            print_warning(
                f"Ignoring invalid sub_provider entry (missing keys: {', '.join(missing)})"
            )
            continue
        role = entry.get("role")
        if not role:
            entry = dict(entry)
            entry["role"] = "sub_agent"
        role = entry["role"]
        if role in seen_roles:
            print_debug(f"Multiple sub_providers with role '{role}'; using first match")
        seen_roles.add(role)
        normalized.append(entry)
    return normalized


def _pick_main_from_sub_providers(
    config_data: dict[str, Any],
    sub_provider: Any,
    sub_providers: Any,
) -> None:
    """If ``config_data`` lacks a ``model`` key, pick a sub_provider as the main provider.

    Priority (highest first):
      1. Entry with no ``role`` key (or empty role)
      2. Entry with ``role = "sub_agent"``
      3. Entry with ``role = "planner"``

    Mutates ``config_data`` in place by copying all non-``role`` keys from the winner.
    """
    from kimix.base import print_debug

    if "model" in config_data:
        return

    # Collect raw entries (pre-normalization) to distinguish "no role" from explicit roles.
    raw_entries: list[dict[str, Any]] = []
    for src in (sub_provider, sub_providers):
        if src is None:
            continue
        if isinstance(src, dict):
            raw_entries.append(src)
        elif isinstance(src, list):
            raw_entries.extend(src)

    picked: dict[str, Any] | None = None
    for role_priority in _SUB_PROVIDER_PICK_PRIORITY:
        if picked is not None:
            break
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            if role_priority is None:
                # no-role: entry has no 'role' key or role is empty/falsy
                if not entry.get("role"):
                    picked = entry
                    break
            else:
                if entry.get("role") == role_priority:
                    picked = entry
                    break

    if picked is not None:
        for k, v in picked.items():
            if k != "role":
                config_data[k] = v
        print_debug(
            f"Picked sub_provider (role='{picked.get('role', '')}') "
            f"as main provider (no 'model' in root)."
        )


# ── Config file loading ────────────────────────────────────────────────────


def _load_config_file(config_path: Path) -> dict[str, Any]:
    """Search for and load a JSON config file.

    Searches in order:
      1. Direct path (``config_path`` itself)
      2. Parent directories of CWD
      3. Parent directories of this file (``config.py``)
      4. ``PATH`` environment variable directories
    """
    from kimix.base import print_error

    found = False
    path = config_path

    if path.exists() and path.is_file():
        found = True
    else:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / path.name
            if candidate.exists() and candidate.is_file():
                path = candidate
                found = True
                break
        if not found:
            file_dir = Path(__file__).resolve().parent
            for parent in [file_dir, *file_dir.parents]:
                candidate = parent / path.name
                if candidate.exists() and candidate.is_file():
                    path = candidate
                    found = True
                    break
        if not found:
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                path_dir = path_dir.strip()
                if not path_dir:
                    continue
                candidate = Path(path_dir) / path.name
                if candidate.exists() and candidate.is_file():
                    path = candidate
                    found = True
                    break

    if not found:
        print_error(f"Config file not found: {str(config_path)}")
        sys.exit(1)

    path = path.resolve()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return orjson.loads(f.read())
    except orjson.JSONDecodeError as e:
        from kimix.base import print_warning
        print_warning(f"Invalid JSON in config file: {str(path)} ({e})")
    except Exception as e:
        from kimix.base import print_warning
        print_warning(f"Failed to load config file: {str(path)} ({e})")
    return {}


# ── Skill JSON loading ─────────────────────────────────────────────────────


def _load_skill_json() -> None:
    """Auto-load skill directories from ``.kimix/skill.json`` in CWD."""
    from kimix.base import print_debug, print_warning
    from kaos.path import KaosPath

    config_json_path = Path(".kimix/skill.json")
    if not config_json_path.exists():
        return

    print_debug(".kimix/skill.json exists.")
    try:
        config_json = orjson.loads(config_json_path.read_text(encoding="utf-8"))
        skill_dir_cfg = config_json.get("skill_dir")
        if skill_dir_cfg is not None:
            if isinstance(skill_dir_cfg, str):
                skill_dir_cfg = [skill_dir_cfg]
            if isinstance(skill_dir_cfg, list):
                skill_dirs_from_cfg = []
                for sd in skill_dir_cfg:
                    if not isinstance(sd, str):
                        continue
                    sd_path = Path(sd)
                    if not sd_path.is_absolute():
                        sd_path = Path.cwd() / sd_path
                    sd_path = sd_path.resolve()
                    if sd_path.exists() and sd_path.is_dir():
                        skill_dirs_from_cfg.append(KaosPath(str(sd_path)))
                        print_debug(f"Skill dir from config: {str(sd_path)}")
                    else:
                        print_warning(f"Skill dir from config not found: {str(sd_path)}")
                if skill_dirs_from_cfg:
                    existing = list(base._default_skill_dirs)
                    base.set_default_skill_dirs(existing + skill_dirs_from_cfg)
    except (orjson.JSONDecodeError, Exception) as e:
        print_warning(f"Failed to read skill_dir from .kimix/skill.json: {e}")


# ── Load provider config dict (used by init and by default_config path) ─────


def _load_and_set_provider(config_data: dict[str, Any]) -> None:
    """Extract sub_provider info, normalize, and set on base globals.

    Mutates ``config_data`` in place (pops ``sub_provider``/``sub_providers``).
    """
    from kimix.base import print_debug

    sub_provider = config_data.pop("sub_provider", None)
    sub_providers = config_data.pop("sub_providers", None)
    _inherit_sub_provider_defaults(config_data, sub_provider, sub_providers)
    normalized_providers = _normalize_sub_providers(sub_provider, sub_providers)
    _pick_main_from_sub_providers(config_data, sub_provider, sub_providers)
    base.set_default_provider(config_data)
    print_debug(f"Provider model: {config_data.get('model', 'None')}")
    if normalized_providers:
        base.set_default_sub_providers(normalized_providers)
        roles = [p.get("role", "sub_agent") for p in normalized_providers]
        print_debug(f"Sub-provider roles loaded: {', '.join(roles)}")


# ── Public init() ──────────────────────────────────────────────────────────


def init(
    config_path: str | Path | None = None,
    config_json: str | None = None,
    *,
    yolo: bool = True,
    think: bool = True,
    skill_dir: list[str] | None = None,
    ralph: int | None = None,
    manually_cot: bool = False,
    colorful_print: bool = True,
    clean: bool = False,
) -> None:
    """Initialize kimix global state. Must be called before ``prompt()`` in non-CLI scripts.

    Args:
        config_path: Path to a JSON config file (provider settings).
        config_json: JSON string with provider settings (alternative to config_path).
        yolo: Enable YOLO mode (auto-approve tool calls).
        think: Enable thinking mode.
        skill_dir: Additional skill directories to load.
        ralph: Ralph mode iteration limit.
        manually_cot: Enable manually CoT mode.
        colorful_print: Enable ANSI-colorful output.
        clean: Delete session cache after quit.
    """
    # Lazy imports to avoid circular dependencies:
    #   session.py imports _create_config from this module,
    #   so we import session internals only inside this function body.
    from kimix.utils.session import close_session
    from kimix.cli_impl import constants
    from kimix.base import print_debug, print_warning
    from kaos.path import KaosPath

    # 1. Dispose existing default session (if any)
    if _globals._default_session is not None:
        sess = _globals._default_session
        _globals._default_session = None
        close_session(sess)

    # 2. Reset / set base globals
    base._colorful_print = colorful_print
    base.set_default_thinking(think)
    base.set_default_yolo(yolo)
    base.set_default_manually_cot(manually_cot)
    base._default_ralph = ralph

    # 3. Clean mode
    constants.CLEAN_MODE = clean
    if clean:
        print_debug("Clean mode ON, delete cache file after quit.")

    # 4. Skill directories
    # Reset to auto-detected defaults (empty first, then auto-load)
    base._default_skill_dirs = []

    # Auto-load from .kimix/skill.json
    _load_skill_json()

    # Add explicit skill dirs
    if skill_dir:
        existing = list(base._default_skill_dirs)
        for sd in skill_dir:
            sd_path = Path(sd)
            if not sd_path.is_absolute():
                sd_path = Path.cwd() / sd_path
            sd_path = sd_path.resolve()
            if sd_path.exists() and sd_path.is_dir():
                existing.append(KaosPath(str(sd_path)))
                print_debug(f"Skill dir added: {str(sd_path)}")
            else:
                print_warning(f"Skill dir not found: {str(sd_path)}")
        base.set_default_skill_dirs(existing)

    # 5. Load provider config
    if config_path is not None:
        config_data = _load_config_file(Path(config_path))
        if config_data:
            _load_and_set_provider(config_data)
    elif config_json is not None:
        try:
            config_data = orjson.loads(config_json)
            if not isinstance(config_data, dict):
                raise ValueError("config_json must parse to a JSON object")
            _load_and_set_provider(config_data)
        except orjson.JSONDecodeError as e:
            from kimix.base import print_error
            print_error(f"Invalid config JSON: {e}")
            sys.exit(1)
        except ValueError as e:
            from kimix.base import print_error
            print_error(str(e))
            sys.exit(1)
    else:
        # Fallback to default_config.json
        default_config_path = Path(__file__).parent.parent / "default_config.json"
        if default_config_path.exists():
            try:
                config_data = orjson.loads(default_config_path.read_text(encoding="utf-8"))
                if isinstance(config_data, dict):
                    _load_and_set_provider(config_data)
            except (orjson.JSONDecodeError, Exception):
                pass

    # 6. Ralph override on provider
    if ralph is not None and base._default_provider is not None:
        if "loop_control" not in base._default_provider:
            base._default_provider["loop_control"] = {}
        base._default_provider["loop_control"]["max_ralph_iterations"] = ralph
        print_debug(f"Ralph mode set to {ralph}.")


# ── Existing _create_config (unchanged) ────────────────────────────────────


def _create_config(provider_dict: dict[str, Any] | None = None) -> tuple[Config, dict[str, Any] | None]:
    from kimi_cli.config import LLMModel, LLMProvider
    from kimix.base import print_debug, print_warning

    provider_dict = provider_dict if provider_dict is not None else base._default_provider
    cfg = Config()

    if provider_dict is None:
        try:
            provider_dict = orjson.loads(
                (Path(__file__).parent.parent / 'default_config.json').read_text(encoding='utf-8', errors='replace'))
            if type(provider_dict) != dict:
                provider_dict = None
        except:
            pass
    if provider_dict is not None:
        model = provider_dict.get('model')
        max_context_size = provider_dict.get('max_context_size')
        capabilities = set(provider_dict.get('capabilities', set()))
        url = provider_dict.get('url')
        provider_type = provider_dict.get("type")
        assert provider_type is not None, "`provider_type` must be provided in config"
        assert max_context_size is not None, "`max_context_size` must be provided in  config"
        assert type(model) == str, "model(str) must be provided in config"
        assert url is not None, "url must be provided in config"

        env: dict | None =  provider_dict.get('env')
        if env is not None:
            for k, v in env.items():
                os.environ[k] = v
        max_context_size = int(max_context_size)
        api_key = provider_dict.get('api_key', None)
        if not api_key:
            api_key = os.environ.get("KIMI_API_KEY")
        if not api_key:
            api_key = os.environ.get("KIMIX_API_KEY")
        if not api_key:
            print_warning(
                'api_key not found. May config in JSON, or set to env `KIMI_API_KEY` or `KIMIX_API_KEY`')
            api_key = ''
        oath_dict = provider_dict.get('oauth')
        oath : OAuthRef | None = None
        if isinstance(oath_dict, dict):
            oath = OAuthRef(key=oath_dict.get('key', ''))
            oath.storage = oath_dict.get('storage', 'file')
            assert isinstance(oath.storage, str), 'oath.storage must be str'
            assert isinstance(oath.key, str), 'oath.key must be str'
        else:
            oath = None
        openai_settings_dict = provider_dict.get('openai_settings')
        openai_settings: OpenAISettings | None = None
        if isinstance(openai_settings_dict, dict):
            openai_settings = OpenAISettings(**openai_settings_dict)
        cfg.provider = LLMProvider(
            type=provider_type,
            # example: "https://api.minimaxi.com/anthropic"
            base_url=url,
            api_key=SecretStr(api_key),
            custom_headers=provider_dict.get('custom_headers'),
            oauth=oath,
            openai_settings=openai_settings,
        )
        cfg.model = LLMModel(
            model=model,
            max_context_size=max_context_size,
            capabilities=capabilities,
        )
        # Set loop control
        loop_control = provider_dict.get('loop_control')
        lc = LoopControl()
        if loop_control and isinstance(loop_control, dict):
            for key, value in loop_control.items():
                if hasattr(lc, key):
                    setattr(lc, key, value)
        if base._default_ralph is not None and 'max_ralph_iterations' not in (loop_control or {}): # override
            lc.max_ralph_iterations = base._default_ralph
        cfg.loop_control = lc
        def set_val(name: str, type_var: type) -> None:
            v = provider_dict.get(name)
            if v is not None:
                setattr(cfg, name, type_var(v))
        set_val('show_thinking_stream', bool)
        # Set notifications
        notifications = provider_dict.get('notifications')
        if notifications and isinstance(notifications, dict):
            nc = NotificationConfig()
            for key, value in notifications.items():
                if hasattr(nc, key):
                    setattr(nc, key, value)
            cfg.notifications = nc
        # Set mcp
        mcp = provider_dict.get('mcp')
        if mcp and isinstance(mcp, dict):
            mc = MCPConfig()
            for key, value in mcp.items():
                if hasattr(mc, key):
                    setattr(mc, key, value)
            cfg.mcp = mc
        # Set LLM override settings
        set_val('max_tokens', int)
        set_val('thinking_effort', str)
        set_val('temperature', float)
        set_val('top_p', float)
        set_val('top_k', int)
        # Set background
        background = provider_dict.get('background')
        if background and isinstance(background, dict):
            bc = BackgroundConfig()
            for key, value in background.items():
                if hasattr(bc, key):
                    setattr(bc, key, value)
            cfg.background = bc
        # Warn about unrecognized keys in provider_dict.
        # recognized_keys must contain every key that is explicitly consumed above
        # this point (including nested-dict keys handled via getattr checks), plus
        # legacy keys that are accepted but ignored (model_name/name).
        # When a new top-level key is added to the config contract, add it here.
        recognized_keys = {
            "model", "max_context_size", "capabilities", "url", "type", "env",
            "api_key", "oauth", "openai_settings", "custom_headers", "loop_control",
            "show_thinking_stream", "notifications", "mcp", "max_tokens",
            "thinking_effort", "temperature", "top_p", "top_k", "background",
            "model_name", "name", "role", 
        }
        unrecognized_keys = [k for k in provider_dict if k not in recognized_keys]
        if unrecognized_keys:
            print_warning(f"Unrecognized keys in provider_dict: {unrecognized_keys}")
    return cfg, provider_dict
