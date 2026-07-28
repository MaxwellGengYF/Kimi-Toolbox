from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import orjson

from kimix.ui.printing import print_info, print_success, print_warning
deepseek_default_config = '''
{
    "model": "deepseek-v4-pro",
    "max_context_size": 1048576,
    "capabilities": ["thinking"],
    "url": "https://api.deepseek.com/",
    "type": "openai_legacy",
    "max_tokens": 384000,
    "thinking_effort": "max"
}
'''
minimax_default_config = '''
{
    "model": "minimax-m2.7",
    "max_context_size": 204800,
    "capabilities": ["thinking"],
    "url": "https://api.minimaxi.com/anthropic",
    "type": "anthropic",
    "max_tokens": 128000,
    "thinking_effort": "max"
}
'''
# Kimi coding models: both `kimi-for-coding` and `kimi-for-coding-highspeed`
# are supported. The default below follows the same shape as
# `src/kimix/default_config.json`.
kimi_default_config = '''
{
    "model": "kimi-for-coding",
    "max_context_size": 1048576,
    "capabilities": ["thinking", "image_in"],
    "url": "https://api.kimi.com/coding/v1",
    "type": "kimi",
    "max_tokens": 131072,
    "show_thinking_stream": true,
    "thinking_effort": "max"
}
'''
default_config = kimi_default_config

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "default_config.json"

_CONTEXT_SIZE_OPTIONS: dict[str, int] = {
    "128k": 131072,
    "200k": 204800,
    "256k": 262144,
    "512k": 524288,
    "1M": 1048576,
}

_VALID_TYPES = (
    "kimi",
    "openai_legacy",
    "openai_responses",
    "anthropic",
    "google_genai",
    "gemini",
    "vertexai",
)

_VALID_THINKING_EFFORTS = ('off', 'low', 'medium', 'high', 'xhigh', 'max')

_VALID_CAPABILITIES = ("thinking", "always_thinking", "image_in", "video_in")



def _load_default_config() -> dict[str, Any]:
    if _DEFAULT_CONFIG_PATH.exists():
        try:
            with open(_DEFAULT_CONFIG_PATH, "rb") as f:
                return orjson.loads(f.read())
        except:
            pass
    return orjson.loads(default_config)


def _save_config(config: dict[str, Any]) -> None:
    with open(_DEFAULT_CONFIG_PATH, "wb") as f:
        f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2))


def _ask(prompt: str, default: str) -> str:
    print_info(f"{prompt} [{default}]: ", end="")
    value = input().strip()
    return value if value else default


def _ask_template() -> str:
    print_info("Select provider template ('kimi', 'deepseek' or 'minimax') [kimi]: ", end="")
    choice = input().strip().lower()
    if choice == 'deepseek':
        return deepseek_default_config
    if choice == 'minimax':
        return minimax_default_config
    return kimi_default_config


def _ask_model_name(default: str = "kimi-for-coding") -> str:
    """Prompt for the model name.

    Both `kimi-for-coding` and `kimi-for-coding-highspeed` are supported.
    """
    return _ask("Enter model name", default)


def _ask_model_type(default: str = "kimi") -> str:
    options_str = ", ".join(_VALID_TYPES)
    while True:
        value = _ask(f"Enter model type ({options_str})", default)
        if value in _VALID_TYPES:
            return value
        print_warning(f"Invalid type '{value}', please choose from: {options_str}")


def _ask_api_key() -> str:
    skip = False
    if os.environ.get("KIMI_API_KEY"):
        print_info("API key already set in environment variable KIMI_API_KEY.")
        skip = True
    if os.environ.get("KIMIX_API_KEY"):
        print_info("API key already set in environment variable KIMIX_API_KEY.")
        skip = True
    print_info("Enter API key (usually required, or set KIMI_API_KEY/KIMIX_API_KEY env var): ", end="")
    value = input().strip()
    if not skip and not value:
        print_warning("API key is usually required. Press Enter again to skip.")
        print_info("Enter API key: ", end="")
        value = input().strip()
    return value


def _ask_context_size(config: dict[str, Any] | None = None) -> int:
    default = "1M"
    if config is not None:
        max_ctx = config.get("max_context_size")
        if max_ctx is not None:
            for k, v in _CONTEXT_SIZE_OPTIONS.items():
                if v == max_ctx:
                    default = k
                    break
    options_str = ", ".join(_CONTEXT_SIZE_OPTIONS.keys())
    while True:
        value = _ask(f"Enter model context size ({options_str} or a number)", default)
        if value in _CONTEXT_SIZE_OPTIONS:
            return _CONTEXT_SIZE_OPTIONS[value]
        try:
            num = int(value)
        except ValueError:
            print_warning(f"Invalid size '{value}', please choose from: {options_str} or enter a specific number")
            continue
        if num <= 1:
            print_warning(f"Context size must be larger than 1, got {num}")
            continue
        num = max(num, 1.)
        return num


def _ask_thinking_effort(default: str = "max") -> str:
    options_str = ", ".join(_VALID_THINKING_EFFORTS)
    while True:
        value = _ask(f"Enter thinking effort ({options_str})", default)
        if value in _VALID_THINKING_EFFORTS:
            return value
        print_warning(f"Invalid effort '{value}', please choose from: {options_str}")


def _ask_capabilities(default: tuple[str, ...] = ("thinking", "image_in")) -> list[str]:
    options_str = ", ".join(_VALID_CAPABILITIES)
    prompt = f"Enter capabilities ({options_str}), multiple allowed, 'none' for empty"
    default_str = ", ".join(default)
    while True:
        value = _ask(prompt, default_str).strip()
        if not value:
            return list(default)
        if value.lower() == "none":
            return []
        parts = [p.strip() for p in value.replace(",", " ").split()]
        invalid = [p for p in parts if p not in _VALID_CAPABILITIES]
        if invalid:
            print_warning(f"Invalid capabilities: {', '.join(invalid)}, please choose from: {options_str}")
            continue
        return parts


def _ask_url(default: str = "https://api.kimi.com/coding/v1") -> str:
    return _ask("Enter model URL", default)


def _ask_max_token(context_size: int, reserved: int, default: int) -> int:
    max_allowed = context_size - reserved
    prompt = f"Enter max tokens (max {max_allowed})"
    while True:
        value = _ask(prompt, str(default)).strip()
        if not value:
            return default
        try:
            num = int(value)
        except ValueError:
            print_warning(f"Invalid number '{value}', using default {default}")
            return default
        if num <= 0 or num > max_allowed:
            print_warning(f"Value {num} out of range, using default {default}")
            return default
        return num


def _ask_sub_provider(defaults: dict[str, Any] | None = None) -> dict[str, Any] | None:
    print_info("Configure a sub-agent provider? (y/n)")
    v = input().strip().lower()
    if v != 'y':
        return None

    def _default(key: str, fallback: Any) -> Any:
        if defaults is None:
            return fallback
        return defaults.get(key, fallback)

    _CONTEXT_SIZE_BY_INT = {v: k for k, v in _CONTEXT_SIZE_OPTIONS.items()}

    print_info("--- Sub-provider configuration ---")
    sub: dict[str, Any] = {}

    model = _ask_model_name(_default("model", "kimi-for-coding"))
    sub["model"] = model

    model_type = _ask_model_type(_default("type", "kimi"))
    sub["type"] = model_type

    url = _ask_url(_default("url", "https://api.kimi.com/coding/v1"))
    sub["url"] = url

    api_key = _ask_api_key()
    sub["api_key"] = api_key

    context_size = _ask_context_size(defaults)
    sub["max_context_size"] = context_size

    thinking = _ask_thinking_effort(_default("thinking_effort", "low"))
    sub["thinking_effort"] = thinking

    caps = _default("capabilities", ("thinking",))
    if isinstance(caps, list):
        caps = tuple(caps)
    elif isinstance(caps, str):
        caps = (caps,)
    caps = _ask_capabilities(caps)
    if "always_thinking" in caps and "thinking" in caps:
        caps.remove("thinking")
    sub["capabilities"] = caps

    reserved = 50000
    max_tokens = _ask_max_token(context_size, reserved, _default("max_tokens", 128000))
    sub["max_tokens"] = max_tokens

    # Ensure loop_control for sub-provider disables ralph
    sub["loop_control"] = {
        "max_ralph_iterations": 0,
    }


    print_success("Sub-provider configuration complete.")
    return sub


def init(initialize: bool = True) -> None:
    if not initialize:
        v = input('default config not found, initialize? you can use /init any time. (y/n)').strip().lower() 
        initialize = v == 'y' or not v
    template = default_config
    if initialize:
        template = _ask_template()
    config = _load_default_config()
    defaults = orjson.loads(template)
    # Merge loaded config over defaults so missing keys are filled in
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict) and key in config and isinstance(config[key], dict):
            for sub_key, sub_value in value.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_value
    try:
        if initialize:
            model = _ask_model_name(config.get("model", "kimi-for-coding"))
            config["model"] = model

            model_type = _ask_model_type(config.get("type", "kimi"))
            config["type"] = model_type

            api_key = _ask_api_key()
            config["api_key"] = api_key

            context_size = _ask_context_size(config)
            config["max_context_size"] = context_size

            reserved = config.get("loop_control", {}).get("reserved_context_size", 50000)
            max_tokens = _ask_max_token(context_size, reserved, config.get('max_tokens', 128000))
            config["max_tokens"] = max_tokens

            thinking = _ask_thinking_effort(config.get("thinking_effort", "low"))
            config["thinking_effort"] = thinking

            caps = config.get("capabilities", ["thinking", "image_in"])
            if isinstance(caps, str):
                caps = (caps,)
            capabilities = _ask_capabilities(tuple(caps))
            if "always_thinking" in capabilities and "thinking" in capabilities:
                capabilities.remove("thinking")
            config["capabilities"] = capabilities

            url = _ask_url(config.get("url", "https://api.kimi.com/coding/v1"))
            config["url"] = url

            sub_provider = _ask_sub_provider(config.get("sub_provider"))
            if sub_provider is not None:
                config["sub_provider"] = sub_provider
            elif "sub_provider" in config:
                del config["sub_provider"]
    except KeyboardInterrupt:
        print_warning('keyboard interruped.')
        return
    _save_config(config)
    if initialize:
        print_success(f"Configuration saved successfully to {_DEFAULT_CONFIG_PATH}.")
    if sys.platform == "win32":
        os.startfile(str(_DEFAULT_CONFIG_PATH))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(_DEFAULT_CONFIG_PATH)])
    else:
        subprocess.run(["xdg-open", str(_DEFAULT_CONFIG_PATH)])
