import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import orjson
import yaml


def check_json_text(text: str, json_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of a JSON string.

    Args:
        text: JSON text to validate.
        json_callback: Optional callback invoked with the parsed object.

    Returns:
        None if the JSON is valid, error message string otherwise.
    """
    try:
        js = orjson.loads(text)
        if json_callback is not None:
            json_callback(js)
        return None
    except orjson.JSONDecodeError as exc:
        return f"JSON decode error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except Exception as exc:
        return f"failed to validate JSON file: {str(exc)}"

def check_xml_text(text: str, xml_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of an XML string.

    Args:
        text: XML text to validate.
        xml_callback: Optional callback invoked with the parsed tree.

    Returns:
        None if the XML is valid, error message string otherwise.
    """
    try:
        tree = ET.fromstring(text)
        if xml_callback is not None:
            xml_callback(tree)
        return None
    except ET.ParseError as exc:
        return f"XML parse error: {str(exc)}"
    except Exception as exc:
        return f"failed to validate XML file: {str(exc)}"

def check_yaml_text(text: str, yaml_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of a YAML string.

    Args:
        text: YAML text to validate.
        yaml_callback: Optional callback invoked with the parsed object.

    Returns:
        None if the YAML is valid, error message string otherwise.
    """
    try:
        data = yaml.safe_load(text)
        if yaml_callback is not None:
            yaml_callback(data)
        return None
    except yaml.YAMLError as exc:
        return f"YAML parse error: {str(exc)}"
    except Exception as exc:
        return f"failed to validate YAML file: {str(exc)}"

def check_toml_text(text: str, toml_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of a TOML string.

    Args:
        text: TOML text to validate.
        toml_callback: Optional callback invoked with the parsed object.

    Returns:
        None if the TOML is valid, error message string otherwise.
    """
    try:
        data = tomllib.loads(text)
        if toml_callback is not None:
            toml_callback(data)
        return None
    except tomllib.TOMLDecodeError as exc:
        return f"TOML parse error: {str(exc)}"
    except Exception as exc:
        return f"failed to validate TOML file: {str(exc)}"



# ---------------------------------------------------------------------------
# File-path and legacy string variants (canonical home since P9; previously
# duplicated in ``kimix.tools.check_fmt``, which is now a shim).
# ---------------------------------------------------------------------------


def check_json(file_path: str, json_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of a JSON file.

    Args:
        file_path: Path to the JSON file to validate.

    Returns:
        None if the JSON file is valid, error message string otherwise.
    """
    try:
        js = None
        with open(file_path, "r", encoding="utf-8") as f:
            js = orjson.loads(f.read())
        if json_callback:
            json_callback(js)
        return None

    except orjson.JSONDecodeError as exc:
        return f"JSON decode error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except Exception as exc:
        return f"Failed to validate JSON file: {str(exc)}"


def check_xml(file_path: str, xml_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of an XML file.

    Args:
        file_path: Path to the XML file to validate.

    Returns:
        None if the XML file is valid, error message string otherwise.
    """
    try:
        tree = ET.parse(file_path)
        if xml_callback:
            xml_callback(tree)
        return None

    except ET.ParseError as exc:
        return f"XML parse error: {str(exc)}"
    except Exception as exc:
        return f"Failed to validate XML file: {str(exc)}"


def check_json_str(content: str, json_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of a JSON string.

    Args:
        content: JSON string content to validate.

    Returns:
        None if the JSON string is valid, error message string otherwise.
    """
    try:
        js = orjson.loads(content)
        if json_callback:
            json_callback(js)
        return None

    except orjson.JSONDecodeError as exc:
        return f"JSON decode error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    except Exception as exc:
        return f"Failed to validate JSON content: {str(exc)}"


def check_xml_str(content: str, xml_callback: Callable[[Any], None] | None = None) -> str | None:
    """Validate the format of an XML string.

    Args:
        content: XML string content to validate.

    Returns:
        None if the XML string is valid, error message string otherwise.
    """
    try:
        root = ET.fromstring(content)
        if xml_callback:
            xml_callback(root)
        return None

    except ET.ParseError as exc:
        return f"XML parse error: {str(exc)}"
    except Exception as exc:
        return f"Failed to validate XML content: {str(exc)}"
