from __future__ import annotations

import copy
from typing import cast

import regex as re

from kosong.utils.typing import JsonType

type JsonDict = dict[str, JsonType]

# JSON Schema keywords that describe a property's shape without (or in
# addition to) a ``type`` keyword. When any of these are present we skip
# the type-filling step so we don't distort the schema's meaning —
# ``not``/``if``/``then``/``else`` are less common but every bit as valid
# as ``anyOf``/``oneOf``/``allOf``.
_TYPE_COMPLETION_SKIP_KEYS = frozenset(
    {
        "$ref",
        "allOf",
        "anyOf",
        "else",
        "if",
        "not",
        "oneOf",
        "then",
    }
)

# Child-schema positions that the Kimi normalizer knows how to walk. This is
# also the source of truth for child-schema keywords that imply the parent
# schema's type. It is not a list of keywords that Moonshot accepts on the
# wire. Entries are ``(key, kind, parent_type)`` where kind is one of
# ``single`` / ``array`` / ``map`` / ``schema-or-array``.
_CHILD_SCHEMA_SLOTS: tuple[tuple[str, str, str | None], ...] = (
    ("$defs", "map", None),
    ("definitions", "map", None),
    ("dependencies", "map", "object"),
    ("dependentSchemas", "map", "object"),
    ("patternProperties", "map", "object"),
    ("properties", "map", "object"),
    ("additionalItems", "single", "array"),
    ("additionalProperties", "single", "object"),
    ("contains", "single", "array"),
    ("contentSchema", "single", "string"),
    ("else", "single", None),
    ("if", "single", None),
    ("not", "single", None),
    ("propertyNames", "single", "object"),
    ("then", "single", None),
    ("unevaluatedItems", "single", "array"),
    ("unevaluatedProperties", "single", "object"),
    ("allOf", "array", None),
    ("anyOf", "array", None),
    ("oneOf", "array", None),
    ("prefixItems", "array", "array"),
    ("items", "schema-or-array", "array"),
)


def _child_schema_keys_for_parent_type(parent_type: str) -> frozenset[str]:
    return frozenset(key for key, _kind, parent in _CHILD_SCHEMA_SLOTS if parent == parent_type)


# Structural keywords that only make sense for a given JSON Schema type.
# Used to infer `type` when enum/const are absent but the node otherwise
# clearly describes an object or array or constrained scalar — setting
# `type: "string"` on such a node would misadvertise the parameter shape
# and cause the model to emit arguments that then fail downstream
# `jsonschema.validate` against the tool's real parameter schema.
_OBJECT_STRUCTURE_KEYS = _child_schema_keys_for_parent_type("object") | frozenset(
    {"dependentRequired", "maxProperties", "minProperties", "required"}
)
_ARRAY_STRUCTURE_KEYS = _child_schema_keys_for_parent_type("array") | frozenset(
    {"maxContains", "maxItems", "minContains", "minItems", "uniqueItems"}
)
_STRING_STRUCTURE_KEYS = _child_schema_keys_for_parent_type("string") | frozenset(
    {"contentEncoding", "contentMediaType", "format", "maxLength", "minLength", "pattern"}
)
_NUMERIC_STRUCTURE_KEYS = frozenset(
    {"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum", "multipleOf"}
)

_JSON_POINTER_INDEX_RE = re.compile(r"0|[1-9]\d*")


def deref_json_schema(schema: JsonDict) -> JsonDict:
    """Expand local `$ref` entries in a JSON Schema by inlining definitions
    from local JSON pointers such as ``$defs`` and draft-7 ``definitions``.

    Sibling keywords next to a ``$ref`` are preserved (JSON Schema 2020-12
    semantics); sibling keys on the local node take precedence over the
    resolved definition. Circular references are detected and left as ``$ref``
    to avoid infinite recursion; in that case the referenced definition bucket
    is preserved so the remaining local ``$ref`` pointers stay resolvable to a
    JSON Schema validator. References that cannot be resolved locally (remote
    URIs, dangling pointers) are left untouched.
    """
    # Work on a deep copy so we never mutate the caller's schema.
    full_schema: JsonDict = copy.deepcopy(schema)
    visited: set[str] = set()

    def resolve_pointer(pointer: str) -> tuple[bool, JsonType]:
        """Resolve a JSON Pointer (e.g. ``#/$defs/User``) inside the schema."""
        if pointer == "#":
            return True, full_schema
        current: JsonType = full_schema
        for raw_part in pointer[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                if part not in current:
                    return False, None
                current = current[part]
            elif isinstance(current, list):
                if not _JSON_POINTER_INDEX_RE.fullmatch(part):
                    return False, None
                index = int(part)
                if index >= len(current):
                    return False, None
                current = current[index]
            else:
                return False, None
        return True, current

    def traverse(node: JsonType) -> JsonType:
        """Recursively traverse every node to inline local references."""
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref == "#" or ref.startswith("#/"):
                    if ref in visited:
                        # Circular reference — keep the $ref as-is to avoid
                        # infinite recursion.
                        return node
                    found, target = resolve_pointer(ref)
                    if found:
                        visited.add(ref)
                        resolved = traverse(target)
                        visited.discard(ref)
                        if isinstance(resolved, dict):
                            # Merge sibling keywords over the resolved
                            # definition; local keys take precedence.
                            merged: JsonDict = dict(resolved)
                            for key, value in node.items():
                                if key == "$ref":
                                    continue
                                merged[key] = traverse(value)
                            return merged
                        return resolved
                # Remote or unresolvable reference — leave the node as-is.
                return node
            return {key: traverse(value) for key, value in node.items()}
        if isinstance(node, list):
            return [traverse(item) for item in node]
        return node

    resolved = cast(JsonDict, traverse(full_schema))

    # Only drop definition buckets when no refs into them remain in the
    # result. Cyclic refs are intentionally preserved by ``traverse`` and
    # still need their definition buckets; dropping them would leave
    # dangling pointers.
    if not _has_unresolved_definition_ref(resolved, "$defs"):
        resolved.pop("$defs", None)
    if not _has_unresolved_definition_ref(resolved, "definitions"):
        resolved.pop("definitions", None)

    return resolved


def _has_unresolved_definition_ref(node: JsonType, bucket_key: str) -> bool:
    """Whether any ``$ref: #/<bucket_key>/...`` pointer remains outside the
    definition bucket itself."""
    if isinstance(node, list):
        return any(_has_unresolved_definition_ref(child, bucket_key) for child in node)
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(f"#/{bucket_key}/"):
            return True
        return any(
            _has_unresolved_definition_ref(value, bucket_key)
            for key, value in node.items()
            if key != bucket_key
        )
    return False


def ensure_property_types(schema: JsonDict) -> JsonDict:
    """Return a deep copy of ``schema`` with an explicit ``type`` on every
    nested property schema.

    The Moonshot (Kimi) API rejects tool parameter schemas where a nested
    property schema omits ``type`` — for example ``{"enum": ["smart", "full"]}``
    with no ``"type": "string"``. JSON Schema itself permits this (the property
    then accepts any value), and providers such as OpenAI and Anthropic accept
    it, but Moonshot's stricter validator returns HTTP 400 with
    ``"At path 'properties.X': type is not defined"``.

    This is a provider-compatibility normalizer, not a complete JSON Schema
    compiler. It walks every well-known child-schema position (``properties``,
    ``items``, ``prefixItems``, ``additionalProperties``, ``patternProperties``,
    ``contains``, ``if``/``then``/``else``/``not``, ``allOf``/``anyOf``/
    ``oneOf``, ``$defs``/``definitions``, ...) and:

    - when ``type`` is missing, infers it from ``enum`` / ``const`` values,
      then from structural keywords, falling back to ``"string"``;
    - when an explicit ``type`` contradicts the ``enum`` / ``const`` values
      (a known Xcode MCP generator bug), repairs the type and drops structure
      keywords that no longer apply.

    Nodes that use combinators (``anyOf``/``oneOf``/``allOf``/``$ref`` etc.)
    are left alone since they legitimately declare their shape without
    ``type``. The outer schema object itself is treated as a container and
    never normalized — only the property schemas it contains are.
    """
    result: JsonDict = copy.deepcopy(schema)
    _recurse_schema(result)
    return result


def _recurse_schema(node: JsonType) -> None:
    """Walk into child-schema positions under ``node`` and normalize them.

    ``node`` itself is treated as a container and is not normalized.
    """
    if not isinstance(node, dict):
        return
    for key, kind, _parent_type in _CHILD_SCHEMA_SLOTS:
        value = node.get(key)
        if kind == "single":
            if isinstance(value, dict):
                _normalize_property(value)
        elif kind == "array":
            if isinstance(value, list):
                for item in value:
                    _normalize_property(item)
        elif kind == "map":
            if isinstance(value, dict):
                for item in value.values():
                    _normalize_property(item)
        else:  # schema-or-array
            if isinstance(value, dict):
                _normalize_property(value)
            elif isinstance(value, list):
                for item in value:
                    _normalize_property(item)


def _normalize_property(node: JsonType) -> None:
    """Ensure ``node`` (a property schema) declares a ``type``, then recurse."""
    if not isinstance(node, dict):
        return

    has_skip_keys = any(key in node for key in _TYPE_COMPLETION_SKIP_KEYS)
    if "type" not in node and not has_skip_keys:
        enum_values = node.get("enum")
        if isinstance(enum_values, list) and enum_values:
            node["type"] = _infer_type_from_values(enum_values)
        elif "const" in node:
            node["type"] = _infer_type_from_values([node["const"]])
        else:
            node["type"] = _infer_type_from_structure(node)
    elif not has_skip_keys and isinstance(node.get("type"), str):
        # Some MCP servers emit schemas where a $ref merge or a generator bug
        # leaves an explicit type that contradicts the enum/const values (e.g.
        # ``type: "object"`` alongside string enum values). Moonshot rejects
        # these as invalid, so repair the type when it disagrees with the
        # values.
        #
        # Known trigger: Xcode MCP (xcrun mcpbridge) starting with
        # Version 26.5 (17F42) generates this bug for String-backed Swift
        # enums.
        enum_values = node.get("enum")
        values: list[JsonType] | None = None
        if isinstance(enum_values, list) and enum_values:
            values = enum_values
        elif "const" in node:
            values = [node["const"]]
        if values is not None:
            inferred = _try_infer_single_type(values)
            if inferred is not None and node["type"] != inferred:
                node["type"] = inferred
                _remove_irrelevant_structure_keys(node, inferred)

    _recurse_schema(node)


def _remove_irrelevant_structure_keys(node: JsonDict, new_type: str) -> None:
    """Drop object/array structure keywords that no longer apply after a
    type repair changed the node's ``type``."""
    if new_type != "object":
        for key in _OBJECT_STRUCTURE_KEYS:
            node.pop(key, None)
    if new_type != "array":
        for key in _ARRAY_STRUCTURE_KEYS:
            node.pop(key, None)


def _infer_type_from_structure(node: JsonDict) -> str:
    """Infer a JSON Schema ``type`` from structural keywords in ``node``.

    Used as the fallback when no ``enum`` / ``const`` is present. Defaults
    to ``"string"`` only when the node carries no structural hints at all.
    """
    if any(key in node for key in _OBJECT_STRUCTURE_KEYS):
        return "object"
    if any(key in node for key in _ARRAY_STRUCTURE_KEYS):
        return "array"
    if any(key in node for key in _STRING_STRUCTURE_KEYS):
        return "string"
    if any(key in node for key in _NUMERIC_STRUCTURE_KEYS):
        return "number"
    return "string"


def _classify_value(value: JsonType) -> str | None:
    """Classify a JSON value into its JSON Schema type string."""
    # ``bool`` is a subclass of ``int`` in Python, but JSON Schema treats
    # booleans as a distinct type, so classify it before the numeric checks.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):  # pyright: ignore[reportUnnecessaryIsInstance]
        return "array"
    return None


def _infer_type_from_values(values: list[JsonType]) -> str:
    """Infer a JSON Schema ``type`` string from a list of concrete values.

    Classify each value, then:
    - single type → return it
    - ``{integer, number}`` → ``"number"`` (integer is a subset of number)
    - anything else mixed (e.g. ``[True, 1]`` or ``["a", 1]``) → fall back to
      ``"string"``, which Moonshot tolerates without cross-checking enum
      values against the declared type
    """
    inferred: set[str] = set()
    for value in values:
        kind = _classify_value(value)
        if kind is None:
            # Unreachable for well-formed JSON values, but defensive for
            # non-JSON inputs (e.g. if a caller passes a tuple or custom
            # object): fall back to the safe string type.
            return "string"
        inferred.add(kind)

    if len(inferred) == 1:
        return next(iter(inferred))
    if inferred == {"integer", "number"}:
        return "number"
    return "string"


def _try_infer_single_type(values: list[JsonType]) -> str | None:
    """Like :func:`_infer_type_from_values` but strict: returns ``None`` when
    the values have mixed or unclassifiable types instead of falling back to
    ``"string"``. Used by the type-repair path, which must leave ambiguous
    schemas untouched rather than guess."""
    inferred: set[str] = set()
    for value in values:
        kind = _classify_value(value)
        if kind is None:
            return None
        inferred.add(kind)
    if "number" in inferred:
        # integer is a subset of number
        inferred.discard("integer")
    if len(inferred) == 1:
        return next(iter(inferred))
    return None
