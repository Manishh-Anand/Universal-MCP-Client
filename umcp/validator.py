"""Tool input validation and auto-coercion against JSON Schema."""
from __future__ import annotations

from typing import Any

import jsonschema
from jsonschema import Draft7Validator


def validate_and_coerce(
    schema: dict[str, Any],
    arguments: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate and auto-coerce arguments against a JSON Schema.

    Returns:
        (valid, errors, coerced_arguments)

    Steps:
        1. Try auto-coercion of common type mismatches (string→int, string→bool, etc.)
        2. Validate coerced arguments against schema.
        3. Return (True, [], coerced) on success or (False, [errors], original) on failure.
    """
    coerced = _coerce(arguments, schema)
    errors = _validate(schema, coerced)
    if not errors:
        return True, [], coerced
    return False, errors, arguments


def _validate(schema: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """Run JSON Schema validation. Returns list of error messages (empty = valid)."""
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    return [_format_error(e) for e in errors]


def _format_error(error: jsonschema.ValidationError) -> str:
    path = ".".join(str(p) for p in error.path) if error.path else "(root)"
    return f"{path}: {error.message}"


def _coerce(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Auto-coerce argument values to match schema types where safely possible."""
    props = schema.get("properties", {})
    if not props:
        return arguments

    result = dict(arguments)
    for key, value in arguments.items():
        prop_schema = props.get(key)
        if prop_schema is None:
            continue
        expected_type = prop_schema.get("type")
        if expected_type is None:
            continue
        coerced = _coerce_value(value, expected_type)
        if coerced is not None:
            result[key] = coerced
    return result


def _coerce_value(value: Any, expected_type: str) -> Any | None:
    """Try to coerce a single value to the expected JSON Schema type.

    Returns the coerced value if successful, None if no coercion needed or possible.
    """
    actual_type = _json_type(value)
    if actual_type == expected_type:
        return None  # already correct

    try:
        if expected_type == "integer" and isinstance(value, (str, float)):
            return int(value)
        if expected_type == "number" and isinstance(value, (str, int)):
            return float(value)
        if expected_type == "boolean" and isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        if expected_type == "string" and not isinstance(value, str):
            return str(value)
        if expected_type == "array" and isinstance(value, str):
            import json
            return json.loads(value)
        if expected_type == "object" and isinstance(value, str):
            import json
            return json.loads(value)
    except (ValueError, TypeError):
        pass
    return None


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "unknown"
