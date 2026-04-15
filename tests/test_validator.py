"""Tests for tool input validation and auto-coercion."""
import pytest
from umcp.validator import validate_and_coerce, _coerce_value


def test_valid_args_pass():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    valid, errors, coerced = validate_and_coerce(schema, {"city": "Tokyo"})
    assert valid is True
    assert errors == []
    assert coerced["city"] == "Tokyo"


def test_missing_required_field():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    valid, errors, _ = validate_and_coerce(schema, {})
    assert valid is False
    assert any("city" in e for e in errors)


def test_string_to_int_coercion():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    valid, errors, coerced = validate_and_coerce(schema, {"count": "42"})
    assert valid is True
    assert coerced["count"] == 42


def test_string_to_bool_coercion():
    schema = {
        "type": "object",
        "properties": {"flag": {"type": "boolean"}},
    }
    valid, errors, coerced = validate_and_coerce(schema, {"flag": "true"})
    assert valid is True
    assert coerced["flag"] is True

    valid2, _, coerced2 = validate_and_coerce(schema, {"flag": "false"})
    assert valid2 is True
    assert coerced2["flag"] is False


def test_float_to_int_coercion():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    }
    valid, _, coerced = validate_and_coerce(schema, {"n": 3.0})
    assert valid is True
    assert coerced["n"] == 3


def test_wrong_type_not_coercible():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    # "hello" cannot be coerced to int
    valid, errors, _ = validate_and_coerce(schema, {"count": "hello"})
    assert valid is False


def test_empty_schema_accepts_anything():
    schema = {"type": "object", "properties": {}}
    valid, errors, coerced = validate_and_coerce(schema, {"anything": "goes"})
    assert valid is True


def test_coerce_value_string_to_int():
    assert _coerce_value("42", "integer") == 42


def test_coerce_value_same_type_returns_none():
    assert _coerce_value(42, "integer") is None  # already correct, no coerce needed
