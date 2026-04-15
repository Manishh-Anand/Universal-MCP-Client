"""Tests for typed retry strategy."""
import pytest
from umcp.retry import (
    RetryDecision, RetryReason, RetryState,
    decide_retry, build_correction_message,
)


def test_tool_error_retries_then_aborts():
    assert decide_retry(RetryReason.TOOL_ERROR, 0, 2) == RetryDecision.RETRY_TOOL
    assert decide_retry(RetryReason.TOOL_ERROR, 1, 2) == RetryDecision.RETRY_TOOL
    assert decide_retry(RetryReason.TOOL_ERROR, 2, 2) == RetryDecision.ABORT


def test_hallucination_always_reprompts():
    assert decide_retry(RetryReason.HALLUCINATION, 0, 2) == RetryDecision.REPROMPT
    assert decide_retry(RetryReason.HALLUCINATION, 1, 2) == RetryDecision.REPROMPT
    assert decide_retry(RetryReason.HALLUCINATION, 2, 2) == RetryDecision.ABORT


def test_schema_failure_reprompts():
    assert decide_retry(RetryReason.SCHEMA_FAILURE, 0, 2) == RetryDecision.REPROMPT


def test_invalid_json_first_retry_then_reprompt():
    assert decide_retry(RetryReason.INVALID_JSON, 0, 2) == RetryDecision.RETRY_TOOL
    assert decide_retry(RetryReason.INVALID_JSON, 1, 2) == RetryDecision.REPROMPT


def test_retry_state_tracks_counts():
    state = RetryState(max_per_tool=2, max_total=10)
    assert state.count_for("tool_a") == 0
    state.increment("tool_a")
    assert state.count_for("tool_a") == 1
    state.increment("tool_a")
    assert state.budget_exceeded("tool_a") is True


def test_retry_state_total_budget():
    state = RetryState(max_per_tool=5, max_total=3)
    state.increment("a")
    state.increment("b")
    state.increment("c")
    assert state.budget_exceeded("d") is True


def test_hallucination_correction_message():
    msg = build_correction_message(
        RetryReason.HALLUCINATION, "fake_tool", ["weather.get_forecast", "db.query"]
    )
    assert "fake_tool" in msg
    assert "weather.get_forecast" in msg
    assert "does not exist" in msg


def test_schema_correction_message():
    msg = build_correction_message(
        RetryReason.SCHEMA_FAILURE, "weather.get_forecast", [], "city: field required"
    )
    assert "weather.get_forecast" in msg
    assert "city" in msg
