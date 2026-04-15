"""Tests for SessionStore — load/save/list/delete/trace roundtrips."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from umcp.session import RunSession, SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(storage_path=tmp_path)


# ------------------------------------------------------------------ #
# RunSession
# ------------------------------------------------------------------ #

def test_run_session_add_messages():
    s = RunSession()
    s.add_system("You are helpful.")
    s.add_user("Hello")
    s.add_assistant("Hi there!", tool_calls=None)
    assert len(s.messages) == 3
    assert s.messages[0]["role"] == "system"
    assert s.messages[1]["role"] == "user"
    assert s.messages[2]["role"] == "assistant"


def test_run_session_tool_result_native():
    s = RunSession()
    s.add_tool_result("22C sunny", native_mode=True)
    assert s.messages[-1]["role"] == "tool"


def test_run_session_tool_result_fallback():
    s = RunSession()
    s.add_tool_result("22C sunny", native_mode=False)
    assert s.messages[-1]["role"] == "user"


def test_run_session_inject_notice():
    s = RunSession()
    s.inject_notice("Server X unavailable.")
    assert s.messages[-1]["content"] == "Server X unavailable."


def test_run_session_snapshot_is_copy():
    s = RunSession()
    s.add_user("hi")
    snap = s.snapshot()
    snap.append({"role": "extra"})
    assert len(s.messages) == 1  # original unchanged


def test_run_session_add_assistant_with_tool_calls():
    s = RunSession()
    calls = [{"function": {"name": "w.forecast", "arguments": {}}}]
    s.add_assistant("", tool_calls=calls)
    assert "tool_calls" in s.messages[-1]


# ------------------------------------------------------------------ #
# SessionStore — save/load roundtrip
# ------------------------------------------------------------------ #

def test_session_store_save_and_load(store):
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    store.save("test-session", messages)
    loaded = store.load("test-session")
    assert loaded == messages


def test_session_store_load_missing_returns_empty(store):
    assert store.load("nonexistent") == []


def test_session_store_list_sessions(store):
    store.save("session-a", [])
    store.save("session-b", [])
    sessions = store.list_sessions()
    assert "session-a" in sessions
    assert "session-b" in sessions


def test_session_store_list_excludes_trace_files(store):
    """_trace.json files should not appear in session list."""
    store.save("my-session", [])
    store.save_trace("my-session", [{"trace_id": "abc"}])
    sessions = store.list_sessions()
    assert "my-session" in sessions
    assert "my-session_trace" not in sessions


def test_session_store_delete_existing(store):
    store.save("deleteme", [{"role": "user", "content": "x"}])
    deleted = store.delete("deleteme")
    assert deleted is True
    assert store.load("deleteme") == []


def test_session_store_delete_nonexistent(store):
    deleted = store.delete("no-such-session")
    assert deleted is False


def test_session_store_sanitizes_session_id(store):
    """Session IDs with special chars should be stored safely."""
    store.save("my session/with:special", [{"role": "user", "content": "x"}])
    loaded = store.load("my session/with:special")
    assert len(loaded) == 1


# ------------------------------------------------------------------ #
# SessionStore — trace persistence
# ------------------------------------------------------------------ #

def test_session_store_save_and_load_trace(store):
    trace_entries = [
        {"trace_id": "abc123", "tool": "weather.forecast", "status": "success"},
        {"trace_id": "def456", "tool": "db.insert", "status": "error"},
    ]
    store.save_trace("my-session", trace_entries)
    loaded = store.load_trace("my-session")
    assert loaded == trace_entries


def test_session_store_load_trace_missing_returns_empty(store):
    assert store.load_trace("ghost-session") == []


# ------------------------------------------------------------------ #
# Multi-turn history accumulation (integration-style)
# ------------------------------------------------------------------ #

def test_session_store_history_accumulates_across_turns(store):
    """Simulate two chat turns writing history."""
    # Turn 1
    messages_t1 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "It's sunny!"},
    ]
    store.save("convo", messages_t1)

    # Turn 2: load + append
    history = store.load("convo")
    history.append({"role": "user", "content": "And tomorrow?"})
    history.append({"role": "assistant", "content": "It will rain."})
    store.save("convo", history)

    final = store.load("convo")
    assert len(final) == 5
    assert final[-1]["content"] == "It will rain."
