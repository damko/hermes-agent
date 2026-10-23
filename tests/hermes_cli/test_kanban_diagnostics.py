"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48








# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True


def test_dispatcher_host_conflict_fires_on_cross_host_reclaim():
    """A reclaim whose claim belonged to another host is a two-dispatcher
    config conflict, not a repeated agent failure (2026-10-21: bee:2224)."""
    now = int(time.time())
    events = [
        _event("claimed", ts=now - 1200, lock="bee:2224"),
        _event("reclaimed", ts=now - 600, stale_lock="bee:2224",
               host_local=False, cross_host=True),
    ]
    diags = kd.compute_task_diagnostics(_task(status="blocked"), events, [], now=now)
    found = [d for d in diags if d.kind == "dispatcher_host_conflict"]
    assert found, [d.kind for d in diags]
    d = found[0]
    assert d.severity == "error"
    assert d.data["conflict"] == "two_dispatcher_hosts"
    assert d.data["stale_lock"] == "bee:2224"
    assert "bee" in d.title
    assert d.actions and d.actions[-1].kind == "unblock"


def test_dispatcher_host_conflict_fires_on_foreign_deferral():
    """Holding a live foreign claim is recorded as ``reclaim_deferred`` with
    ``host_local: False`` — also proof of a second dispatcher host."""
    now = int(time.time())
    events = [
        _event("reclaim_deferred", ts=now - 100, claim_lock="bee:20749",
               host_local=False, reason="foreign_host_claim_live"),
    ]
    diags = kd.compute_task_diagnostics(
        _task(status="running", claim_lock="bee:20749"), events, [], now=now)
    assert [d.kind for d in diags if d.kind == "dispatcher_host_conflict"]


def test_dispatcher_host_conflict_fires_on_two_claim_hosts_without_reclaim():
    """Two distinct claimer hosts on one task prove two dispatchers even when
    neither side has reclaimed yet. At most one of them can be this host."""
    now = int(time.time())
    events = [
        _event("claimed", ts=now - 3000, lock="bee:100"),
        _event("claimed", ts=now - 1000, lock="u-dam-dev:200"),
    ]
    diags = kd.compute_task_diagnostics(
        _task(status="running", claim_lock="u-dam-dev:200"), events, [], now=now)
    found = [d for d in diags if d.kind == "dispatcher_host_conflict"]
    assert found
    assert found[0].severity == "warning"
    assert found[0].data["claim_hosts"] == ["bee:", "u-dam-dev:"]


def test_dispatcher_host_conflict_silent_on_one_host():
    """Normal single-dispatcher operation must never raise this diagnostic."""
    now = int(time.time())
    events = [
        _event("claimed", ts=now - 600, lock="bee:100"),
        _event("reclaimed", ts=now - 300, stale_lock="bee:100",
               host_local=True, cross_host=False),
        _event("claimed", ts=now - 200, lock="bee:300"),
    ]
    diags = kd.compute_task_diagnostics(
        _task(status="running", claim_lock="bee:300"), events, [], now=now)
    assert [d for d in diags if d.kind == "dispatcher_host_conflict"] == []


def test_dispatcher_host_conflict_ignores_events_outside_window():
    """A conflict older than ``host_conflict_window_seconds`` is not current."""
    now = int(time.time())
    events = [
        _event("reclaimed", ts=now - 3 * 86400, stale_lock="bee:2224",
               host_local=False, cross_host=True),
    ]
    diags = kd.compute_task_diagnostics(_task(status="ready"), events, [], now=now)
    assert [d for d in diags if d.kind == "dispatcher_host_conflict"] == []
