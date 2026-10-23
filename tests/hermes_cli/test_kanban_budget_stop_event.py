"""A stop's structured cause must survive into ``task_events`` (card t_99b457d1).

``turn_finalizer._record_kanban_budget_exhausted`` passed ``budget_used`` / ``budget_max``
through ``_record_task_failure(event_payload_extra=...)``, and the function dropped them on
the way to ``task_events`` (row check on card ``t_0be97215``: the event carried ``error``,
``failures``, ``retry_status`` only). The operator notification then had no structured
cause to read and printed ``max_runtime=0s`` for a stop that used no runtime cap at all.

These tests pin the event PAYLOAD — the surface the formatter reads — not the prose.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

BUDGET_ERROR = (
    "Iteration budget exhausted (150/150) — task could not complete within the allowed iterations"
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _stop(conn: sqlite3.Connection, tid: str, extra: dict | None) -> bool:
    """The exact call ``_record_kanban_budget_exhausted`` makes."""
    kwargs = {} if extra is None else {"event_payload_extra": extra}
    return kbd._record_task_failure(
        conn,
        tid,
        error=BUDGET_ERROR,
        outcome="timed_out",
        release_claim=True,
        end_run=True,
        **kwargs,
    )


def _stop_event(conn: sqlite3.Connection, tid: str) -> kb.Event:
    return [e for e in kb.list_events(conn, tid) if e.kind == "timed_out"][-1]


def test_budget_stop_event_carries_the_structured_cause(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="budget stop")
        kb.claim_task(conn, tid)

        assert _stop(
            conn, tid,
            {"stop_reason": "budget_exhausted", "budget_used": 150, "budget_max": 150},
        ) is False  # below the breaker threshold: a plain retry, not a block

        payload = _stop_event(conn, tid).payload
        assert payload["stop_reason"] == "budget_exhausted"
        assert payload["budget_used"] == 150
        assert payload["budget_max"] == 150
        # No runtime cap was involved: the event must not invent one.
        assert "limit_seconds" not in payload
        # The fields the event already carried are still there.
        assert payload["failures"] == 1
        assert payload["retry_status"] == "ready"


def test_run_outcome_stays_timed_out_by_design(kanban_home: Path) -> None:
    """``task_runs.outcome`` keeps ``timed_out``; only the EVENT gains ``stop_reason``.

    The dispatcher, ``_retry_status_for_run`` and the wake/ping kind maps key on the
    outcome value, and the column is schema-visible. Changing it is a migration plus every
    consumer; the event payload is where the cause is disambiguated instead.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="budget stop outcome")
        kb.claim_task(conn, tid)
        _stop(conn, tid, {"stop_reason": "budget_exhausted", "budget_used": 150, "budget_max": 150})

        row = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        assert row[0] == "timed_out"


def test_a_real_runtime_cap_reaches_the_event(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="runtime cap stop")
        kb.claim_task(conn, tid)
        _stop(conn, tid, {"stop_reason": "max_runtime", "limit_seconds": 3600})

        payload = _stop_event(conn, tid).payload
        assert payload["stop_reason"] == "max_runtime"
        assert payload["limit_seconds"] == 3600
        assert "budget_used" not in payload


def test_stop_without_extra_fields_is_unchanged(kanban_home: Path) -> None:
    """Absent ``event_payload_extra`` → the old payload shape, exactly."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain stop")
        kb.claim_task(conn, tid)
        _stop(conn, tid, None)

        payload = _stop_event(conn, tid).payload
        assert set(payload) == {"error", "failures", "retry_status"}
