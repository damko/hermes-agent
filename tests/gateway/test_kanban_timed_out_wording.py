"""The Telegram-side ping must name the stop cause too (card t_99b457d1).

``_fmt_timed_out`` read only ``limit_seconds`` and said "its time limit" when the key was
absent — which is every budget stop. The event now carries ``stop_reason`` and the budget
counters, so the ping can name the real cause and stay neutral when none was recorded.
"""

from types import SimpleNamespace

from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS


class _Ev:
    def __init__(self, payload: dict):
        self.payload = payload


def _ping(payload: dict) -> str:
    msg, _, _ = _EVENT_FORMATTERS["timed_out"](_Ev(payload), SimpleNamespace(head="Kanban t_x", task_id="t_x"))
    return msg


def test_budget_stop_names_the_iterations() -> None:
    msg = _ping({
        "error": "Iteration budget exhausted (150/150) — task could not complete within the allowed iterations",
        "stop_reason": "budget_exhausted",
        "budget_used": 150,
        "budget_max": 150,
    })
    assert "iterations=150/150" in msg
    assert "time limit" not in msg
    assert "minute limit" not in msg
    assert "retried automatically" in msg


def test_budget_counters_alone_name_the_iterations() -> None:
    msg = _ping({"budget_used": 40, "budget_max": 40})
    assert "iterations=40/40" in msg


def test_real_runtime_cap_keeps_the_minutes_wording() -> None:
    msg = _ping({"stop_reason": "max_runtime", "limit_seconds": 3600})
    assert "60-minute limit" in msg
    assert "iterations=" not in msg


def test_unreadable_limit_is_not_a_limit() -> None:
    msg = _ping({"limit_seconds": "not-a-number"})
    assert "without a completion report" in msg
    assert "time limit" not in msg
