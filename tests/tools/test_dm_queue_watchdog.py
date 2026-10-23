"""Fork additions for the DM lane (upstream #122370): never drop a delivery, never let a silent
turn hold the profile lock.

Covers the two loss paths measured live on 2026-10-23:
  * the profile turn lock's bounded wait ended in ``target_busy`` and the payload was deleted;
  * a turn whose provider socket went silent held the lock for 42 minutes on 51 s of CPU.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import bot_mode_dm, bot_relay
from tools.bot_relay import TurnBusyError, acquire_turn_lock, turn_lock_busy, turn_lock_holder

SILENT = "import time\ntime.sleep(60)\n"

# A child that behaves like the real one: it publishes its own PROGRESS clock at the path the
# spawner handed it. Output is bookkeeping, the clock is the witness (#122370 review).
BEAT = (
    "import json, os, sys, time\n"
    "path = os.environ['HERMES_QUIET_HEARTBEAT_FILE']\n"
    "def beat(desc='work', pid=None):\n"
    "    tmp = path + '.tmp'\n"
    "    with open(tmp, 'w') as fh:\n"
    "        json.dump({'pid': pid or os.getpid(), 'ts': time.time(),\n"
    "                   'progress_ts': time.time(), 'desc': desc}, fh)\n"
    "    os.replace(tmp, path)\n"
)


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


# ── idle watchdog ────────────────────────────────────────────────────────────────────────────────


def test_watchdog_kills_a_silent_turn_and_its_children():
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.8)

    assert outcome.killed_idle is True
    assert outcome.elapsed < 20, "the watchdog must not wait for the child to finish on its own"
    assert outcome.returncode < 0, "the turn must be signalled, not exited cleanly"
    grandchild = int(outcome.stdout.split()[0])
    assert _wait_gone(grandchild), "the whole process group must die, or a child keeps the lock"


def test_watchdog_leaves_a_turn_that_keeps_making_progress_alone():
    """A quiet tool (17 min, no output) is NOT idle: the child's progress clock is the witness."""
    script = BEAT + (
        "for i in range(6):\n"
        "    beat('tool: terminal')\n"
        "    time.sleep(0.2)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.8)

    assert outcome.killed_idle is False
    assert outcome.returncode == 0


def test_watchdog_kills_a_turn_that_prints_but_makes_no_progress():
    """Output is not progress: a wedged turn that keeps painting its spinner must still die."""
    script = "import time\nfor i in range(6):\n    print('tick', i, flush=True)\n    time.sleep(0.2)\n"
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.8)

    assert outcome.killed_idle is True
    assert outcome.returncode < 0


def test_watchdog_kills_a_turn_whose_progress_stops():
    script = BEAT + (
        "for i in range(3):\n"
        "    beat('tool: terminal')\n"
        "    time.sleep(0.2)\n"
        "time.sleep(60)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.8)

    assert outcome.killed_idle is True
    assert outcome.elapsed < 20


def test_watchdog_never_kills_a_child_that_already_reported():
    """#113608: after the turn ends the child lingers for a nested reply — silence is by design."""
    script = BEAT + (
        "with open(os.environ['HERMES_QUIET_TURN_REPORT_FILE'], 'w') as fh:\n"
        "    json.dump({'pid': os.getpid(), 'exit_code': 0, 'error': '', 'reply': 'done'}, fh)\n"
        "time.sleep(2.0)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.5)

    assert outcome.killed_idle is False and outcome.timed_out is False


def test_watchdog_stops_a_turn_that_works_past_the_ceiling():
    script = BEAT + (
        "while True:\n"
        "    beat('tool: terminal')\n"
        "    time.sleep(0.1)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script],
                                            idle_seconds=5.0, turn_cap_seconds=0.8)

    assert outcome.timed_out is True
    assert outcome.killed_idle is False
    assert outcome.returncode < 0


def test_watchdog_ignores_a_heartbeat_from_another_process():
    """A stale file from an earlier turn must not keep this turn alive."""
    # A pid that is certainly dead and was a child of this test: ``os.getpid() + 500000`` used to
    # land on a live process on a busy box, and the guard then blocked the real kill it provoked.
    script = BEAT + (
        f"beat('stale', pid={_dead_pid()})\n"
        "time.sleep(60)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.8)

    assert outcome.killed_idle is True


def test_watchdog_records_the_last_tool_for_the_evidence_line():
    script = (
        "import json, time\n"
        "print(json.dumps({'tool': 'terminal'}), flush=True)\n"
        "time.sleep(60)\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.6)

    assert outcome.killed_idle is True
    assert outcome.last_tool == "terminal"


def test_watchdog_closes_stdin_for_the_child():
    """The old subprocess.run(stdin=DEVNULL) guarantee survives the Popen rewrite."""
    script = "import sys\ndata = sys.stdin.read()\nprint('stdin-closed' if data == '' else data)\n"
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=5)

    assert outcome.returncode == 0
    assert outcome.stdout.strip() == "stdin-closed"


def test_idle_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("HERMES_DM_IDLE_SECONDS", "42.5")
    assert bot_mode_dm.dm_idle_seconds() == pytest.approx(42.5)

    monkeypatch.setenv("HERMES_DM_IDLE_SECONDS", "not-a-number")
    assert bot_mode_dm.dm_idle_seconds() == bot_mode_dm._DM_IDLE_SECONDS_FALLBACK

    monkeypatch.setenv("HERMES_DM_IDLE_SECONDS", "0")
    assert bot_mode_dm.dm_idle_seconds() == 0.0, "0 disables the watchdog"


# ── lock holder metadata ────────────────────────────────────────────────────────────────────────


def test_turn_lock_records_its_holder_and_reports_it_to_the_next_waiter(tmp_path):
    assert turn_lock_busy(tmp_path, "default") is False

    with acquire_turn_lock(tmp_path, "default", timeout_seconds=5):
        assert turn_lock_busy(tmp_path, "default") is True
        holder_pid, holder_elapsed = turn_lock_holder(tmp_path, "default")
        assert holder_pid == os.getpid()
        assert holder_elapsed >= 0.0

        with pytest.raises(TurnBusyError) as excinfo:
            with acquire_turn_lock(tmp_path, "default", timeout_seconds=0.2):
                pass
        err = excinfo.value
        assert err.reason == "target_busy"
        assert err.holder_pid == os.getpid()
        assert "held by pid" in str(err)

    assert turn_lock_busy(tmp_path, "default") is False
    assert turn_lock_holder(tmp_path, "default") == (0, 0.0)


# ── durable queue ──────────────────────────────────────────────────────────────────────────────

ARGV = ["hermes", "-p", "default", "chat", "--bot-mode"]


@pytest.fixture()
def relay(tmp_path, monkeypatch):
    """Point the queue at a temp relay root and keep the watchdog log out of the real home."""
    monkeypatch.setattr(bot_mode_dm, "_lock_root", lambda: tmp_path)
    monkeypatch.setattr(bot_mode_dm, "_relay_base", lambda: tmp_path / "bot_relay")
    monkeypatch.setattr(bot_mode_dm, "_default_home", lambda: str(tmp_path))
    return tmp_path / "bot_relay" / bot_mode_dm._DM_QUEUE_DIRNAME / "default"


def _dm_file(tmp_path: Path, text: str = "please review the queue fix") -> str:
    path = tmp_path / "dm-abc123.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_target_busy_parks_the_payload_instead_of_deleting_it(relay, tmp_path, capsys):
    dm_file = _dm_file(tmp_path)

    code = bot_mode_dm._queue_instead_of_drop(
        "default", ARGV, dm_file, None, reason="target_busy",
        detail="queued behind a live turn (pid 4242, 3m12s)")

    assert code == 0, "a queued delivery is not an error"
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "queued"
    assert payload["reason"] == "target_busy"
    assert "NOT delivered yet" in payload["detail"]
    assert "pid 4242" in payload["detail"]

    queued = Path(payload["queued_file"])
    assert queued.read_text(encoding="utf-8") == "please review the queue fix"
    envelope = json.loads((queued.parent / f"{payload['delivery_id']}.json").read_text())
    assert envelope["reason"] == "target_busy"
    assert envelope["profile"] == "default"
    assert envelope["argv"] == ARGV
    assert envelope["attempts"] == 1


def test_drain_starts_a_runner_only_when_the_turn_window_is_free(relay, tmp_path, monkeypatch):
    dm_file = _dm_file(tmp_path)
    bot_mode_dm._queue_instead_of_drop("default", ARGV, dm_file, None,
                                       reason="target_busy", detail="queued behind a live turn")
    started: list[dict] = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda record, profile_dir: started.append(record))

    with acquire_turn_lock(tmp_path, "default", timeout_seconds=5):
        assert bot_mode_dm.drain_queued_dms("default") == 0
        assert started == []
        assert list(relay.glob("*.json")), "a busy profile keeps its envelope queued"

    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert started and started[0]["payload"].endswith(".payload.txt")
    assert not list(relay.glob("*.json")), "a drained envelope is removed"


def test_run_delivery_queues_on_target_busy_and_keeps_the_text(relay, tmp_path, monkeypatch, capsys):
    """The exact regression: a second DM into a busy profile must survive, not vanish."""
    from tools import bot_relay

    dm_file = _dm_file(tmp_path, "second message that used to be lost")
    monkeypatch.setattr(bot_mode_dm, "_local_delivery_home", lambda argv: None)
    monkeypatch.setattr(bot_mode_dm, "_live_intent_file", lambda path: str(tmp_path / "no-intent"))
    monkeypatch.setattr(bot_relay, "delivery_env", lambda author, home: {})

    def busy_lock(argv, *, stdin_file):
        raise TurnBusyError("default", 300.0, 4242, 192.0)

    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", busy_lock)
    started: list[dict] = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda record, profile_dir: started.append(record) or 4242)

    code = bot_mode_dm._run_delivery(ARGV, dm_file, stdin_file=False)

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "queued"
    assert payload["holder"] == {"pid": 4242, "elapsed_seconds": 192.0}
    assert not os.path.exists(dm_file), "the scratch payload is swept after it is copied"
    assert Path(payload["queued_file"]).read_text(encoding="utf-8") == "second message that used to be lost"
    assert [r["reason"] for r in started] == ["target_busy"], "the park must drain the queue it filled"


def test_idle_timeout_queues_the_delivery(relay, tmp_path, monkeypatch, capsys):
    from tools import bot_relay

    dm_file = _dm_file(tmp_path, "message behind a wedged turn")
    monkeypatch.setattr(bot_mode_dm, "_local_delivery_home", lambda argv: None)
    monkeypatch.setattr(bot_mode_dm, "_live_intent_file", lambda path: str(tmp_path / "no-intent"))
    monkeypatch.setattr(bot_relay, "delivery_env", lambda author, home: {})
    outcome = bot_mode_dm._TurnOutcome(returncode=-signal.SIGKILL, stdout="", stderr="", pid=515,
                                       elapsed=181.4, idle_seconds=180.0, killed_idle=True,
                                       last_tool="terminal")

    def wedged_turn(*args, **kwargs):
        raise bot_mode_dm.TurnIdleExceeded(outcome)

    # A real lock double, not nullcontext: queueing a wedged delivery must still drop the
    # per-profile turn lock, or every later DM into that profile waits forever (#93091).
    lock_state = {"held": False, "released": False}

    class _Lock:
        def __enter__(self):
            lock_state["held"] = True
            return self

        def __exit__(self, *exc_info):
            lock_state["held"] = False
            lock_state["released"] = True
            return False

    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", lambda *a, **k: _Lock())
    monkeypatch.setattr(bot_mode_dm, "_run_local_turn", wedged_turn)
    started: list[dict] = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda record, profile_dir: started.append(record) or 4242)

    code = bot_mode_dm._run_delivery(ARGV, dm_file, stdin_file=False)

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "queued"
    assert payload["reason"] == "idle_timeout"
    assert "last tool: terminal" in payload["detail"]
    assert Path(payload["queued_file"]).exists()
    assert lock_state["released"] and not lock_state["held"], "the turn lock must be released"
    assert [r["attempts"] for r in started] == [1], \
        "the park must start the queued delivery: the killed turn freed the window"


# ── durable queue: atomic claim, boot death, bounded attempts ───────────────────────────────────


def _dead_pid() -> int:
    """A pid that is certainly dead: this process started it and reaped it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _park_one(dm_file: str) -> dict:
    return bot_mode_dm._enqueue_delivery(
        "default", ARGV, dm_file, None, reason="target_busy", detail="queued behind a live turn",
        holder={"pid": 4242, "elapsed_seconds": 1.0})


def _claim_file(record: dict) -> Path:
    return Path(f"{record['payload']}{bot_mode_dm._DM_CLAIM_SUFFIX}")


def test_two_concurrent_drains_cannot_start_two_runners_for_one_envelope(relay, tmp_path):
    """Done-when #2: the claim is one atomic rename, so exactly one drain wins the envelope."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    record = _park_one(_dm_file(tmp_path))
    envelope = relay / f"{record['delivery_id']}.json"
    barrier = threading.Barrier(8)

    def claim(_: int) -> object:
        barrier.wait()
        return bot_mode_dm._claim_envelope(envelope, record)

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim, range(8)))

    assert sum(1 for c in claims if c is not None) == 1, "one envelope, one runner"
    assert not envelope.exists()


def test_a_live_claim_keeps_the_next_drain_away(relay, tmp_path, monkeypatch):
    record = _park_one(_dm_file(tmp_path))
    started: list[dict] = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda record, profile_dir: started.append(dict(record)) or os.getpid())
    # The stub runner is this process, whose argv does not name the payload; the real child is
    # spawned with ``query-file <payload>``, which is what the ownership check reads.
    monkeypatch.setattr(bot_mode_dm, "_pid_runs_payload", lambda pid, payload: True)

    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert bot_mode_dm.drain_queued_dms("default") == 0, "a claimed delivery is not handed out twice"
    assert len(started) == 1
    assert json.loads(_claim_file(record).read_text(encoding="utf-8"))["runner_pid"] == os.getpid()


def test_a_queued_message_is_delivered_by_the_drain_that_parks_it(relay, tmp_path, monkeypatch):
    """Done-when #1 end to end: the park starts a runner at once, and that runner delivers the
    message with no further turn into the profile. The stub spawner delivers in this process."""
    record = _park_one(_dm_file(tmp_path))
    payload = record["payload"]
    _run_as_runner(monkeypatch, tmp_path, 0)
    spent = []

    def spawn_and_deliver(rec, profile_dir):
        spent.append(bot_mode_dm._run_delivery(list(rec["argv"]), rec["payload"], stdin_file=False))
        return os.getpid()

    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner", spawn_and_deliver)
    monkeypatch.setattr(bot_mode_dm, "_pid_runs_payload", lambda pid, path: True)

    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert spent == [0], "the queued delivery must run and report success"
    assert not Path(payload).exists(), "a delivered payload leaves the queue"
    assert not list(relay.glob("*.json")), "no envelope survives a delivery"


def test_a_runner_that_fails_to_start_leaves_the_payload_queued(relay, tmp_path, monkeypatch):
    """Done-when #2 (second shape): a dead spawn must hand the envelope back, not eat it."""
    record = _park_one(_dm_file(tmp_path))

    def boom(record: dict, profile_dir: Path) -> None:
        raise OSError("no interpreter")

    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner", boom)
    assert bot_mode_dm.drain_queued_dms("default") == 0

    envelope = relay / f"{record['delivery_id']}.json"
    assert json.loads(envelope.read_text(encoding="utf-8"))["payload"] == record["payload"]
    assert not list(relay.glob("*.claimed"))
    assert Path(record["payload"]).read_text(encoding="utf-8") == "please review the queue fix"


def _run_as_runner(monkeypatch, tmp_path: Path, code: int) -> None:
    """Make a claimed delivery run in this process, with every outside call stubbed."""
    from tools import bot_relay

    monkeypatch.setattr(bot_mode_dm, "_local_delivery_home", lambda argv: None)
    monkeypatch.setattr(bot_mode_dm, "_live_intent_file", lambda path: str(tmp_path / "no-intent"))
    monkeypatch.setattr(bot_relay, "delivery_env", lambda author, home: {})
    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(bot_mode_dm, "_run_local_turn", lambda *a, **k: code)


def test_two_concurrent_drains_start_exactly_one_runner(relay, tmp_path, monkeypatch):
    """Finding 8: the claim test counted renames; this counts the runners that a race really starts."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    spawns: list[dict] = []
    # The stub runner "lives" as this process, so a claim left in one round is not swept in the
    # next: only the race under test can start a second runner.
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda record, profile_dir: spawns.append(dict(record)) or os.getpid())
    monkeypatch.setattr(bot_mode_dm, "_pid_runs_payload", lambda pid, payload: True)
    rounds = 6
    for _ in range(rounds):                 # a check-then-act claim only rarely loses one race
        _park_one(_dm_file(tmp_path))
        barrier = threading.Barrier(8)

        def drain(_: int) -> int:
            barrier.wait()
            return bot_mode_dm.drain_queued_dms("default")

        with ThreadPoolExecutor(max_workers=8) as pool:
            started = list(pool.map(drain, range(8)))
        assert sum(started) == 1, "one envelope, one runner"

    assert len(spawns) == rounds


def test_claim_moves_the_envelope_instead_of_copying_it(relay, tmp_path):
    """Finding 8: a copy leaves the envelope claimable twice; the rename is what makes one owner."""
    record = _park_one(_dm_file(tmp_path))
    envelope = relay / f"{record['delivery_id']}.json"

    claim = bot_mode_dm._claim_envelope(envelope, record)

    assert claim is not None and claim.exists()
    assert not envelope.exists(), "the claim is a rename: a copy would leave a second claimable envelope"
    assert bot_mode_dm._claim_envelope(envelope, record) is None, "a second claim must lose"


def test_a_failed_claimed_runner_keeps_the_payload_and_the_sweep_retries_it(relay, tmp_path,
                                                                           monkeypatch):
    """Finding 1: a claimed payload deleted on a failed attempt is a message nobody can retry."""
    _run_as_runner(monkeypatch, tmp_path, code=3)
    record = _park_one(_dm_file(tmp_path))
    payload = Path(record["payload"])
    bot_mode_dm._claim_envelope(relay / f"{record['delivery_id']}.json", record)

    assert bot_mode_dm._run_delivery(ARGV, str(payload), stdin_file=False) == 3
    claim = _claim_file(record)
    assert payload.exists() and claim.exists(), "a failed attempt must keep the message"

    # The runner died with its claim: only then may the sweep hand the payload out again.
    owned = json.loads(claim.read_text(encoding="utf-8"))
    owned["runner_pid"] = _dead_pid()
    claim.write_text(json.dumps(owned), encoding="utf-8")
    retried: list[dict] = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda record, profile_dir: retried.append(dict(record)) or _dead_pid())

    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert retried[0]["payload"] == str(payload) and retried[0]["attempts"] == 2
    # The sweep restored it and the drain claimed it again for the retry: still one message.
    assert json.loads(claim.read_text(encoding="utf-8"))["attempts"] == 2


def test_a_delivered_claimed_payload_is_removed_without_a_drop_record(relay, tmp_path, monkeypatch):
    """The happy path still tidies up: no payload, no claim, and no ``queue_dropped`` in the log."""
    _run_as_runner(monkeypatch, tmp_path, code=0)
    record = _park_one(_dm_file(tmp_path))
    payload = Path(record["payload"])
    bot_mode_dm._claim_envelope(relay / f"{record['delivery_id']}.json", record)

    assert bot_mode_dm._run_delivery(ARGV, str(payload), stdin_file=False) == 0

    assert not payload.exists() and not _claim_file(record).exists()
    log = tmp_path / "logs" / bot_mode_dm._DM_WATCHDOG_LOG
    assert "queue_dropped" not in (log.read_text(encoding="utf-8") if log.exists() else "")


def test_a_vanished_payload_is_logged_not_taken_for_a_delivery(relay, tmp_path):
    """Finding 2: an absent payload used to remove the only record of the message, silently."""
    profile_dir = relay
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = profile_dir / f"gone{bot_mode_dm._DM_PAYLOAD_SUFFIX}"
    claim = bot_mode_dm._claim_path(str(payload))
    claim.write_text(json.dumps({"delivery_id": "gone", "profile": "default", "attempts": 1,
                                 "runner_pid": _dead_pid()}), encoding="utf-8")

    assert bot_mode_dm.drain_queued_dms("default") == 0

    log = tmp_path / "logs" / bot_mode_dm._DM_WATCHDOG_LOG
    text = log.read_text(encoding="utf-8")
    assert "queue_dropped" in text and "payload vanished" in text
    assert not claim.exists()


def test_a_claim_with_a_live_drain_is_not_restored_while_that_drain_may_still_spawn(relay,
                                                                                   tmp_path):
    """Finding 4: the drain records its own pid before the spawn, so a stalled drain keeps its claim."""
    profile_dir = relay
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = profile_dir / f"spawning{bot_mode_dm._DM_PAYLOAD_SUFFIX}"
    payload.write_text("still mine", encoding="utf-8")
    claim = bot_mode_dm._claim_path(str(payload))
    # Claimed by this very process and never annotated with a runner pid: the spawn window. The
    # mtime is older than the grace, so only the live drain pid can keep the claim — which is
    # exactly what a sweep that consults the runner pid alone fails to do.
    claim.write_text(json.dumps({"delivery_id": "spawning", "profile": "default", "attempts": 1,
                                 "drain_pid": os.getpid(), "payload": str(payload)}),
                     encoding="utf-8")
    stale = time.time() - bot_mode_dm._DM_CLAIM_GRACE_SECONDS - 60
    os.utime(claim, (stale, stale))

    assert bot_mode_dm.drain_queued_dms("default") == 0
    assert claim.exists() and payload.exists(), "a live drain keeps its claim"


def test_a_runner_that_dies_at_boot_is_requeued_with_a_bumped_attempt(relay, tmp_path, monkeypatch):
    """The claim outlives a runner killed before it delivered; the sweep hands the payload back."""
    record = _park_one(_dm_file(tmp_path))
    dead = _dead_pid()
    started: list[dict] = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda r, d: started.append(dict(r)) or dead)

    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert not list(relay.glob("*.json")), "the envelope is claimed, not yet requeued"

    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda r, d: started.append(dict(r)) or os.getpid())
    assert bot_mode_dm.drain_queued_dms("default") == 1, "the sweep must restore and re-drain it"
    assert [r["attempts"] for r in started] == [1, 2]


def test_attempts_rise_on_every_requeue_and_the_limit_stops_a_repeated_kill(relay, tmp_path,
                                                                           monkeypatch, capsys):
    """Done-when #3: a provider that hangs on every attempt is retried a bounded number of times."""
    from tools import bot_relay

    monkeypatch.setattr(bot_mode_dm, "_local_delivery_home", lambda argv: None)
    monkeypatch.setattr(bot_mode_dm, "_live_intent_file", lambda path: str(tmp_path / "no-intent"))
    monkeypatch.setattr(bot_relay, "delivery_env", lambda author, home: {})
    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", lambda *a, **k: contextlib.nullcontext())
    outcome = bot_mode_dm._TurnOutcome(returncode=-signal.SIGKILL, stdout="", stderr="", pid=515,
                                       elapsed=181.4, idle_seconds=180.0, killed_idle=True,
                                       last_tool="terminal")

    def wedged_turn(*args, **kwargs):
        raise bot_mode_dm.TurnIdleExceeded(outcome)

    monkeypatch.setattr(bot_mode_dm, "_run_local_turn", wedged_turn)
    started: list[dict] = []
    dead = _dead_pid()
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda r, d: started.append(dict(r)) or dead)

    assert bot_mode_dm._run_delivery(ARGV, _dm_file(tmp_path, "hung provider"),
                                     stdin_file=False) == 0
    payload = started[0]["payload"]
    for _ in range(bot_mode_dm._DM_QUEUE_MAX_ATTEMPTS):  # every pass re-parks from the claimed file
        bot_mode_dm._run_delivery(ARGV, payload, stdin_file=False)

    assert [r["attempts"] for r in started] == list(range(1, bot_mode_dm._DM_QUEUE_MAX_ATTEMPTS + 1))
    assert not list(relay.glob("*")), "an exhausted delivery is dropped, not left behind"
    log = (tmp_path / "logs" / bot_mode_dm._DM_WATCHDOG_LOG).read_text(encoding="utf-8")
    assert "queue_dropped" in log
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "refused"


def test_a_live_runner_past_the_old_flat_ceiling_keeps_its_claim(relay, tmp_path, monkeypatch):
    """Review finding 4: the ceiling must outlive the turn cap, or a long live turn is swept and
    the same payload is handed to a second runner — a duplicate delivery."""
    monkeypatch.setattr(bot_mode_dm, "dm_turn_cap_seconds", lambda: 5400.0)
    monkeypatch.setattr(bot_mode_dm, "_pid_runs_payload", lambda pid, payload: True)
    relay.mkdir(parents=True, exist_ok=True)
    payload = relay / f"longturn{bot_mode_dm._DM_PAYLOAD_SUFFIX}"
    payload.write_text("still running", encoding="utf-8")
    claim = bot_mode_dm._claim_path(str(payload))
    claim.write_text(json.dumps({"delivery_id": "longturn", "profile": "default", "attempts": 1,
                                 "runner_pid": os.getpid(), "payload": str(payload)}),
                     encoding="utf-8")
    old = time.time() - bot_mode_dm._DM_CLAIM_MAX_AGE_SECONDS - 60
    os.utime(claim, (old, old))

    assert bot_mode_dm._claim_owner_alive(claim, json.loads(claim.read_text(encoding="utf-8")))
    assert bot_mode_dm.drain_queued_dms("default") == 0
    assert claim.exists() and not list(relay.glob("*.json")), "a live turn keeps its payload"


def test_a_repark_leaves_the_claim_alone_and_the_next_drain_still_starts(relay, tmp_path,
                                                                        monkeypatch):
    """Review finding: the re-park moved the claim path aside to judge it, so a drain that claimed
    the new envelope in that window lost its own claim, and its runner read the payload as a live
    file. The re-park now writes the envelope and touches nothing else; ``_sweep_claims`` deletes a
    claim whose envelope is back in the queue, so the drain this park starts is not blocked."""
    record = _park_one(_dm_file(tmp_path))
    payload = record["payload"]
    claim = bot_mode_dm._claim_path(payload)
    bot_mode_dm._claim_envelope(relay / f"{record['delivery_id']}.json", record)
    before = claim.read_text(encoding="utf-8")

    again = bot_mode_dm._enqueue_delivery("default", ARGV, payload, None, reason="target_busy",
                                          detail="the turn window hung again",
                                          holder={"pid": 4242, "elapsed_seconds": 1.0})

    assert again["attempts"] == 2
    assert claim.read_text(encoding="utf-8") == before, "the re-park must not touch the claim"
    started = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda rec, d: started.append(rec["attempts"]) or os.getpid())
    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert started == [2], "the superseded claim must not hold the queued envelope back"
    fresh = json.loads(claim.read_text(encoding="utf-8"))
    assert fresh["attempts"] == 2 and fresh.get("runner_pid") == os.getpid(), \
        "the drain claimed the envelope the re-park published and recorded its runner"


def test_a_claim_of_a_live_runner_is_never_handed_out_again(relay, tmp_path, monkeypatch):
    """Review finding: claiming an envelope replaced a claim a live runner was using."""
    monkeypatch.setattr(bot_mode_dm, "_pid_runs_payload", lambda pid, payload: True)
    record = _park_one(_dm_file(tmp_path))
    envelope = relay / f"{record['delivery_id']}.json"
    bot_mode_dm._claim_envelope(envelope, record)
    claim = _claim_file(record)
    claim.write_text(json.dumps(dict(json.loads(claim.read_text(encoding="utf-8")),
                                     runner_pid=os.getpid())), encoding="utf-8")
    again = relay / f"{record['delivery_id']}.json"
    again.write_text(json.dumps(dict(record, attempts=2)), encoding="utf-8")

    assert bot_mode_dm._claim_envelope(again, record) is None
    assert json.loads(claim.read_text(encoding="utf-8"))["attempts"] == 1
    assert again.exists(), "the envelope waits for the sweep that clears a dead claim"


def test_a_recycled_pid_does_not_hold_the_queue_hostage(relay, tmp_path, monkeypatch):
    """Review finding: a claim whose pid now belongs to another process was read as owned, so the
    message waited out the whole 15 h ceiling instead of being delivered."""
    record = _park_one(_dm_file(tmp_path))
    bot_mode_dm._claim_envelope(relay / f"{record['delivery_id']}.json", record)
    claim = _claim_file(record)
    claim.write_text(json.dumps(dict(json.loads(claim.read_text(encoding="utf-8")),
                                     runner_pid=os.getpid())), encoding="utf-8")
    monkeypatch.setattr(bot_mode_dm, "_pid_runs_payload", lambda pid, payload: False)
    started = []
    monkeypatch.setattr(bot_mode_dm, "_spawn_queued_runner",
                        lambda rec, d: started.append(rec["attempts"]) or os.getpid())

    assert bot_mode_dm.drain_queued_dms("default") == 1
    assert started == [2], "the claim is handed back with attempts raised, not held for hours"


def test_a_failed_directory_sync_does_not_publish_the_envelope(tmp_path, monkeypatch):
    """Review finding: the publish synced the directory through a helper that swallowed the error,
    then the caller deleted the only claim, so a power loss left a payload with no record."""
    tmp = tmp_path / "a.json.tmp"
    tmp.write_text("{}", encoding="utf-8")
    target = tmp_path / "a.json"

    def failing_sync(path, *, strict=False):
        if strict:
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(bot_mode_dm, "_fsync_dir", failing_sync)

    assert bot_mode_dm._publish_no_clobber(tmp, target) is False
    assert not target.exists(), "an entry that may not survive a power loss is not published"
    assert not tmp.exists()


def test_an_escaping_exception_is_reported_and_not_left_for_a_lost_message(relay, tmp_path,
                                                                         monkeypatch, capsys):
    """Review finding, kept as upstream pins it: an exception outside the two park handlers deletes
    the original file, and the wrapper's typed refusal on stdout is what tells the sender. Changing
    the deletion would break
    tests/tools/test_bot_mode_dm.py::test_delivery_main_maps_launch_exception_to_one_and_unlinks, so
    this test records the boundary instead of moving it."""
    dm_file = _dm_file(tmp_path, "survives a crash")

    def boom(*args, **kwargs):
        raise RuntimeError("the delivery environment failed")

    monkeypatch.setattr(bot_relay, "delivery_env", boom)
    capsys.readouterr()

    assert bot_mode_dm._delivery_main(["--run-delivery", "query-file", str(dm_file), "argv"]) == 1
    refusal = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert refusal["reason"] and "delivery environment failed" in refusal["error"]
    assert not Path(dm_file).exists(), "upstream contract: a refused delivery is reported, not retried"


def test_a_refused_park_keeps_the_only_copy_of_the_message(relay, tmp_path, monkeypatch, capsys):
    """Review finding: the queue refused the bytes, then the runner deleted the original anyway."""
    dm_file = _dm_file(tmp_path, "the only copy of this message")

    def busy_lock(argv, *, stdin_file):
        raise TurnBusyError("default", 300.0, 4242, 192.0)

    def eio(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", busy_lock)
    monkeypatch.setattr(os, "fsync", eio)

    code = bot_mode_dm._run_delivery(ARGV, dm_file, stdin_file=False)

    assert code == 1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "refused"
    assert Path(dm_file).exists(), "a refusal must not delete the only copy of the message"


def test_publishing_a_restored_envelope_never_overwrites_a_taken_name(relay, tmp_path):
    """Review finding: the restore checked for an envelope, then replaced the name anyway."""
    relay.mkdir(parents=True, exist_ok=True)
    target = relay / "taken.json"
    target.write_text(json.dumps({"attempts": 7}), encoding="utf-8")
    tmp = relay / "taken.json.tmp"
    tmp.write_text(json.dumps({"attempts": 1}), encoding="utf-8")

    assert bot_mode_dm._publish_no_clobber(tmp, target) is False
    assert json.loads(target.read_text(encoding="utf-8"))["attempts"] == 7
    assert not tmp.exists()


def test_a_restore_does_not_replace_the_envelope_a_repark_wrote(relay, tmp_path):
    record = _park_one(_dm_file(tmp_path))
    envelope = relay / f"{record['delivery_id']}.json"
    bot_mode_dm._claim_envelope(envelope, record)
    claim = _claim_file(record)
    envelope.write_text(json.dumps(dict(record, attempts=4)), encoding="utf-8")

    bot_mode_dm._restore_claim(claim, envelope, dict(record, attempts=1))

    assert json.loads(envelope.read_text(encoding="utf-8"))["attempts"] == 4
    assert not claim.exists()


def test_a_delivery_whose_bytes_cannot_be_synced_is_refused_not_queued(relay, tmp_path,
                                                                      monkeypatch, capsys):
    """Review finding 7: a suppressed fsync still printed ``status: queued`` for bytes the disk
    never took. The sender must hear a refusal instead."""
    def eio(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", eio)
    code = bot_mode_dm._queue_instead_of_drop("default", ARGV, _dm_file(tmp_path), None,
                                              reason="target_busy", detail="behind a live turn",
                                              holder={"pid": 4242, "elapsed_seconds": 1.0})

    assert code == 1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "refused" and "NOT delivered" in out["error"]
    assert not list(relay.glob("*.json")), "no envelope for a payload the disk never took"


def test_watchdog_does_not_kill_a_finished_turn_with_a_detached_pipe_holder():
    """#122370 review finding 2: a grandchild holding the pipe must not turn a finished turn into idle."""
    script = BEAT + (
        "print('turn-done', flush=True)\n"
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
    )
    outcome = bot_mode_dm._run_turn_watched([sys.executable, "-c", script], idle_seconds=0.8)

    assert outcome.killed_idle is False and outcome.timed_out is False
    assert outcome.elapsed < 10, "the drain must be bounded by the child's exit, not the idle budget"
