"""Bot Mode agent-to-agent DM tool — ``message_agent``.

Lets a Bot Mode agent message a teammate (a profile on this install, an agent on
a registered peer gateway, or one on another Desktop-connected machine): the
target is validated against the live roster, the attribution prefix is applied
server-side, and the reply arrives later via the background-process completion
notification (fire-and-forget). Containment: the schema is injected ONLY into a
bot's canonical "Bot Chat" session on a Bot-Mode-managed install (same gate as
``tools/bot_mode_probe.py``; never in the registry or any toolset), and dispatch
re-checks that gate so a forged call returns a structured error. Transports:
local → ``hermes -p <name> chat --in ~ -c "Bot Chat" --create-if-missing -Q
--query-file <tmp>``; peer → ``hermes peer dm <peer>[/<name>] < <tmp>``; both via
``terminal_tool(background=True, notify_on_complete=True)``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Top-level imports stay stdlib-only: this module also runs directly as the background
# delivery runner (``python bot_mode_dm.py --run-delivery …``); Hermes helpers import lazily.

logger = logging.getLogger(__name__)

MESSAGE_AGENT_TOOL_NAME = "message_agent"

# Message body cap — generous for real work, small enough that a runaway paste can't
# turn one DM into a context bomb on the recipient.
MESSAGE_MAX_CHARS = 16000
# The delivery process's completion notification IS the reply: size it like a message, plus the
# runner's header and failure prose, instead of the 2000-char tail a build log gets.
REPLY_COMPLETION_CHARS = MESSAGE_MAX_CHARS + 2000
# A runner owns and removes each DM file; this bounds residual plaintext lifetime if
# the machine dies between spawn ack and the runner's finally.
_DM_DIR_NAME = "hermes-dm"
_DM_STALE_SECONDS = 24 * 60 * 60
_LIVE_WAIT_SECONDS = 300
# A live owner may run a turn past one wait window, but each waiting runner holds ~46 MB and the
# sender's notify slot: past this the runner reports pending (do not resend) and exits.
_LIVE_WAIT_MAX_SECONDS = 30 * 60

# A SESSION_NOT_OWNED refusal is a lease report, not a busy agent: the holder is usually the
# other leg of the same round trip still winding down (or a headless one-shot turn), and the
# turn never started — no row, no model call — so retrying the SAME payload cannot duplicate
# anything. The runner waits out the holder on this bounded schedule before reporting the
# refusal. FORK ADDITION, damko 2026-10-14 (fleet decision; upstream #122370).
_REFUSAL_RETRY_WAITS = (10.0, 30.0, 60.0, 120.0)

# A delivery that cannot start its turn is QUEUED, never dropped (FORK ADDITION, damko 2026-10-23;
# upstream #122370). Two loss paths existed: the profile turn lock's bounded wait ended in a
# 'target_busy' refusal and the runner deleted the payload, and a turn whose provider socket went
# silent held the lock until TCP died (42 min measured live, 51 s of CPU). Both now park the payload
# in a durable queue under <relay>/queue/<profile>/ and the next free turn window drains it; the
# sender is told "queued behind a live turn (pid N, Nm Ns)" and never "sent".
_DM_IDLE_SECONDS_ENV = "HERMES_DM_IDLE_SECONDS"
_DM_IDLE_SECONDS_FALLBACK = 180.0
# Hard ceiling on ONE Bot Chat turn that keeps making progress. The idle budget cannot bound a turn
# that ticks forever, and a public DM is untrusted input, so the ceiling is a real cap, not a hint.
_DM_TURN_CAP_SECONDS_ENV = "HERMES_DM_TURN_MAX_SECONDS"
_DM_TURN_CAP_SECONDS_FALLBACK = 5400.0
_DM_TURN_CAP_OFF_SECONDS = 86400.0
_DM_QUEUE_DIRNAME = "queue"
_DM_QUEUE_MAX_ATTEMPTS = 5
# One queued delivery = <delivery_id>.json (envelope) + <delivery_id>.payload.txt (message).
_DM_PAYLOAD_SUFFIX = ".payload.txt"
# The envelope is renamed onto its payload's sidecar (``<payload>.claimed``) before the runner
# starts: an atomic rename, so two concurrent drains can never start two runners for one envelope,
# and the claim outlives a runner that dies before it delivers (FORK ADDITION, damko 2026-10-23).
_DM_CLAIM_SUFFIX = ".claimed"
# A claim with no recorded runner pid is still inside the drain's spawn window (milliseconds); only
# a claim older than this may be treated as abandoned.
_DM_CLAIM_GRACE_SECONDS = 300.0
# A live pid can be a reused one: past this age a claim is requeued even though its pid answers,
# or a delivery wedges for ever behind an unrelated process that inherited the number. Long enough
# that no legitimate DM turn reaches it — the watchdog stops a wedged turn in minutes.
_DM_CLAIM_MAX_AGE_SECONDS = 3600.0
_DM_WATCHDOG_LOG = "dm-watchdog.log"

# '<peer>/<agent>' — peer names are lowercase (``hermes peer`` normalizes them).
_PEER_TARGET_RE = re.compile(r"^([a-z0-9][a-z0-9_-]{0,63})/([a-zA-Z0-9][a-zA-Z0-9_-]{0,63})$")
# Same shape as ``tools.bot_relay._HANDLE_RE`` (kept local: see import note above).
_LOCAL_TARGET_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _default_home() -> str:
    from hermes_constants import get_process_hermes_home
    return str(get_process_hermes_home())


def message_agent_tool_schema() -> dict:
    """OpenAI-format schema for ``message_agent`` (injected, not registered)."""
    return {
        "type": "function",
        "function": {
            "name": MESSAGE_AGENT_TOOL_NAME,
            "description": (
                "Send a message to ANOTHER agent (teammate) on this install, or to an "
                "agent on a registered peer gateway. This is FIRE-AND-FORGET and "
                "asynchronous, like texting: it validates the target against the live "
                "roster, delivers your message into that agent's own Bot Chat with your "
                "attribution automatically prefixed, and returns immediately with a "
                "dispatch acknowledgement — status queued plus a delivery_id (the hand-off to a background "
                "delivery process, not a delivery receipt). It does NOT return their reply and you must "
                "not wait or poll for one — send it, finish your turn, and that process's "
                "completion notification wakes you with the outcome: their reply, or the "
                "delivery failure — unless the ack returns reply_delivery=\"poll\", in which case "
                "follow its process(action=\"wait\") instruction before ending the turn. COMPOSE the message yourself: write what YOU want to say to "
                "that agent (lead with the point; include the concrete ask or result). "
                "Never paste the user's words verbatim — paraphrase the actionable "
                "substance, and keep private 1:1 chat content private. Message one "
                "clearly relevant teammate when it genuinely helps the user's goal; "
                "don't fan out to several agents unless the user explicitly asked. "
                "Use the teammate roster in your system prompt (names + roles) to pick "
                "the right recipient; targets: a teammate name (e.g. 'researcher'), "
                "'<peer>/<agent>' for an agent on a registered peer gateway "
                "(e.g. 'spark/researcher', or just '<peer>' for the peer's main agent), "
                "or an agent on another connected machine from your roster (use "
                "'<handle>@<connection>' if the same handle exists on several)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "Who to message: a teammate profile name from your roster "
                            "('researcher', 'hermes' for the default agent), or "
                            "'<peer>' / '<peer>/<agent>' for a registered peer gateway."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "The message YOU composed for that agent (max "
                            f"{MESSAGE_MAX_CHARS} chars). Do not include the "
                            "'Message from …' prefix — it is added automatically."
                        ),
                    },
                },
                "required": ["target", "message"],
            },
        },
    }


# Sessions that must NOT get the DM tool: a kanban worker turn has no live surface to
# receive the answer, and its consultation lane is the card (shared conventions
# §Consultation). FORK ADDITION, damko 2026-09-30.
_DM_EXCLUDED_SOURCES = frozenset({"kanban"})


def dm_worker_excluded(agent: Any) -> bool:
    """True when this turn belongs to an excluded automation source (kanban worker).

    Reads the identifier the dispatcher stamps on a worker turn
    (``HERMES_SESSION_SOURCE=kanban``, ``hermes_cli/kanban_db_dispatch.py``) through the
    gateway's ContextVar-aware accessor, so a gateway-hosted worker is seen too. This
    deliberately does NOT go through ``run_agent._session_source_for_agent``: that import
    can fail outside a live agent process, and a gate must not depend on it. An
    unrecognised or unreadable value INCLUDES the session — on doubt the tool stays, it is
    never silently removed. Never raises."""
    try:
        try:
            from gateway.session_context import get_session_env
        except Exception:
            get_session_env = os.environ.get  # type: ignore[assignment]
        source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip().lower()
        if not source:
            source = str(os.environ.get("HERMES_SESSION_SOURCE", "") or "").strip().lower()
        return source in _DM_EXCLUDED_SOURCES
    except Exception:  # pragma: no cover — include on doubt
        return False


def message_agent_authorized(agent: Any) -> bool:
    """The ``message_agent`` gate: a protocol-enabled agent on a Bot-Mode-managed install.

    FORK PATCH (damko, 2026-09-30): the tool used to be injected ONLY into the canonical
    ``Bot Chat``. It is now injected into EVERY session of a managed install, so sibilla,
    claudio and every other agent can reach diogene from an ordinary session; the excluded
    automation sources (``_DM_EXCLUDED_SOURCES``) keep their card lane. The gate stays a
    pure function of session-stable inputs (install marker + session source), so it remains
    prompt-cache safe to re-evaluate on every tool-snapshot rebuild. Never raises."""
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False
        from tools.bot_mode_probe import is_bot_mode_managed

        # Managed-install check, NOT section non-emptiness: a SOUL.md carrying the
        # legacy protocol text gets an empty section but must still get the tool.
        if not is_bot_mode_managed(_agent_home(agent)):
            return False
        return not dm_worker_excluded(agent)
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("message_agent_authorized failed", exc_info=True)
        return False


def ensure_message_agent_tool(agent: Any) -> bool:
    """Inject the ``message_agent`` schema into a Bot Chat agent's tool list (once per turn).
    Idempotent and deterministic for the session's life (the gate is stable from the
    first turn), so the tool list is byte-identical across turns — prompt-cache safe. Never raises."""
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False
        tools = getattr(agent, "tools", None)
        present = bool(tools) and any(
            isinstance(t, dict) and t.get("function", {}).get("name") == MESSAGE_AGENT_TOOL_NAME
            for t in tools
        )
        if not present:
            if not message_agent_authorized(agent):
                return False
            if agent.tools is None:
                agent.tools = []
            agent.tools.append(message_agent_tool_schema())
        # Success means BOTH halves hold: a tool-surface rebuild (compaction, MCP refresh)
        # can keep the schema while valid_tool_names is republished without it, and an
        # advertised-but-nondispatchable tool sends the model hunting for shellouts (#96105).
        valid = getattr(agent, "valid_tool_names", None)
        if isinstance(valid, set):
            valid.add(MESSAGE_AGENT_TOOL_NAME)
        return True
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("ensure_message_agent_tool failed", exc_info=True)
        return False


def _resolve_local_name(target: str, roster: list[str], root: Path | None = None) -> Optional[str]:
    """Map a target to a local profile FOLDER id: 'hermes' → 'default'; an exact folder id
    (case-insensitive); else — when ``root`` is given — a friendly name or its Desktop @-slug
    (profile.yaml ``display_name`` / Bot Mode title: 'Scribe', '@scribe', 'Dr. Foo' → 'foo').
    Ambiguous friendly names resolve to None so a DM never lands on the wrong bot (#100671)."""
    want = target.strip().lower()
    if not want:
        return None
    if want == "hermes":
        return "default" if "default" in roster else None
    exact = next((name for name in roster if name.lower() == want), None)
    if exact is not None or root is None:
        return exact
    from tools.bot_mode_probe import alias_forms, local_alias_map

    aliases = local_alias_map(root)
    hits = set().union(*(aliases.get(form, set()) for form in alias_forms(want) | {want}))
    return next(iter(hits)) if len(hits) == 1 else None


def _err(message: str, *, roster: list[str] | None = None, peers: list[str] | None = None) -> str:
    from tools.bot_failure_reasons import classify_agent_error

    payload: dict[str, Any] = {"error": message, "reason": classify_agent_error(message)}
    if roster is not None:
        payload["teammates"] = roster
    if peers is not None:
        payload["peers"] = peers
    return json.dumps(payload)


def message_agent_tool(target: str = "", message: str = "", task_id: Optional[str] = None, agent: Any = None) -> str:
    """Deliver ``message`` to ``target``'s Bot Chat. Returns a JSON ack/error.
    ``agent`` is the calling AIAgent — used for the Bot Chat gate and sender identity."""
    home = _agent_home(agent)
    try:
        from tools.bot_mode_probe import (
            _dm_teammate, _display_name, _handle, _hermes_root, _peers,
            _profile_name as _self_profile_name, _roster, is_bot_mode_managed,
        )
        from tools.bot_relay import BOT_CHAT_TURN_ARGS, _hermes_cli

        if dm_worker_excluded(agent):
            return _err("message_agent is not available in a kanban worker turn — "
                        "ask through a consultation card instead; do not retry.")
        if not is_bot_mode_managed(home):
            return _err("This install is not Bot-Mode-managed (no bot roster); "
                        "message_agent is unavailable. Do not retry.")
    except Exception as exc:  # pragma: no cover — defensive
        return _err(f"Bot Mode gate check failed: {exc}")
    from hermes_cli.observability.shared_metrics_signals import record_feature_used
    record_feature_used("bot_mode", hermes_home=home)

    root, me = _hermes_root(Path(home)), _self_profile_name(Path(home))
    # Only DM teammates are addressable: Bot-Mode-managed profiles, plus the primary
    # ('default', the bare "@hermes" the relay hands out) so it never disappears from its
    # own install's roster. FORK ADDITION, damko 2026-10-15 (fleet DM-lane fix, t_43cea3b5).
    roster_homes = {name: d for name, d in _roster(root) if _dm_teammate(name, d)}
    roster = list(roster_homes)
    peers = _peers(root)
    teammates = [_handle(n) for n in roster if n != me]

    def _roster_err(msg: str) -> str:
        return _err(msg, roster=teammates, peers=peers)

    body = str(message or "").strip()
    if not body:
        return _err("message is required — compose what you want to say to that agent.")
    if len(body) > MESSAGE_MAX_CHARS:
        return _err(f"message too long ({len(body)} chars > {MESSAGE_MAX_CHARS}). "
                    "Send the essentials; share large content as a file path instead.")

    raw_target = str(target or "").strip().lstrip("@")
    if not raw_target:
        return _roster_err("target is required.")
    # Sender signature: the friendly name when the bot has one (#89720); the @handle stays the routing alias.
    content = f"Message from 🤖 {_display_name(me, roster_homes.get(me, Path(home)))} (@{_handle(me)}): " + body
    delivery = dict(task_id=task_id, agent=agent)
    # Attribution for the recipient's memory hooks; the text prefix above stays the human-facing signature.
    author = {"id": f"bot:{me}", "name": _handle(me), "is_bot": True}

    # Peer target: '<peer>/<agent>' or a bare registered peer name.
    peer_match = _PEER_TARGET_RE.match(raw_target)
    if peer_match or raw_target.lower() in peers:
        peer_name, peer_profile = peer_match.groups() if peer_match else (raw_target.lower(), None)
        if peer_name not in peers:
            return _roster_err(f"No registered peer named '{peer_name}'.")
        dm_target = f"{peer_name}/{peer_profile}" if peer_profile else peer_name
        # A peer dm crosses installs: qualify the id with this host so the peer's own '<me>' stays distinct.
        from agent.turn_author import bot_author_id, local_origin
        peer_author = {**author, "id": bot_author_id(me, local_origin())}
        # Pin the registry-owning profile: `hermes peer` resolves bot_peers via the profile-scoped
        # load_config(), while the roster above reads the machine-root config — the CLI must run
        # in that same profile or a secondary-profile bot sees an empty registry.
        # The delivery runs in a background service context whose PATH lacks the gateway's
        # venv bin dir, so a bare "hermes" resolves to a system install and dies on import
        # under the wrong interpreter (#108628). _hermes_cli pins the entrypoint beside
        # this interpreter; _delivery_lock/_local_delivery_home match argv[0] by basename,
        # so the absolute path stays compatible.
        # A bare peer name targets the peer's MAIN agent (its default profile) — on a
        # single-profile remote install that is a stock husk. Name the transport form so a
        # husk delivery is visible in the ack. FORK ADDITION, damko 2026-10-15.
        peer_label = (f"@{peer_profile} on peer '{peer_name}'" if peer_profile
                      else f"@{peer_name} (the peer's main agent) on peer '{peer_name}'")
        return _start_delivery([_hermes_cli(), "-p", _self_profile_name(root), "peer", "dm", dm_target], content,
                               peer_label, stdin_file=True,
                               author=peer_author, **delivery)

    # A connection-qualified target ('hermes@mini') names a relay row outright; it is the form the relay itself
    # hands out for a colliding row, and stamps on replies. Resolved locally first, a local bot whose friendly
    # name slugs to 'hermes-mini' captured it. An '@' name no connection answers to still resolves locally.
    if "@" in raw_target.strip().lstrip("@"):
        relayed = _try_relay_delivery(root, raw_target, content, me, **delivery)
        if relayed is not None:
            return relayed
    # Local teammate — folder id, or a friendly name / Desktop @-slug ('Scribe', 'Dr. Foo').
    resolved = _resolve_local_name(raw_target, roster, root)
    is_local_shape = bool(_LOCAL_TARGET_RE.match(raw_target))
    if resolved is None and not is_local_shape and "@" not in raw_target:
        return _roster_err(f"Invalid target: {raw_target!r}.")
    if resolved is None or resolved == me:
        # Unknown locally, or same-name target on ANOTHER connection (this gateway's 'default'
        # messaging the cloud 'default'): every Desktop-connected gateway is reachable via the
        # relay roster, so try that before reporting a resolution failure / self-message.
        relayed = _try_relay_delivery(root, raw_target, content, me, **delivery)
        if relayed is not None:
            return relayed
        if resolved == me:
            return _err("You can't message yourself. Pick a teammate from the roster.")
        return _roster_err(f"No teammate named '{raw_target}' on this install, on a connected "
                           "machine, or on a registered peer. Pick a name from the roster "
                           "(roles are listed in your system prompt).")
    return _start_delivery([_hermes_cli(), "-p", resolved, *BOT_CHAT_TURN_ARGS], content, f"@{_handle(resolved)}",
                           stdin_file=False, profile_home=roster_homes[resolved], author=author, **delivery)


def _try_relay_delivery(root: Path, raw_target: str, content: str, me: str, *,
                        task_id: Optional[str], agent: Any) -> Optional[str]:
    """Cross-connection delivery via the Desktop relay; None when the target doesn't
    resolve against the relay roster. The envelope is queued on disk for the Desktop
    to drain; a background waiter is spawned immediately so the relayed reply wakes
    the sender through the standard completion-notification path."""
    try:
        from tools.bot_mode_probe import _handle, local_taken_forms
        from tools.bot_relay import (
            EnvelopeRefusedError, _target_aliases, enqueue_envelope, read_remote_roster, remote_target_forms,
            resolve_remote_target, waiter_command,
        )

        roster = read_remote_roster(root)
        match = resolve_remote_target(raw_target, roster) if roster else None
        if match is None:
            return None
        if match == "ambiguous":
            want = raw_target.strip().lstrip("@").partition("@")[0].lower()
            forms = ", ".join(form for r, form in zip(roster, remote_target_forms(roster, local_taken_forms(root)))
                              if want in _target_aliases(r))
            return _err(f"'{raw_target}' exists on several connected machines — disambiguate with one of: {forms}.")
        try:
            envelope = enqueue_envelope(root, target=match, message=content, sender_profile=me, sender_handle=_handle(me))
        except EnvelopeRefusedError as exc:
            # Fail fast: target definitively offline — nothing was queued.
            # Structured refusal so the agent can distinguish it from a resolution error ('runtime_offline'
            # per the #93091 reason enum).
            return json.dumps({"error": str(exc), "reason": exc.reason})
        label = f"@{match['handle']} on {match['connection_label'] or match['connection_id']}"
        raw = _spawn_delivery(waiter_command(root, envelope), label, delivery_id=envelope["id"], task_id=task_id, agent=agent)
        waiter_error = json.loads(raw).get("error")
        if not waiter_error:
            return raw
        # The envelope is already queued and the Desktop drains it on its own, so a waiter that
        # failed to start loses only the reply wake-up. Reporting a hard failure here makes the
        # sender resend and deliver the message twice. Same shape as the live-owner branch of
        # _start_delivery: queued + notification_error.
        return json.dumps({
            "status": "queued", "delivery_id": envelope["id"], "to": label, "notification_error": waiter_error,
            "detail": (f"Message queued for {label}; the relay delivers it on its own, but the reply "
                       "waiter did not start, so the reply will NOT wake you. Do NOT resend."),
        })
    except Exception:
        logger.debug("relay delivery attempt failed", exc_info=True)
        return None


def _dm_dir() -> Path:
    uid_getter = getattr(os, "getuid", None)
    uid = uid_getter() if callable(uid_getter) else None
    path = Path(tempfile.gettempdir()) / (f"{_DM_DIR_NAME}-{uid}" if uid is not None else _DM_DIR_NAME)
    path.mkdir(mode=0o700, exist_ok=True)
    # Shared POSIX temp roots need a per-user directory. Fail closed if an
    # attacker pre-created the expected path or replaced it with a symlink.
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"DM temp path is not a directory: {path}")
    if uid is not None and info.st_uid != uid:
        raise PermissionError(f"DM temp directory is owned by another user: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        path.chmod(0o700)
    return path


def cleanup_bot_dm_cache(max_age_hours: float = _DM_STALE_SECONDS / 3600, *, now: float | None = None) -> int:
    """Delete orphaned DM payload files older than *max_age_hours*; returns count.
    Same contract as the other ``cleanup_*_cache`` helpers (hourly gateway housekeeping);
    legacy temp-root locations from versions predating the dedicated directory are swept too."""
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    temp_root = Path(tempfile.gettempdir())
    locations = [(temp_root, "hermes-dm-*.txt"), (temp_root, "hermes-relay-dm-*.txt")]
    with contextlib.suppress(OSError):
        dm_dir = _dm_dir()
        locations.append((dm_dir, "*.txt"))
        # Live-delivery intents (``<dm file>.live.json``, message plaintext included) outlive
        # their runner on purpose — a retry replays the same delivery id from them — so the
        # orphans of runners that never settled are swept here too.
        locations.append((dm_dir, "*.live.json"))
    from tools.bot_relay import unlink_files_older_than

    return sum(unlink_files_older_than(d, pattern, cutoff) for d, pattern in locations)


def _unlink_dm_file(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _write_dm_file(content: str) -> str:
    """The message rides a temp file — never inline shell text."""
    cleanup_bot_dm_cache()
    fd, path = tempfile.mkstemp(prefix="dm-", suffix=".txt", dir=_dm_dir(), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except BaseException:
        # If fdopen itself failed the raw descriptor is still ours; closing twice is harmless.
        with contextlib.suppress(OSError):
            os.close(fd)
        _unlink_dm_file(path)
        raise
    return path


class TurnIdleExceeded(RuntimeError):
    """A Bot Chat turn made no PROGRESS for the idle budget and was stopped by the watchdog.

    Progress, not output: the child's own progress clock (tool stamps, stream chunks, real agent
    activity) is the witness. A quiet tool that runs for 17 minutes keeps the clock fresh; a parked
    provider read that prints a wait notice every minute does not (#122370).
    """

    reason = "idle_timeout"

    def __init__(self, outcome: "_TurnOutcome"):
        self.outcome = outcome
        super().__init__(
            f"idle_timeout: turn pid {outcome.pid} made no progress for {outcome.idle_seconds:.0f}s "
            f"(elapsed {outcome.elapsed:.0f}s, last tool: {outcome.last_tool or 'unknown'}) — stopped. "
            "The message was NOT delivered.")


class TurnCapExceeded(TurnIdleExceeded):
    """A Bot Chat turn kept making progress past the hard ceiling and was stopped.

    The idle budget cannot bound a turn that ticks forever, and a public DM is untrusted input, so
    the ceiling caps the wall clock as well (#122370 review finding: unbounded run).
    """

    reason = "turn_timeout"

    def __init__(self, outcome: "_TurnOutcome"):
        RuntimeError.__init__(self, (
            f"turn_timeout: turn pid {outcome.pid} still running after {outcome.elapsed:.0f}s "
            f"(last tool: {outcome.last_tool or 'unknown'}) — stopped at the turn ceiling. "
            "The message was NOT delivered."))
        self.outcome = outcome


class _TurnOutcome:
    """``subprocess.CompletedProcess`` stand-in: the rest of this module reads rc/stdout/stderr."""

    def __init__(self, *, returncode: int, stdout: str, stderr: str, pid: int = 0,
                 elapsed: float = 0.0, idle_seconds: float = 0.0, killed_idle: bool = False,
                 last_tool: str = "", timed_out: bool = False):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr
        self.pid, self.elapsed = pid, elapsed
        self.idle_seconds, self.killed_idle, self.last_tool = idle_seconds, killed_idle, last_tool
        self.timed_out = timed_out


def dm_idle_seconds() -> float:
    """Idle budget for one Bot Chat turn: no output for this long means the turn is wedged."""
    raw = (os.environ.get(_DM_IDLE_SECONDS_ENV) or "").strip()
    try:
        value = float(raw) if raw else _DM_IDLE_SECONDS_FALLBACK
    except ValueError:
        value = _DM_IDLE_SECONDS_FALLBACK
    return max(0.0, value)


def dm_turn_cap_seconds() -> float:
    """Hard ceiling for one Bot Chat turn (0 disables it)."""
    raw = (os.environ.get(_DM_TURN_CAP_SECONDS_ENV) or "").strip()
    try:
        value = float(raw) if raw else _DM_TURN_CAP_SECONDS_FALLBACK
    except ValueError:
        value = _DM_TURN_CAP_SECONDS_FALLBACK
    return max(0.0, value)


_TOOL_MARKERS = (
    re.compile(r'"tool":\s*"([A-Za-z0-9_.:-]+)"'),
    re.compile(r"(?:→|->)\s*(?:tool\s+)?([a-z][a-z0-9_]*(?:_[a-z0-9_]+)+)\s*\("),
    re.compile(r"^Tool:?\s+([A-Za-z0-9_.:-]+)", re.MULTILINE),
)


def _last_tool_name(text: str) -> str:
    """Best-effort name of the tool the wedged turn was running (evidence, not a claim)."""
    if not text:
        return ""
    tail = text[-200_000:]
    for pattern in _TOOL_MARKERS:
        found = pattern.findall(tail)
        if found:
            return str(found[-1])[:80]
    return ""


def _run_turn_watched(argv: list[str], *, idle_seconds: float,
                      env: Optional[dict[str, str]] = None,
                      turn_cap_seconds: float = 0.0) -> _TurnOutcome:
    """Run one CLI turn; stop it when it makes no PROGRESS for ``idle_seconds``.

    Why not ``subprocess.run``: a provider socket with no read deadline parked a turn for 42 minutes
    on 51 s of CPU, holding the profile lock (measured 2026-10-23). Why not "output went quiet": a
    healthy offline turn is silent for 7–17 minutes while a tool runs, and a one-shot child prints
    nothing until the turn ends, so silence alone cannot tell "slow" from "wedged" (#122370 review).

    The child now witnesses its own progress: ``run_reported_turn`` hands it a heartbeat path and the
    agent writes its REAL progress clock there (``agent.activity_tracking._publish_turn_heartbeat``;
    provider-wait keepalives are classified ``PROVIDER_WAIT`` and never touch that clock). The same
    helper makes the child record its turn report the moment the turn ends, so a turn that finished
    while the idle kill was landing is booked as finished, never replayed (#122370 review finding 2).
    ``turn_cap_seconds`` (0 = off) bounds a turn that keeps making progress for too long.
    """
    from hermes_cli.quiet_single_query import ReportedTurnIdle, run_reported_turn

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dm-turn-") as workdir:
        try:
            cp = run_reported_turn(
                argv, env=env or {}, report_path=os.path.join(workdir, "turn.json"),
                heartbeat_path=os.path.join(workdir, "progress.json"),
                timeout=turn_cap_seconds if turn_cap_seconds > 0 else _DM_TURN_CAP_OFF_SECONDS,
                idle_timeout=idle_seconds if idle_seconds > 0 else None,
                # UTF-8 regardless of locale: the child writes UTF-8 and the DM lane must not mangle
                # a non-ASCII reply just because the daemon runs under a C/POSIX locale.
                encoding="utf-8", exit_grace=None, kill_group=True,
            )
        except (ReportedTurnIdle, subprocess.TimeoutExpired) as exc:
            idle = isinstance(exc, ReportedTurnIdle)
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or ""
            returncode = getattr(exc, "returncode", None)
            return _TurnOutcome(
                returncode=returncode if isinstance(returncode, int) else -signal.SIGKILL,
                stdout=stdout, stderr=stderr, pid=int(getattr(exc, "pid", 0) or 0),
                elapsed=time.monotonic() - start,
                idle_seconds=idle_seconds if idle else 0.0, killed_idle=idle,
                timed_out=not idle, last_tool=_last_tool_name(stdout + stderr))
        return _TurnOutcome(
            returncode=cp.returncode, stdout=cp.stdout or "", stderr=cp.stderr or "",
            pid=getattr(cp, "pid", 0) or 0, elapsed=time.monotonic() - start,
            last_tool=_last_tool_name((cp.stdout or "") + (cp.stderr or "")))


def _lock_root() -> Path:
    """The install root the relay locks and queue live under (same resolution as ``_delivery_lock``)."""
    from tools.bot_mode_probe import _hermes_root

    return _hermes_root(Path(_default_home()))


def _relay_base() -> Path:
    from tools.bot_relay import relay_root

    return relay_root(_lock_root())


def _queue_dir(profile: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(profile or ""))[:64] or "_"
    path = _relay_base() / _DM_QUEUE_DIRNAME / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _claim_path(dm_file: str | os.PathLike) -> Path:
    """Sidecar an envelope is renamed onto while its runner is alive (same dir, same stem)."""
    return Path(f"{Path(dm_file)}{_DM_CLAIM_SUFFIX}")


def _read_json_dict(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _claim_record(dm_file: str | os.PathLike) -> dict:
    """The envelope this payload was claimed from — {} when no drain claimed it."""
    return _read_json_dict(_claim_path(dm_file))


def _pid_alive(pid: int) -> bool:
    """Whether a pid still runs. A zombie has exited: it holds no lock and delivers nothing."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    with contextlib.suppress(OSError, IndexError, ValueError):
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    return True


def _claim_age(claim: Path) -> float:
    """Seconds since the claim was last written; 0 for an unreadable one, so its pid decides."""
    with contextlib.suppress(OSError):
        return max(0.0, time.time() - claim.stat().st_mtime)
    return 0.0


def _claim_max_age_seconds() -> float:
    """Age past which a claim is handed out again even though its pid still answers.

    The ceiling exists to defeat pid reuse, so it must outlive any legitimate runner: a runner
    waits for the profile turn lock and then runs one turn under the turn cap. A flat hour is
    shorter than the 5400 s cap, so a long live turn used to be swept and delivered twice
    (review finding, damko 2026-10-23).
    """
    cap = _DM_TURN_CAP_OFF_SECONDS
    with contextlib.suppress(Exception):
        cap = dm_turn_cap_seconds() or _DM_TURN_CAP_OFF_SECONDS
    wait = 0.0
    with contextlib.suppress(Exception):
        from tools.bot_relay import turn_wait_seconds
        wait = float(turn_wait_seconds())
    # One delivery runs more than one turn: ``_run_local_turn`` re-runs a classified failure once,
    # and every lease refusal replays the pair. Ten capped turns plus the refusal waits is the
    # honest bound, or a live retrying runner is swept and delivered twice (review finding, damko
    # 2026-10-23).
    turns = 2 * (1 + len(_REFUSAL_RETRY_WAITS))
    return max(_DM_CLAIM_MAX_AGE_SECONDS,
               turns * cap + sum(_REFUSAL_RETRY_WAITS) + wait + 600.0)


def _pid_runs_payload(pid: int, payload: str) -> bool:
    """Whether this pid's command line names this payload — the runner's argv carries it.

    A pid on its own proves nothing: Linux reuses pids, and a claim read as "owned" blocks the
    queue until its ceiling expires (review finding, damko 2026-10-23). No /proc means no evidence,
    so the caller keeps the older, pid-only reading.
    """
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return True
    return payload in raw.decode("utf-8", "replace")


def _claim_owner_alive(claim: Path, record: dict) -> bool:
    """Whether the process that claimed this payload can still deliver it.

    The runner owns the claim once it exists; the drain owns it only in the window between the
    rename and the spawn, and ignoring that window would hand an envelope out again while the drain
    was still about to start it (FORK FIX, damko 2026-10-23).
    """
    pid = int(record.get("runner_pid") or 0) or int(record.get("drain_pid") or 0)
    if pid > 0:
        if _claim_age(claim) >= _claim_max_age_seconds():
            return False
        if not _pid_alive(pid):
            return False
        runner = int(record.get("runner_pid") or 0)
        payload = str(record.get("payload") or "")
        if runner and payload and not _pid_runs_payload(runner, payload):
            # A recycled pid is not this claim's runner, and waiting out a 15 h ceiling on a claim
            # nobody owns would hold the message back as surely as losing it (review finding,
            # damko 2026-10-23). Only ``runner_pid`` is checked: the drain annotates its own pid a
            # moment before it spawns, and that process does not name the payload yet.
            return False
        return True
    # No pid at all: a claim written by hand or by an older drain. Only the claim's age can tell,
    # and guessing "dead" there would start a second runner for one envelope.
    return _claim_age(claim) < _DM_CLAIM_GRACE_SECONDS


def _claim_envelope(envelope: Path, record: dict) -> Optional[Path]:
    """Take an envelope out of the queue by atomic rename; None when another drain won it.

    The rename is the lock: two drains that both saw one envelope cannot both find it there
    (FORK ADDITION, damko 2026-10-23)."""
    claim = _claim_path(str(record.get("payload") or envelope))
    owned = _read_json_dict(claim)
    if owned and _claim_owner_alive(claim, owned):
        # A live runner already owns this payload. Overwriting its claim would hand the same
        # message to a second runner and leave the first unable to record its delivery (review
        # finding, damko 2026-10-23). ``_sweep_claims`` clears the claim of a runner that is gone.
        return None
    try:
        os.replace(envelope, claim)
    except OSError:
        return None
    return claim


def _note_claim_owner(claim: Path, record: dict, pid: Optional[int], *,
                      key: str = "runner_pid", **extra: Any) -> None:
    """Annotate the claim in place: which process owns it, or that it was delivered.

    The claim's own content is the base, never the caller's copy: a runner that re-parked rewrites
    the record, and an annotator writing from a stale copy would lower ``attempts`` back and let
    the queue retry past its limit (FORK FIX, damko 2026-10-23).
    """
    if not claim.exists():
        return  # the runner already re-parked and removed the claim; nothing to annotate
    owner = _read_json_dict(claim) or dict(record)
    owner["claimed_at"] = int(time.time())
    if pid:
        owner[key] = int(pid)
    owner.update(extra)
    tmp = claim.with_name(f"{claim.name}.tmp")
    try:
        tmp.write_text(json.dumps(owner, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, claim)
    except OSError:
        # Not fatal: the claim then carries no pid, and only the 300 s grace protects it from a
        # sweep. Loud enough to explain a duplicate delivery later (review finding, damko
        # 2026-10-23).
        logger.warning("dm queue: could not annotate the claim %s", claim.name, exc_info=True)


def _publish_no_clobber(tmp: Path, target: Path) -> bool:
    """Give the queue a file only when that name is still free. True when the file is now ``target``.

    ``os.link`` is an atomic test-and-create inside one directory, so a restore can never overwrite
    an envelope a runner re-parked between its existence check and its write, attempts included
    (review finding, damko 2026-10-23). False means the name is taken, or the disk refused.
    """
    try:
        os.link(tmp, target)
    except FileExistsError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False
    except OSError:
        logger.error("dm queue: could not publish %s", target.name, exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False
    try:
        _fsync_dir(target.parent, strict=True)  # the name is the commit: make the entry durable too
    except OSError:
        # An entry that may not survive a power loss is not a published envelope. Undo it, so the
        # caller keeps the claim it would otherwise delete: a claim plus a payload is recoverable,
        # a payload with no record is a lost message (review finding, damko 2026-10-23).
        logger.error("dm queue: could not sync %s; keeping the record, not the entry",
                     target.parent, exc_info=True)
        with contextlib.suppress(OSError):
            target.unlink()
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False
    with contextlib.suppress(OSError):
        tmp.unlink()
    return True


def _restore_claim(claim: Path, envelope: Path, record: dict) -> None:
    """Put the envelope back after a failed spawn — unless a newer envelope already owns it.

    Writing the stale copy back would clobber an envelope a runner re-parked in the meantime,
    attempt count included, so the write is a link that fails when the name is taken (FORK FIX,
    damko 2026-10-23).
    """
    tmp = envelope.with_name(f"{envelope.name}.tmp")
    try:
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        _fsync_file(tmp, strict=True)       # the published bytes must survive a power loss
    except OSError:
        logger.error("dm queue: could not restore %s after a failed spawn", envelope.name,
                     exc_info=True)
        return                              # the claim stays: the next sweep retries the restore
    if _publish_no_clobber(tmp, envelope):
        with contextlib.suppress(OSError):
            claim.unlink()
        return
    if envelope.exists():
        with contextlib.suppress(OSError):
            claim.unlink()                  # the newer envelope owns it; this claim must not compete


def _sweep_claims(profile_dir: Path) -> None:
    """Finish or repair the claims an earlier drain left behind.

    A claim its runner marked ``delivered`` is deleted with any payload still on disk; a claim
    whose envelope is back in the queue lost a re-park race and is deleted; a claim whose payload
    vanished with no delivery recorded is logged as a loss, never read as a success; a claim whose
    owners are all dead is restored to the queue with ``attempts`` incremented, so a runner that
    dies at boot retries instead of orphaning the payload (FORK ADDITION, damko 2026-10-23).
    """
    for claim in sorted(profile_dir.glob(f"*{_DM_PAYLOAD_SUFFIX}{_DM_CLAIM_SUFFIX}")):
        record = _read_json_dict(claim)
        payload = Path(str(claim)[: -len(_DM_CLAIM_SUFFIX)])
        delivery_id = str(record.get("delivery_id") or "")
        if record.get("delivered"):
            for path in (payload, claim):
                with contextlib.suppress(OSError):
                    path.unlink()       # delivered: a runner that died before tidying up
            continue
        if not delivery_id:
            with contextlib.suppress(OSError):
                claim.unlink()          # unusable: no envelope can be rebuilt from it
            continue
        if not payload.exists():
            # No delivery marker and no payload: name the loss instead of reading the absence as
            # success, or a payload deleted on a failed attempt vanishes without a trace
            # (FORK FIX, damko 2026-10-23).
            _watchdog_log("queue_dropped",
                          dict(record, detail="payload vanished before delivery was recorded"))
            with contextlib.suppress(OSError):
                claim.unlink()
            continue
        envelope = profile_dir / f"{delivery_id}.json"
        if envelope.exists():
            with contextlib.suppress(OSError):
                claim.unlink()          # already back in the queue; the envelope wins
            continue
        if _claim_owner_alive(claim, record):
            continue
        attempts = int(record.get("attempts") or 1) + 1
        if attempts > _DM_QUEUE_MAX_ATTEMPTS:
            _watchdog_log("queue_dropped", dict(record, attempts=attempts,
                                               detail=f"runner died before delivering; "
                                                      f"{attempts - 1} attempts used"))
            for path in (payload, claim):
                with contextlib.suppress(OSError):
                    path.unlink()
            continue
        restored = {k: v for k, v in record.items() if k not in ("runner_pid", "claimed_at")}
        restored.update(attempts=attempts, payload=str(payload), queued_at=int(time.time()),
                        detail=f"previous runner (pid {record.get('runner_pid') or 'unknown'}) "
                               f"exited before delivering")
        tmp = profile_dir / f"{delivery_id}.json.tmp"
        try:
            tmp.write_text(json.dumps(restored, indent=2, ensure_ascii=False), encoding="utf-8")
            _fsync_file(tmp, strict=True)   # the published bytes must survive a power loss
        except OSError:
            logger.error("dm queue: could not restore claim %s", claim.name, exc_info=True)
            continue
        if not _publish_no_clobber(tmp, envelope):
            if envelope.exists():
                with contextlib.suppress(OSError):
                    claim.unlink()          # a re-park won the race; its envelope is the live one
            continue
        with contextlib.suppress(OSError):
            claim.unlink()
        _watchdog_log("queue_requeued", restored)


def _delivery_profile(argv: list[str]) -> str:
    """Profile a local delivery argv targets (``hermes -p <name> ...``), else ''."""
    return str(argv[2]) if len(argv) >= 3 and argv[1] == "-p" else ""


def _holder_detail(pid: int, elapsed: float) -> str:
    """One line a sender can read: queued behind pid N, running for Nm Ns."""
    from tools.bot_relay import format_elapsed

    if pid:
        return f"queued behind a live turn (pid {pid}, {format_elapsed(elapsed)})"
    return "queued behind a live turn (holding pid not recorded)"


def _watchdog_log(event: str, record: dict) -> None:
    """Append one JSON line to <hermes>/logs/dm-watchdog.log — evidence, never fatal."""
    try:
        log_dir = Path(_default_home()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event,
                           "delivery_id": record.get("delivery_id"),
                           "profile": record.get("profile"), "reason": record.get("reason"),
                           "detail": record.get("detail"), "holder": record.get("holder"),
                           "attempts": record.get("attempts")},
                          ensure_ascii=False)
        with open(log_dir / _DM_WATCHDOG_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # pragma: no cover — logging must never break delivery
        logger.debug("dm watchdog log write failed", exc_info=True)


def _fsync_file(path: Path, *, strict: bool = False) -> None:
    """Force a queued file to the disk before its sender is told the queue holds it.

    A rename alone survives a process crash but not a power loss, so the bytes are flushed first
    and the rename is the commit (FORK ADDITION, damko 2026-10-23). ``strict`` re-raises: the
    caller then refuses the delivery instead of promising a queue place to bytes the disk never
    took (review finding, damko 2026-10-23).
    """
    sink = contextlib.nullcontext() if strict else contextlib.suppress(OSError)
    with sink:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_dir(path: Path, *, strict: bool = False) -> None:
    """A rename is only durable once its directory entry is synced too.

    ``strict`` asks the caller to hear about a failed sync: the queue treats "the name may not
    survive a power loss" as "not queued", which is the only safe reading when the payload would
    otherwise have no record at all (review finding, damko 2026-10-23).
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        if strict:
            raise
        logger.warning("dm queue: could not sync %s", path, exc_info=True)


def _enqueue_delivery(profile: str, argv: list[str], dm_file: str, author: Optional[dict], *,
                      reason: str, detail: str, holder: Optional[dict] = None) -> dict:
    """Copy the payload into the durable queue and write its envelope atomically.

    Re-parking a payload the queue already owns (the runner of a queued delivery lost the turn
    window again) keeps the delivery id the sender was told and increments ``attempts``, so an
    immediate re-drain cannot spin forever on a provider that hangs every time, and
    ``_DM_QUEUE_MAX_ATTEMPTS`` bounds the retries (FORK ADDITION, damko 2026-10-23).
    """
    from tools.bot_relay import turn_lock_holder

    claim = _claim_path(dm_file)            # only a payload a drain claimed has one
    previous = _read_json_dict(claim)
    delivery_id = str(previous.get("delivery_id") or "") or _dm_delivery_id(dm_file)
    qdir = _queue_dir(profile)
    payload = qdir / f"{delivery_id}{_DM_PAYLOAD_SUFFIX}"
    attempts = int(previous.get("attempts") or 0) + 1
    if attempts > _DM_QUEUE_MAX_ATTEMPTS:
        # Terminal: nobody will run this payload again, so drop its bytes and log the loss here
        # instead of leaving a file that no envelope points at (FORK FIX, damko 2026-10-23).
        for path in (payload, claim):
            with contextlib.suppress(OSError):
                path.unlink()
        _watchdog_log("queue_dropped", dict(previous, attempts=attempts,
                                           detail="attempt limit reached; message not delivered"))
        raise RuntimeError(
            f"this delivery already used {attempts - 1} attempts (the profile's turn window kept "
            f"hanging on it); the queue limit is {_DM_QUEUE_MAX_ATTEMPTS}")
    source = Path(dm_file)
    data = source.read_bytes()
    if source.resolve() != payload.resolve():  # a re-park hands the queue its own file back
        tmp_payload = qdir / f"{delivery_id}{_DM_PAYLOAD_SUFFIX}.tmp"
        tmp_payload.write_bytes(data)
        _fsync_file(tmp_payload, strict=True)   # a lost message is worse than a refused one
        os.replace(tmp_payload, payload)
    if holder is None:  # the refusal already knows the holder; only probe when it does not
        holder_pid, holder_elapsed = turn_lock_holder(_lock_root(), profile)
        holder = {"pid": holder_pid, "elapsed_seconds": round(holder_elapsed, 1)}
    record = {"delivery_id": delivery_id, "profile": profile, "argv": list(argv),
              "author": author, "payload": str(payload), "queued_at": int(time.time()),
              "reason": reason, "detail": detail, "attempts": attempts, "holder": holder}
    envelope = qdir / f"{delivery_id}.json"
    tmp = qdir / f"{delivery_id}.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    _fsync_file(tmp, strict=True)        # the bytes are durable before the rename commits them
    os.replace(tmp, envelope)
    _fsync_dir(qdir, strict=True)        # and so is the directory entry the rename created
    # The envelope is the live record now, and the claim this re-park superseded is left alone on
    # purpose: ``_sweep_claims`` deletes any claim whose envelope is back in the queue, so the drain
    # that runs next — the one this park starts — clears it. Touching it here was a race: moving the
    # claim to judge it handed a live drain's own claim out of the way, and the payload behind it
    # was reachable only through an envelope that another process could already have claimed
    # (review finding, damko 2026-10-23).
    return record


def _queue_instead_of_drop(profile: str, argv: list[str], dm_file: str, author: Optional[dict], *,
                           reason: str, detail: str, holder: Optional[dict] = None) -> int:
    """Park the delivery and tell the sender it is queued. Never claims delivery."""
    try:
        record = _enqueue_delivery(profile, argv, dm_file, author, reason=reason, detail=detail,
                                   holder=holder)
    except Exception as exc:
        print(json.dumps({"status": "refused", "reason": reason, "to": profile,
                          "error": f"{detail}; queueing the payload failed too: {exc}. "
                                   "The message was NOT delivered."}))
        return 1
    _watchdog_log("queued", record)
    print(json.dumps({"status": "queued", "reason": reason, "to": profile,
                      "delivery_id": record["delivery_id"],
                      "holder": record["holder"],
                      "detail": f"{detail} — NOT delivered yet. It is parked in the durable queue; "
                                "the queue is drained right after every DM turn into that profile "
                                "(and immediately when its turn window is already free). "
                                "Do not resend.",
                      "queued_file": record["payload"]}))
    return 0


def _park_and_drain(profile: str, argv: list[str], dm_file: str, author: Optional[dict], *,
                    reason: str, detail: str, holder: dict) -> int:
    """Park the payload, then hand the window it just freed to the queue.

    Draining here is what makes a parked message actually arrive: nothing called
    ``drain_queued_dms`` after an enqueue, so an envelope written behind a turn that had just
    ended sat in the queue until an unrelated DM into the same profile (FORK FIX, damko
    2026-10-23). A failed park still drains, so the sweep can repair or drop a stale claim.
    """
    code = _queue_instead_of_drop(profile, argv, dm_file, author, reason=reason, detail=detail,
                                  holder=holder)
    if profile:
        _drain_after_turn(profile)
    return code


def _spawn_queued_runner(record: dict, profile_dir: Path) -> Optional[int]:
    """Detached runner for a queued envelope: same argv, the queued payload as the DM file.

    Returns the runner's pid so the claim records who owns the payload; a runner that dies before
    delivering leaves a claim with a dead pid, which the next sweep hands back to the queue.
    """
    argv = list(record.get("argv") or [])
    if not argv:
        raise ValueError("queued envelope has no argv")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--run-delivery"]
    if record.get("author"):
        cmd += ["--author", json.dumps(record["author"], separators=(",", ":"))]
    cmd += ["query-file", str(record["payload"]), *argv]
    with open(profile_dir / "drain.log", "ab") as fh:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=fh,
                                start_new_session=True,
                                cwd=str(Path(__file__).resolve().parent.parent))
    return proc.pid


def drain_queued_dms(profile: Optional[str] = None) -> int:
    """Start a runner for every queued payload whose profile lock is free. Returns how many started.

    Each envelope is claimed by atomic rename before its runner starts, so two concurrent drains
    can never start two runners for one envelope, and a runner that fails to start puts the
    envelope straight back. A claim left by a runner that died is repaired by ``_sweep_claims``
    (FORK ADDITION, damko 2026-10-23)."""
    from tools.bot_relay import turn_lock_busy

    base = _relay_base() / _DM_QUEUE_DIRNAME
    if not base.is_dir():
        return 0
    started = 0
    for profile_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if profile and profile_dir.name != profile:
            continue
        _sweep_claims(profile_dir)
        for envelope in sorted(profile_dir.glob("*.json")):
            record = _read_json_dict(envelope)
            if not record:
                continue
            payload = str(record.get("payload") or "")
            if not payload or not Path(payload).exists():
                _watchdog_log("queue_dropped", dict(record, detail="payload file missing"))
                with contextlib.suppress(OSError):
                    envelope.unlink()
                continue
            if int(record.get("attempts") or 1) > _DM_QUEUE_MAX_ATTEMPTS:
                _watchdog_log("queue_dropped", record)
                for path in (Path(payload), envelope):
                    with contextlib.suppress(OSError):
                        path.unlink()
                continue
            if turn_lock_busy(_lock_root(), profile_dir.name):
                continue                      # the window is busy: the payload stays queued
            claim = _claim_envelope(envelope, record)
            if claim is None:
                continue                      # another drain claimed it first
            # The file just renamed onto the claim is the truth: the record read above may predate a
            # re-park that raised ``attempts`` (FORK FIX, damko 2026-10-23).
            fresh = _read_json_dict(claim) or record
            _note_claim_owner(claim, fresh, os.getpid(), key="drain_pid")
            try:
                pid = _spawn_queued_runner(fresh, profile_dir)
            except Exception as exc:
                logger.error("dm queue: could not start %s: %s", envelope.name, exc)
                _restore_claim(claim, envelope, fresh)
                continue
            _note_claim_owner(claim, fresh, pid)
            _watchdog_log("queue_drained", dict(fresh, claimed_file=str(claim)))
            started += 1
    return started


def _drain_after_turn(profile: str) -> None:
    """Hand a now-free turn window to a queued payload (best effort, never fatal)."""
    try:
        drain_queued_dms(profile)
    except Exception:  # pragma: no cover
        logger.debug("dm queue drain failed", exc_info=True)


def _delivery_lock(argv: list[str], *, stdin_file: bool):
    """Per-profile turn lock for a LOCAL teammate delivery: local and relay deliveries
    into one profile both run a Bot Chat turn here, so the turn window is serialized on
    ``tools.bot_relay``'s cross-process lock. Peer transports (stdin mode) are locked
    on the remote gateway by its own deliver path.

    See #93091.
    """
    # Match the CLI element by basename: argv[0] may be an absolute venv path
    # (service contexts lack PATH) and carries .exe on Windows; split on both separators.
    # Split on both separators so the shape matches regardless of which platform built the argv. See #93590.
    cli = (argv[0] if argv else "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if stdin_file or len(argv) < 3 or cli not in ("hermes", "hermes.exe") or argv[1] != "-p":
        return contextlib.nullcontext()
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import acquire_turn_lock

    return acquire_turn_lock(_hermes_root(Path(_default_home())), argv[2])


def _lease_refused(proc) -> bool:
    """Whether a finished CLI turn was refused because another surface holds the session
    lease. The CLI writes ``hermes-refusal-reason: SESSION_NOT_OWNED`` on stderr; older
    CLIs without the marker get the historical wording fallback, and a code wins over
    prose (including unknown codes from newer CLIs)."""
    stderr_text = proc.stderr or ""
    reason = next((line.removeprefix("hermes-refusal-reason: ").strip()
                   for line in stderr_text.splitlines()
                   if line.startswith("hermes-refusal-reason: ")), None)
    refused_not_owned = (reason == "SESSION_NOT_OWNED" if reason is not None
                         else "already has a live owner" in stderr_text)
    return refused_not_owned and proc.returncode != 0


def _run_local_turn(argv: list[str], dm_file: str, *, env: Optional[dict[str, str]] = None) -> int:
    """One Bot Chat turn via ``--query-file`` (plus one policy-gated retry); re-emits
    the transport's streams and returns its exit code. Transient failures re-run the
    same session; a context_overflow re-run lets the retried turn's pre-API compaction
    compact the transcript first (no fresh session is ever minted). Auth/quota/config never retry.
    A lease refusal (``SESSION_NOT_OWNED``) is retried on ``_REFUSAL_RETRY_WAITS`` first — the
    refused turn never ran, so replaying the same payload cannot double-deliver
    (FORK ADDITION, damko 2026-10-14)."""

    def _turn(turn_env=env):
        # Watchdog instead of a blocking run(): a turn that stops producing output (a provider socket
        # with no read deadline) must not hold the profile lock until TCP dies. FORK ADDITION 2026-10-23.
        outcome = _run_turn_watched([*argv, "--query-file", dm_file],
                                    idle_seconds=dm_idle_seconds(), env=turn_env,
                                    turn_cap_seconds=dm_turn_cap_seconds())
        if outcome.killed_idle:
            raise TurnIdleExceeded(outcome)
        if outcome.timed_out:
            raise TurnCapExceeded(outcome)
        return outcome

    def _attempt():
        proc = _turn()
        if proc.returncode != 0:
            from tools.bot_failure_reasons import RETRY_NONE, classify_agent_error, retry_action, turn_failure_text
            from tools.bot_relay import retry_turn_env

            # The re-run replays the same session and payload; the failed attempt already persisted the
            # user row, so the retried process is told to resume it (RESUME_UNANSWERED_TURN_ENV).
            if retry_action(classify_agent_error(turn_failure_text(proc.stdout, proc.stderr))) != RETRY_NONE:
                proc = _turn(retry_turn_env(env))
        return proc

    proc = _attempt()
    retries, waited = 0, 0.0
    for wait_seconds in _REFUSAL_RETRY_WAITS:
        if not _lease_refused(proc):
            break
        time.sleep(wait_seconds)
        retries, waited = retries + 1, waited + wait_seconds
        proc = _attempt()
    if _lease_refused(proc):
        # A lease held by another surface — usually the other leg of a round trip still
        # winding down — is not a busy agent. The vendor dropped the message here (#100523);
        # FORK ADDITION, damko 2026-10-14: report honestly after the bounded retry above.
        who = argv[argv.index("-p") + 1] if "-p" in argv[:-1] else "the teammate"
        detail = f"{retries + 1} attempts over ~{round(waited)}s" if retries else "one attempt"
        print(json.dumps({
            "error": f"Delivery failed: @{who}'s Bot Chat stayed held by another surface for the "
                     f"whole retry window ({detail}), so your message was NOT delivered. The holder "
                     "is usually the other leg of a round trip still winding down; re-send once it frees.",
            "reason": "target_busy",
        }))
        return 1
    # Re-emit the transport's streams: stdout is the reply text the
    # completion notification carries back to the sending agent. A successful bare
    # silence marker is a delivery decision (same rule as the gateway and the live
    # Bot Chat completion): the turn stays in the target's transcript, the sender
    # never sees the marker as prose.
    from gateway.response_filters import is_intentional_silence_response
    reply = proc.stdout or ""
    if proc.returncode == 0 and is_intentional_silence_response(reply):
        reply = ""
    for stream, text in ((sys.stdout, reply), (sys.stderr, proc.stderr)):
        if text:
            stream.write(text)
            stream.flush()
    return proc.returncode


def _live_intent_file(dm_file: str | os.PathLike) -> str:
    """The pinned live-delivery intent beside a DM file (the cache sweep globs ``*.live.json``)."""
    return f"{os.fspath(dm_file)}.live.json"


def _dm_delivery_id(dm_file: str | os.PathLike) -> str:
    """One delivery id per DM file: the dispatch ack, the live-owner intent and every retry
    of the runner derive it the same way, so the sender can correlate all of them."""
    return hashlib.sha256(str(Path(dm_file).resolve()).encode()).hexdigest()


def _admit_live_dm(profile_home: Path | None, dm_file: str, author: Optional[dict] = None) -> dict | None:
    """Pin intent before admission; retries may inspect, never change transport."""
    from tools.bot_live_delivery import deliver_to_live_owner, find_canonical_live_owner, read_delivery_result
    from utils import fsync_directory

    intent: dict[str, Any]
    intent_path = Path(_live_intent_file(dm_file))
    if intent_path.exists():
        intent = json.loads(intent_path.read_text(encoding="utf-8-sig"))
    else:
        assert profile_home is not None
        owner = find_canonical_live_owner(profile_home)
        if owner is None:
            return None
        intent = dict(owner=owner, message=Path(dm_file).read_text(encoding="utf-8-sig"),
                      delivery_id=_dm_delivery_id(dm_file),
                      **({"author": author} if author else {}))
        try:
            fd = os.open(intent_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            intent = json.loads(intent_path.read_text(encoding="utf-8-sig"))
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(intent, stream)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_directory(intent_path.parent)
    home = intent["owner"]["profile_home"]
    record = read_delivery_result(home, intent["delivery_id"])
    if record is None:
        record = deliver_to_live_owner(home, intent["owner"], intent["message"],
                                       delivery_id=intent["delivery_id"], author=intent.get("author"))
    return record


def _wait_live_dm(home: str, delivery_id: str, *, dm_file: str | os.PathLike | None = None) -> int:
    from tools.bot_failure_reasons import RUNTIME_OFFLINE
    from tools.bot_live_delivery import await_delivery, cancel_queued_delivery, owner_holds_delivery

    deadline = time.monotonic() + _LIVE_WAIT_MAX_SECONDS
    missed = 0
    while True:
        record = await_delivery(home, delivery_id, max(0.0, min(_LIVE_WAIT_SECONDS, deadline - time.monotonic())))
        if record is None or record["status"] not in ("queued", "claimed") or time.monotonic() >= deadline:
            break
        try:
            held = owner_holds_delivery(home, record)
        except Exception:
            logger.debug("Could not check live DM owner", exc_info=True)
            held = None
        if held:
            missed = 0
            continue
        if held is False and record["status"] == "queued":
            # Under the same mailbox lock as claim, either cancellation wins or
            # the claim wins and we must keep waiting for its outcome.
            record = cancel_queued_delivery(
                home, delivery_id, error="Live Bot Chat owner is no longer available",
                reason=RUNTIME_OFFLINE)
            if record is None or record["status"] != "claimed":
                break
            missed = 0
            continue
        # A vanished consumer may leave a permanent claim, and an owner we cannot look up
        # may be gone: allow one more window for an in-flight turn, then release the runner.
        missed += 1
        if missed > 1:
            break
    status = record["status"] if record else "ambiguous"
    payload = {key: record[key] for key in ("reply", "error", "reason") if record and record.get(key)}
    payload.update(status=status, delivery_id=delivery_id)
    if status in ("queued", "claimed", "ambiguous"):
        payload["detail"] = "Delivery remains pending or its outcome is unknown. Do not resend; receipt is retained."
    elif status == "settled" and dm_file is not None:
        # The intent carries the message plaintext so a retry can replay the SAME delivery id;
        # once the owner settled it nothing retries, so it goes along with the dm file (same
        # plaintext) — the live branch returns before _run_delivery's own unlink.
        _unlink_dm_file(_live_intent_file(dm_file))
        _unlink_dm_file(str(dm_file))
    print(json.dumps(payload))
    return 0 if status in ("settled", "queued", "claimed") else 1


def _local_delivery_home(argv: list[str]) -> Path | None:
    cli = (argv[0] if argv else "").rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if len(argv) < 3 or cli not in ("hermes", "hermes.exe") or argv[1] != "-p":
        return None
    from tools.bot_mode_probe import _hermes_root, _roster

    return dict(_roster(_hermes_root(Path(_default_home())))).get(argv[2])


def _live_outcome_unknown(dm_file: str, cause: object) -> str:
    """The runner's stdout when a live admission may have happened but its outcome is unknown."""
    return json.dumps({"status": "ambiguous", "delivery_id": _dm_delivery_id(dm_file),
                       "error": f"Live admission outcome unknown: {cause}. Do not resend.",
                       "evidence_file": dm_file})


def _runner_argv(args: list[str]) -> tuple[Optional[str], str, str, list[str]] | None:
    """Split ``--run-delivery [--author <json>] <mode> <dm_file> <argv…>`` into
    ``(author_json, mode, dm_file, argv)``; None when malformed. Stdlib only: the boot-failure
    report runs it when no Hermes import is available."""
    if args[:1] != ["--run-delivery"]:
        return None
    rest, author = args[1:], None
    if rest[:1] == ["--author"]:
        if len(rest) < 2:
            return None
        author, rest = rest[1], rest[2:]
    if len(rest) < 2 or rest[0] not in ("stdin", "query-file"):
        return None
    return author, rest[0], rest[1], rest[2:]


def _run_delivery(argv: list[str], dm_file: str, *, stdin_file: bool,
                  profile_home: Path | None = None, author: Optional[dict] = None) -> int:
    """Route to the live owner before attempting a CLI transport. Live deliveries
    retain their intent/payload and immutable receipt; only CLI/peer payloads are
    removed after consumption. The CLI turn window holds the profile lock, so two
    deliveries into one profile queue; when the bounded wait expires ('target_busy') or the turn
    itself is stopped for producing no output ('idle_timeout') the payload is parked in the durable
    queue under <relay>/queue/<profile>/ and drained at once, then by every free turn window after
    that; each claim is atomic and each re-park increments ``attempts`` up to
    ``_DM_QUEUE_MAX_ATTEMPTS`` (FORK ADDITION, damko 2026-10-23; upstream #122370). It is never
    deleted and never reported as delivered.
    ``author`` rides to the child as HERMES_TURN_AUTHOR; ``hermes peer dm`` forwards it in the request body.

    Local (query-file) turns get one policy-gated retry (#93091 item 5): transient failures re-run the same
    session; a context_overflow re-run lets the retried turn's pre-API compaction pass compact the Bot Chat
    transcript first (agent/conversation_loop.py) — the sanctioned compression lever; no fresh session is
    ever minted. Auth/quota/config failures never retry. Peer transports (stdin mode) retry on their own
    gateway's deliver path, not here. A lease refusal (``SESSION_NOT_OWNED``) is retried on
    ``_REFUSAL_RETRY_WAITS`` before the typed refusal is reported (FORK ADDITION, damko 2026-10-14).
    """
    # The live consumer owns turn admission; never compete for its CLI lease.
    if not stdin_file:
        home = profile_home or _local_delivery_home(argv)
        if home is not None or os.path.exists(_live_intent_file(dm_file)):
            try:
                record = _admit_live_dm(home, dm_file, author)
            except Exception as exc:
                print(_live_outcome_unknown(dm_file, exc))
                return 1
            if record is not None:
                return _wait_live_dm(record["profile_home"], record["delivery_id"], dm_file=dm_file)
    profile = _delivery_profile(argv)
    # A payload a drain claimed is the queue's own file: the runner must not delete it unless the
    # turn really delivered, or a failed attempt loses the only copy (FORK FIX, damko 2026-10-23).
    queued_payload = _claim_path(dm_file).exists()
    delivered = False
    # A park the queue refused (a write or sync failure) means the queue does not hold these bytes:
    # the finally below must keep the original, or the refusal deletes the only copy (review
    # finding, damko 2026-10-23).
    park_refused = False
    try:
        from tools.bot_relay import TurnBusyError, delivery_env, format_elapsed

        env = delivery_env(author, profile_home if not stdin_file else None)
        try:
            with _delivery_lock(argv, stdin_file=stdin_file):
                if not stdin_file:
                    code = _run_local_turn(argv, dm_file, env=env)
                else:
                    # Keep the file open until the transport exits; cleanup occurs
                    # after subprocess.run returns, not merely after stdin reaches EOF.
                    with open(dm_file, "r", encoding="utf-8-sig") as stream:
                        # Passing the file descriptor as stdin bypasses the BOM-aware decoder.
                        code = subprocess.run(argv, input=stream.read().encode("utf-8"),
                                              check=False, env=env).returncode
        except TurnBusyError as exc:
            # Was: payload deleted and 'target_busy' returned, so the message vanished (FORK FIX
            # 2026-10-23, upstream #122370). Now: durable queue, sender told it is queued, and the
            # queue drained at once — the window may have freed while this delivery gave up.
            code = _park_and_drain(
                profile, argv, dm_file, author, reason="target_busy",
                detail=_holder_detail(exc.holder_pid, exc.holder_elapsed),
                holder={"pid": exc.holder_pid, "elapsed_seconds": round(exc.holder_elapsed, 1)})
            # A refused park (a write or sync failure) did not take the bytes anywhere: the
            # finally must keep this file, or the refusal deletes the only copy (review finding,
            # damko 2026-10-23).
            park_refused = code != 0
            return code
        except TurnIdleExceeded as exc:
            # TurnIdleExceeded (no progress) and TurnCapExceeded (past the ceiling) land here.
            detail = (
                f"the turn this message waited for (pid {exc.outcome.pid}) made no progress for "
                f"{format_elapsed(exc.outcome.idle_seconds)} (last tool: "
                f"{exc.outcome.last_tool or 'unknown'}) and was stopped"
                if exc.reason == "idle_timeout" else
                f"the turn this message waited for (pid {exc.outcome.pid}) was still running after "
                f"{format_elapsed(exc.outcome.elapsed)} and was stopped at the turn ceiling")
            # The turn was stopped, so this profile's window is free right now: park the payload and
            # drain at once, or the message waits for an unrelated turn into the same profile.
            code = _park_and_drain(
                profile, argv, dm_file, author, reason=exc.reason, detail=detail,
                # Name the turn the watchdog stopped, not a blank holder: the sender's reply and
                # the queue record are the only evidence the killed pid ever existed.
                holder={"pid": exc.outcome.pid,
                        "elapsed_seconds": round(exc.outcome.elapsed, 1)})
            park_refused = code != 0
            return code
        # Turn window is free again: hand it to whatever queued behind this turn.
        if profile:
            _drain_after_turn(profile)
        delivered = code == 0
        return code
    finally:
        if queued_payload:
            # The queue owns this payload: only a delivered turn may remove it. A failed attempt
            # leaves payload and claim for ``_sweep_claims``, which returns the message to the queue
            # a bounded number of times and logs the drop — deleting here would destroy the only
            # copy while nothing reported a loss (FORK FIX, damko 2026-10-23).
            if delivered:
                claim = _claim_path(dm_file)
                _note_claim_owner(claim, {}, None, delivered=True)  # evidence before the bytes go
                _unlink_dm_file(dm_file)
                with contextlib.suppress(OSError):
                    claim.unlink()
        elif not park_refused:
            # Only a delivery the queue never tried to hold may be deleted here. A refused park
            # leaves this file as the only copy of the message (review finding, damko 2026-10-23).
            # An exception that escapes still deletes it, as it always has: the wrapper prints a
            # typed refusal to the sender and upstream pins that contract in
            # tests/tools/test_bot_mode_dm.py::test_delivery_main_maps_launch_exception_to_one_and_unlinks.
            _unlink_dm_file(dm_file)


def _delivery_command(argv: list[str], dm_file: str, *, stdin_file: bool,
                      profile_home: Path | None = None, author: Optional[dict] = None) -> str:
    """Build an argv-safe command for the cleanup-owning background runner:
    ``--run-delivery [--author <json>] <mode> <dm_file> [--profile-home <path>] <argv...>``."""
    runner_argv = [sys.executable, str(Path(__file__).resolve()), "--run-delivery",
                   "stdin" if stdin_file else "query-file", dm_file]
    if profile_home is not None:
        runner_argv.extend(["--profile-home", str(Path(profile_home).resolve())])
    runner_argv.extend(argv)
    if sys.platform == "win32":
        # The tracked local backend uses Git Bash on native Windows: forward slashes keep drive
        # paths executable there; backslash paths are parsed as command names (exit 127).
        runner_argv = [part.replace("\\", "/") for part in runner_argv]
    if author:
        # Inserted after the slash rewrite: JSON escapes are backslashes too.
        runner_argv[3:3] = ["--author", json.dumps(author, separators=(",", ":"))]
    return shlex.join(runner_argv)


def _start_delivery(argv: list[str], content: str, label: str, *, stdin_file: bool,
                    task_id: Optional[str], agent: Any, profile_home: Path | None = None,
                    author: Optional[dict] = None) -> str:
    """Create a DM file and transfer its cleanup ownership to the runner."""
    dm_file = _write_dm_file(content)
    if profile_home is not None:
        try:
            record = _admit_live_dm(profile_home, dm_file, author)
        except Exception as exc:
            return json.dumps({"status": "ambiguous", "delivery_id": _dm_delivery_id(dm_file),
                "error": f"Live delivery admission could not be confirmed: {exc}. Do not resend.",
                "evidence_file": dm_file})
        if record is not None:
            command = _delivery_command(argv, dm_file, stdin_file=False, profile_home=profile_home, author=author)
            notification = json.loads(_spawn_delivery(command, label, task_id=task_id, agent=agent))
            result = dict(status=record["status"], delivery_id=record["delivery_id"], to=label,
                          detail="Durably queued for the live Bot Chat owner. Do NOT wait or resend; finish your turn.")
            if notification.get("error"):
                result["notification_error"] = notification["error"]
            elif notification.get("process_id"):
                result["process_id"] = notification["process_id"]
                result["reply_delivery"] = notification.get("reply_delivery", "notification")
                if result["reply_delivery"] == "poll":
                    # Same runner, same stdout-borne reply (#101142): a non-push sender must get
                    # the poll instruction here too, not 'finish your turn'.
                    result["detail"] = f"Durably queued for the live Bot Chat owner. Do NOT resend. {notification['detail']}"
            return json.dumps(result)
    try:
        command = _delivery_command(argv, dm_file, stdin_file=stdin_file, profile_home=profile_home, author=author)
    except BaseException:
        _unlink_dm_file(dm_file)
        raise
    return _spawn_delivery(command, label, dm_file=dm_file, task_id=task_id, agent=agent)


def _spawn_delivery(command: str, label: str, *, dm_file: Optional[str] = None, delivery_id: Optional[str] = None,
                    task_id: Optional[str], agent: Any) -> str:
    """Launch the cleanup-owning runner and transfer file ownership on ack. ``dm_file``
    is None for relay deliveries (the waiter watches a reply file; envelope artifacts
    are owned/swept by ``tools/bot_relay.py``), which pass the envelope id as ``delivery_id``
    instead. The ack is ``queued`` + ``delivery_id`` like the live-owner branch: hand-off
    to a background process, never a delivery receipt."""
    transferred = False
    try:
        from tools.terminal_tool import terminal_tool

        raw = terminal_tool(command, background=True, notify_on_complete=True, task_id=task_id,
                            workdir=str(Path(__file__).resolve().parent.parent), _host_local=True,
                            _completion_output_chars=REPLY_COMPLETION_CHARS)
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = {}
        proc_id = parsed.get("session_id") or ""
        if parsed.get("error"):
            return _err(f"Delivery to {label} failed to start: {parsed['error']}")
        if parsed.get("status") == "pending_approval":
            # terminal_tool's approval gate answers with an EMPTY error and no session_id: the runner
            # never launched because nobody in this turn could approve its command.
            return _err(f"Delivery to {label} failed to start: its command needs terminal approval that nobody "
                        "in this turn can grant" + (", so nothing was sent. Approve it (or add it to "
                                                    "command_allowlist) and send again." if dm_file else "."))
        if not proc_id:
            return _err(f"Delivery to {label} failed to start: no process id returned")
        # From here the background runner owns the file (removed after the consumer finishes).
        transferred = True
        if parsed.get("notify_on_complete") is False:
            # terminal_tool refused the completion promise: this session (api_server, one-shot
            # runner) cannot receive an async completion, so the recipient's reply would never
            # be injected here (#101142). Say so and name the return path the surface supports.
            detail = (f"Message handed to a background delivery process for {label}, but THIS session "
                      "cannot receive completion notifications, so the reply will NOT arrive on its own. "
                      f"Before ending your turn, retrieve the outcome with process(action='wait', "
                      f"session_id='{proc_id}') — its output is the reply (relay it, attributed to that "
                      "agent) or the delivery failure (report it; the message was NOT delivered); "
                      "if wait returns status=timeout, call wait again until the process exits.")
            if _persist_reply_when_done(proc_id, agent):
                detail += (" Its outcome is also saved into this session's transcript as a delivery row "
                           "when the process exits, so it survives even if the turn ends first.")
        else:
            detail = (f"Message queued for {label}: this acknowledges the hand-off to a "
                      "background delivery process, not a delivery receipt — do NOT wait or poll. "
                      "Finish your turn now; that process's completion notification carries the "
                      "delivery outcome — the reply (relay it then, attributed to that agent) or "
                      "the delivery failure (report it; the message was NOT delivered).")
        return json.dumps({
            "status": "queued",
            "delivery_id": delivery_id or (_dm_delivery_id(dm_file) if dm_file else ""),
            "to": label,
            "reply_delivery": "poll" if parsed.get("notify_on_complete") is False else "notification",
            "detail": detail,
            "process_id": proc_id,
            "queued_at": int(time.time()),
        })
    except Exception as exc:
        logger.error("message_agent delivery spawn failed: %s", exc, exc_info=True)
        return _err(f"Delivery to {label} could not be started: {exc}")
    finally:
        if dm_file and not transferred:
            _unlink_dm_file(dm_file)


def _persist_reply_when_done(proc_id: str, agent: Any) -> bool:
    """#101142 durable leg. A non-push sender (api_server, one-shot runner) gets no completion
    notification, so once the tracked runner exits its stdout — the recipient's reply or the
    delivery failure — is appended to the sender's session transcript as a DELIVERY row
    (``display_kind="process_complete"``, the shape push surfaces persist for the same
    completion; mirrors gateway.wake.persist_delegation_delivery). A sender that already read
    the outcome via process(action='wait'/'log') is not told twice. Returns False (nothing
    armed) when the sender has no session transcript or the process is not tracked here."""
    from tools.process_registry import process_registry

    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    proc = process_registry.get(proc_id)
    if proc is None or not session_id or not callable(getattr(db, "append_message", None)):
        return False

    def _run() -> None:
        from tools.process_registry_notifications import (
            format_process_notification, process_completion_display_text,
        )

        proc._completion_event.wait()
        if process_registry.is_completion_consumed(proc_id):
            return
        evt = {"type": "completion", "session_id": proc_id, **process_registry._exit_snapshot(proc, "exited")}
        try:
            db.append_message(session_id, "user", content=format_process_notification(evt),
                              display_kind="process_complete",
                              display_metadata={"display_text": process_completion_display_text([evt])})
        except Exception as exc:
            logger.warning("message_agent: could not persist the reply of %s into session %s: %s",
                           proc_id, session_id, exc)

    threading.Thread(target=_run, name=f"message-agent-reply-{proc_id}", daemon=True).start()
    return True


def _wait_reply_main(reply_path: str, label: str, budget_seconds: str) -> int:
    """The relay reply waiter (``tools/bot_relay.waiter_command``): block until the sender-side
    reply file exists, print it as the completion notification the sender wakes on, exit 1 on a
    delivery error or when the budget runs out. Stdlib only: this runs as a background process
    from any bot turn, and the sender's completion notification is exactly its stdout."""
    try:
        deadline = time.time() + float(budget_seconds)
    except ValueError:
        return 2
    while time.time() < deadline:
        if os.path.exists(reply_path):
            with open(reply_path, encoding="utf-8-sig") as fh:
                d = json.load(fh)
            if d.get("error"):
                # Typed reason code rides ahead of the free text so the sender can branch on it
                # without parsing provider prose. See #93091.
                code = str(d.get("reason") or "").strip()
                tag = f" [reason: {code}]" if code else ""
                print(f"Delivery to {label} failed{tag}: {d['error']}")
                return 1
            print(f"Reply from {label}:")
            print(d.get("reply") or "(empty reply)")
            return 0
        # 250ms cadence: stat is cheap and a longer sleep is pure dead air.
        time.sleep(0.25)
    print(f"No reply from {label} within {budget_seconds}s. The message may still be delivered when "
          "the Desktop reconnects; do not resend blindly.")
    return 1


def _delivery_main(args: list[str]) -> int:
    """Runner entry for the argv ``_delivery_command`` and ``bot_relay.waiter_command`` build.
    Malformed argv exits 2 without touching the DM file."""
    if args[:1] == ["--wait-reply"]:
        return _wait_reply_main(*args[1:]) if len(args) == 4 else 2
    if args[:1] == ["--drain-queue"]:
        # Cron/healthcheck entry: start every queued delivery whose profile lock is free.
        started = drain_queued_dms(args[1] if len(args) > 1 else None)
        print(json.dumps({"status": "drained", "started": started}))
        return 0
    parsed = _runner_argv(args)
    if parsed is None:
        return 2
    author_json, mode, dm_file, argv = parsed
    author = None
    if author_json is not None:
        from agent.turn_author import parse_turn_author

        author = parse_turn_author(author_json)
        if author is None:
            return 2
    try:
        profile_home = None
        if len(argv) >= 2 and argv[0] == "--profile-home":
            profile_home, argv = Path(argv[1]), argv[2:]
        return _run_delivery(argv, dm_file, stdin_file=mode == "stdin", profile_home=profile_home, author=author)
    except Exception as exc:
        # Every refusal ships a typed reason on stdout so the completion notification carries it
        # back to the sender (#93091): 'target_busy' from the queue's bounded wait, otherwise the
        # same vocabulary-guarded classification the relay lane applies.
        from tools.bot_failure_reasons import delivery_failure_reason

        print(json.dumps({"error": str(exc), "reason": delivery_failure_reason(exc)}))
        return 1


# agent-context helpers (mirror system_prompt.py's resolution)


def _agent_home(agent: Any) -> str:
    """The calling agent's OWN home (session-db derived), not ambient env."""
    with contextlib.suppress(Exception):
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        if db_path:
            return str(Path(db_path).parent)
    return _default_home()


def _session_title(agent: Any) -> str:
    title = str(getattr(agent, "_session_title_hint", "") or "").strip()
    if title:
        return title
    with contextlib.suppress(Exception):
        sdb, sid = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
        if sdb and sid:
            return str(sdb.get_session_title(sid) or "").strip()
    return ""


if __name__ == "__main__":  # pragma: no cover - exercised as a background process
    # Spawned as a script with the sender's sys.executable, which under PM is the bare store
    # interpreter (dependencies are activated in-process, never inherited), so boot like every
    # entry point before the lazy Hermes imports. Run as a path, sys.path[0] is tools/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # Only the delivery lane boots: the reply waiter is stdlib only and its stdout is the
    # sender's wake-up, and any other argv exits 2 before a Hermes import.
    if sys.argv[1:2] in (["--run-delivery"], ["--drain-queue"]):
        try:
            import hermes_bootstrap
        except (Exception, SystemExit) as exc:
            # A pinned live intent means the sender may already have admitted this DM and was
            # told not to resend; a bare repair hint would read as "NOT delivered". Without an
            # intent nothing was handed over, so the plain failure is the truth. A relaunched
            # child (Windows) already printed its own outcome.
            booted = getattr(exc, "relaunched", False) or (isinstance(exc, SystemExit) and not exc.code)
            parsed = _runner_argv(sys.argv[1:])
            if not booted and parsed and os.path.exists(_live_intent_file(parsed[2])):
                print(_live_outcome_unknown(parsed[2], "the delivery runner could not activate "
                                                       "Hermes dependencies (see stderr)"))
            raise
    raise SystemExit(_delivery_main(sys.argv[1:]))
