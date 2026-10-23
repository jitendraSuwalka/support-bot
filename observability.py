"""
One structured log line per customer turn.

    {"event": "turn", "ts": "2026-10-24T09:12:03.512Z", "request_id": "…",
     "session_id": "…", "turn": 2, "behaviour": "answer", "escalated": false,
     "tools": ["search_knowledge_base"], "guards": ["citation_rewrite"],
     "warnings": [], "evidence_parts": 1, "latency_ms": 8412,
     "cost": {"usd": 0.0121, "input_tokens": 4211, …}, …}

Everything a reviewer, an on-call engineer or a cost report needs about a
turn is in that line, and nothing that identifies what the customer said:
the message and the answer are recorded as a length and the first sixteen
hex digits of a SHA-256 hash, unless the deployment switches log_content on.
A hash lets one turn be matched to a complaint — hash the customer's words
and look for it — without the log becoming a second copy of every
conversation. Lines go to stderr, the place a container platform collects
them from, and optionally to a JSONL file.

The agent itself does not log; ``run_logged_turn`` wraps a turn, so every
front end (the Gradio app, the HTTP API) logs the same thing the same way
and the agent stays testable without a logger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import agent
from config import Settings

LOGGER_NAME = "support_bot"
TURN_LOGGER = f"{LOGGER_NAME}.turns"
_HANDLER_MARK = "_support_bot_handler"

Runner = Callable[[list, dict], "agent.AgentResult"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fingerprint(text: str) -> dict:
    """What the log keeps of a piece of text when the text itself is not kept."""
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    }


def last_user_message(messages: list) -> str:
    return next(
        (
            m["content"]
            for m in reversed(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ),
        "",
    )


def user_turns(messages: list) -> int:
    return sum(1 for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str))


def turn_record(
    result: agent.AgentResult | None,
    *,
    messages: list,
    request_id: str,
    session_id: str,
    seconds: float,
    settings: Settings,
    error: BaseException | None = None,
) -> dict:
    """The log line for one turn, as a dictionary. ``result`` is None when the turn failed."""
    message = last_user_message(messages)
    record: dict = {
        "event": "turn",
        "ts": utc_now(),
        "request_id": request_id,
        "session_id": session_id,
        "turn": user_turns(messages),
        "agent_impl": settings.agent_impl,
        "latency_ms": round(seconds * 1000),
        "message": fingerprint(message),
    }
    if settings.log_content:
        record["message"]["text"] = message
    if result is None:
        record.update(
            behaviour="error",
            error={"type": type(error).__name__, "detail": str(error)[:500]} if error else None,
        )
        return record

    cost = result.cost()
    record.update(
        behaviour=result.behaviour,
        escalated=result.escalated,
        models=result.models,
        tools=list(result.tools_used),
        guards=result.guards(),
        warnings=[note for note in result.notes if note.startswith("WARNING")],
        evidence_parts=len(result.evidence),
        answer=fingerprint(result.answer),
        cost=cost.as_dict(),
    )
    if settings.log_content:
        record["answer"]["text"] = result.answer
    return record


def run_logged_turn(
    messages: list,
    state: dict,
    *,
    session_id: str,
    request_id: str | None = None,
    runner: Runner | None = None,
    settings: Settings | None = None,
) -> tuple[agent.AgentResult, dict]:
    """Run one turn and write its log line; return the result and the record.

    A turn that raises is logged as behaviour "error" with the exception's
    type, then the exception propagates: the caller decides what the
    customer sees, the log still says what happened.
    """
    runner = runner or agent.run_agent
    settings = settings or agent.SETTINGS
    request_id = request_id or uuid.uuid4().hex
    logger = logging.getLogger(TURN_LOGGER)
    started = time.perf_counter()
    try:
        result = runner(messages, state)
    except Exception as error:
        record = turn_record(
            None,
            messages=messages,
            request_id=request_id,
            session_id=session_id,
            seconds=time.perf_counter() - started,
            settings=settings,
            error=error,
        )
        logger.error("turn failed", extra={"turn": record})
        raise
    record = turn_record(
        result,
        messages=messages,
        request_id=request_id,
        session_id=session_id,
        seconds=time.perf_counter() - started,
        settings=settings,
    )
    level = logging.WARNING if record["warnings"] else logging.INFO
    logger.log(level, "turn", extra={"turn": record})
    return result, record


def structured(record: logging.LogRecord) -> dict | None:
    """The structured payload a log call carried, if any: a turn record
    (``extra={"turn": ...}``) or an HTTP request record (``extra={"fields": ...}``)."""
    for name in ("turn", "fields"):
        value = getattr(record, name, None)
        if isinstance(value, dict):
            return value
    return None


class JsonFormatter(logging.Formatter):
    """One JSON object per line. A structured record (a turn, an HTTP request)
    is written as it is; any other log message becomes {"ts", "level",
    "logger", "message"}."""

    def format(self, record: logging.LogRecord) -> str:
        turn = structured(record)
        if turn is not None:
            payload = {"ts": utc_now(), **turn, "level": record.levelname}
        else:
            payload = {
                "event": "log",
                "ts": utc_now(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


class TextFormatter(logging.Formatter):
    """A readable line for a terminal: the essentials of a turn, or the message."""

    def format(self, record: logging.LogRecord) -> str:
        turn = structured(record)
        if turn is not None and turn.get("event") == "request":
            return (
                f"{record.levelname} {turn['method']} {turn['path']} {turn['status']} "
                f"{turn['latency_ms']} ms request={str(turn['request_id'])[:8]}"
            )
        if turn is None:
            return f"{record.levelname} {record.name}: {record.getMessage()}"
        cost = (turn.get("cost") or {}).get("usd")
        return (
            f"{record.levelname} turn {turn.get('turn')} {turn.get('behaviour')} "
            f"{turn.get('latency_ms')} ms"
            + (f" ${cost:.4f}" if cost is not None else "")
            + (f" guards={','.join(turn['guards'])}" if turn.get("guards") else "")
            + f" session={str(turn.get('session_id'))[:8]}"
            + f" request={str(turn.get('request_id'))[:8]}"
        )


def reset_logging() -> None:
    """Remove the handlers configure_logging added (used by tests)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True


def configure_logging(settings: Settings | None = None, stream=None) -> logging.Logger:
    """Send the project's logs to stderr (and the turn log file, if set).

    Idempotent: calling it again replaces the handlers it added before, so a
    reloaded app or a test does not write every line twice. Logs of other
    libraries are left alone.
    """
    settings = settings or agent.SETTINGS
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(settings.log_level)
    logger.propagate = False

    console = logging.StreamHandler(stream or sys.stderr)
    console.setFormatter(JsonFormatter() if settings.log_format == "json" else TextFormatter())
    setattr(console, _HANDLER_MARK, True)
    logger.addHandler(console)

    if settings.turn_log_path:
        path = Path(settings.turn_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())  # the file is for machines, always JSON
        setattr(file_handler, _HANDLER_MARK, True)
        logger.addHandler(file_handler)
    return logger
