"""
The support agent as an HTTP service.

    uvicorn api:app --port 8000          # then open http://127.0.0.1:8000/docs

    POST   /chat                 one customer turn; omit session_id to start a conversation
    DELETE /sessions/{id}        end a conversation and drop what the server held for it
    GET    /health               the process is up (liveness)
    GET    /ready                it can answer: API key set, document index loaded (readiness)
    GET    /docs                 interactive OpenAPI documentation, generated from the code
    GET    /                     redirects to /docs

The service is a thin layer over the same agent the web interface uses:
every turn goes through ``observability.run_logged_turn``, so it is logged
and priced exactly as the app's are, and its reply carries what the agent
recorded — the behaviour, whether it escalated, the sources, any warnings,
the latency and the cost — so a client can act on the outcome without
parsing prose.

Conversations live in server memory, keyed by an id the server issues (a
random 128-bit hex string), never one the client chooses: an unknown or
expired id is refused, so one customer cannot continue another's
conversation by guessing at ids. One idle for ``session_ttl_minutes`` is
dropped, the oldest goes first beyond ``max_sessions``, and a conversation of
``max_turns`` must be restarted. Two requests for one conversation are
answered one after the other, because a turn extends the conversation it
reads. Memory is per process: a restart or a second worker starts empty,
which is stated in the README's limitations.

A failure inside the agent is a 502 with the request id and nothing of the
exception: the log line has the type, the client gets a reference to quote.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

import agent
import observability
from config import Settings

VERSION = "2.0.0-dev"
REQUEST_ID_HEADER = "X-Request-ID"
HTTP_LOGGER = f"{observability.LOGGER_NAME}.http"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_SESSION_ID = r"^[0-9a-f]{32}$"


# =========================================================
# SESSIONS
# =========================================================


@dataclass
class Session:
    id: str
    messages: list = field(default_factory=list)
    state: dict = field(default_factory=lambda: {"unclear_turns": 0})
    created: float = 0.0
    last_seen: float = 0.0
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class SessionNotFound(KeyError):
    pass


class SessionStore:
    """Conversations in memory: issued ids, an idle timeout, a size cap."""

    def __init__(
        self, ttl_seconds: float, max_sessions: int, clock: Callable[[], float] = time.monotonic
    ):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self.clock = clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)

    def _expire(self) -> None:
        cutoff = self.clock() - self.ttl
        for sid in [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]:
            del self._sessions[sid]

    def create(self) -> Session:
        with self._lock:
            self._expire()
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)  # the least recently used goes first
            now = self.clock()
            session = Session(id=uuid.uuid4().hex, created=now, last_seen=now)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(session_id)
            session.last_seen = self.clock()
            self._sessions.move_to_end(session_id)
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


# =========================================================
# SCHEMAS
# =========================================================


class ChatRequest(BaseModel):
    message: str = Field(
        description="The customer's message.",
        examples=["How much can I pay into an ISA this tax year?"],
    )
    session_id: str | None = Field(
        default=None,
        pattern=_SESSION_ID,
        description="The id from an earlier reply, to continue that conversation. "
        "Omit it to start a new one.",
    )

    @field_validator("message")
    @classmethod
    def message_has_words(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class Source(BaseModel):
    title: str
    section: str
    url: str


class Cost(BaseModel):
    usd: float | None = Field(description="None when a model's price is not known.")
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    calls: int


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    turn: int
    answer: str
    behaviour: str = Field(description="answer, escalate, clarify, decline or greeting")
    escalated: bool
    sources: list[Source]
    warnings: list[str]
    latency_ms: int
    cost: Cost


class Health(BaseModel):
    status: str
    version: str


class Readiness(BaseModel):
    status: str
    checks: dict[str, bool]
    sessions: int


class ErrorBody(BaseModel):
    detail: str
    request_id: str | None = None


# =========================================================
# THE APPLICATION
# =========================================================


def create_app(
    settings: Settings | None = None,
    runner: observability.Runner | None = None,
    warm_index: bool = True,
) -> FastAPI:
    """Build the service. Tests pass a scripted runner and skip warming the index."""
    settings = settings or agent.SETTINGS
    sessions = SessionStore(settings.session_ttl_minutes * 60, settings.max_sessions)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        observability.configure_logging(settings)
        if warm_index:
            # Open (on a new machine: download and build) the index before the
            # first request, so readiness means it and no customer pays for it.
            agent.get_retriever()
        yield

    app = FastAPI(
        title="Fernhill Wealth support agent",
        version=VERSION,
        summary="A grounded support agent: answers only what it can evidence, "
        "escalates what a person must handle.",
        lifespan=lifespan,
    )
    app.state.sessions = sessions

    http_log = logging.getLogger(HTTP_LOGGER)

    @app.middleware("http")
    async def request_id_and_access_log(request: Request, call_next):
        given = request.headers.get(REQUEST_ID_HEADER, "")
        rid = given if _REQUEST_ID.match(given) else uuid.uuid4().hex
        request.state.request_id = rid
        started = time.perf_counter()
        status = 500  # what the client gets if the handler raises past every handler
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = rid
            return response
        finally:
            # One JSON line per request, in the same stream and format as the
            # turn log, so refusals (422, 404, 409) are visible too. Liveness
            # probes arrive every few seconds and say nothing; they are not
            # logged. uvicorn's own plain-text access log is switched off in
            # the container (--no-access-log), so the stream stays parseable.
            if request.url.path != "/health":
                fields = {
                    "event": "request",
                    "ts": observability.utc_now(),
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
                level = logging.ERROR if status >= 500 else logging.INFO
                http_log.log(level, "request", extra={"fields": fields})

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            404: {"model": ErrorBody, "description": "Unknown or expired session"},
            409: {"model": ErrorBody, "description": "Conversation too long; start a new one"},
            422: {"description": "Invalid request: empty or over-long message, malformed id"},
            502: {"model": ErrorBody, "description": "The agent could not complete the turn"},
        },
    )
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        rid = request.state.request_id
        if len(body.message) > settings.max_message_chars:
            raise HTTPException(
                422, f"message is longer than {settings.max_message_chars} characters"
            )
        if body.session_id is None:
            session = sessions.create()
        else:
            try:
                session = sessions.get(body.session_id)
            except SessionNotFound:
                raise HTTPException(
                    404, "unknown or expired session; omit session_id to start a new one"
                ) from None

        with session.lock:  # one turn at a time per conversation
            if session.turns >= settings.max_turns:
                raise HTTPException(
                    409, f"this conversation has reached {settings.max_turns} turns"
                )
            session.messages.append({"role": "user", "content": body.message})
            try:
                result, record = observability.run_logged_turn(
                    session.messages,
                    session.state,
                    session_id=session.id,
                    request_id=rid,
                    runner=runner,
                    settings=settings,
                )
            except Exception:
                # The turn failed: take the message back off, so the
                # conversation is as it was and the customer can try again.
                session.messages.pop()
                raise HTTPException(502, "the agent could not complete that request") from None
            session.turns += 1
            cost = result.cost()
            return ChatResponse(
                session_id=session.id,
                request_id=rid,
                turn=session.turns,
                answer=result.answer,
                behaviour=result.behaviour,
                escalated=result.escalated,
                sources=[Source(**source) for source in result.sources()],
                warnings=record["warnings"],
                latency_ms=record["latency_ms"],
                cost=Cost(**{k: v for k, v in cost.as_dict().items() if k != "unpriced_models"}),
            )

    @app.delete("/sessions/{session_id}", status_code=204, responses={404: {"model": ErrorBody}})
    def end_session(session_id: str) -> Response:
        if not re.fullmatch(_SESSION_ID, session_id) or not sessions.delete(session_id):
            raise HTTPException(404, "unknown or expired session")
        return Response(status_code=204)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        # Someone who opens the service's address in a browser wants to know
        # what it is: send them to the documentation instead of a 404.
        return RedirectResponse("/docs")

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=VERSION)

    @app.get("/ready", response_model=Readiness, responses={503: {"model": Readiness}})
    def ready():
        checks = {
            "api_key": bool(getattr(agent.client, "api_key", None)),
            "index": agent._RETRIEVER is not None,
        }
        body = Readiness(
            status="ready" if all(checks.values()) else "not ready",
            checks=checks,
            sessions=len(sessions),
        )
        return JSONResponse(body.model_dump(), status_code=200 if all(checks.values()) else 503)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            {"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
            status_code=exc.status_code,
            headers={REQUEST_ID_HEADER: getattr(request.state, "request_id", "")},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000)
