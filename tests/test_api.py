"""The HTTP API, driven in-process with FastAPI's TestClient: no server, no
network, no model. The agent behind it is either a scripted runner or the
real agent against the scripted fake model."""

import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import agent
import api
import observability
from config import Settings
from tests.conftest import FakeClient, text_reply, tool_reply

STATUS = "Received 1 Aug. With your adviser. Response due by 6 Aug."


def scripted_result(answer="Here you are.", escalated=False):
    trace = agent.Trace()
    trace.log(
        "usage", "u", role="main", model="claude-sonnet-5-5", input_tokens=1000, output_tokens=100
    )
    trace.log(
        "kept",
        "k",
        chunk_id="isa-guide:1",
        title="ISAs explained",
        section="The allowance",
        url="https://www.fernhillwealth.example/guides/isas/",
    )
    return agent.AgentResult(
        answer=answer,
        notes=[],
        evidence=["search_knowledge_base -> [ISAs explained › The allowance] £20,000"],
        trace=trace,
        escalated=escalated,
        tools_used=["search_knowledge_base"],
        models={"main": "claude-sonnet-5-5", "helper": "claude-haiku-5-5"},
    )


class Runner:
    """Records what it was given and returns a scripted result."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.seen = result, error, []

    def __call__(self, messages, state):
        self.seen.append([m["content"] for m in messages if isinstance(m["content"], str)])
        if self.error:
            raise self.error
        return self.result or scripted_result()


@pytest.fixture
def log_stream():
    stream = io.StringIO()
    observability.configure_logging(Settings(), stream=stream)
    yield stream
    observability.reset_logging()


def log_lines(stream):
    return [json.loads(x) for x in stream.getvalue().splitlines() if x.strip()]


def client_for(runner=None, **settings):
    app = api.create_app(Settings(**settings), runner=runner or Runner(), warm_index=False)
    return TestClient(app), app


# --- liveness, readiness, documentation -------------------------------------------------


def test_health_says_the_process_is_up_and_carries_a_request_id():
    client, _ = client_for()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": api.VERSION}
    assert len(response.headers[api.REQUEST_ID_HEADER]) == 32


def test_ready_needs_the_api_key_and_the_index(monkeypatch):
    client, _ = client_for()
    monkeypatch.setattr(agent.client, "api_key", "sk-test")
    response = client.get("/ready")  # the test suite's in-memory index is loaded
    assert response.status_code == 200 and response.json()["status"] == "ready"
    assert response.json()["checks"] == {"api_key": True, "index": True}

    monkeypatch.setattr(agent.client, "api_key", None)
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["checks"]["api_key"] is False

    monkeypatch.setattr(agent.client, "api_key", "sk-test")
    monkeypatch.setattr(agent, "_RETRIEVER", None)
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["checks"]["index"] is False


def test_the_openapi_document_describes_every_endpoint():
    client, _ = client_for()
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/chat", "/sessions/{session_id}", "/health", "/ready"}
    assert {"404", "409", "422", "502"} <= set(paths["/chat"]["post"]["responses"])
    assert client.get("/docs").status_code == 200


# --- a conversation ---------------------------------------------------------------------


def test_a_first_message_starts_a_conversation_and_reports_the_outcome(log_stream):
    runner = Runner()
    client, _ = client_for(runner)
    response = client.post("/chat", json={"message": "  How much can I pay into an ISA?  "})
    assert response.status_code == 200
    body = response.json()
    assert len(body["session_id"]) == 32 and body["turn"] == 1
    assert body["answer"] == "Here you are." and body["behaviour"] == "answer"
    assert body["escalated"] is False and body["warnings"] == []
    assert body["sources"] == [
        {
            "title": "ISAs explained",
            "section": "The allowance",
            "url": "https://www.fernhillwealth.example/guides/isas/",
        }
    ]
    assert body["cost"]["usd"] == round((1000 * 2 + 100 * 10) / 1e6, 6)
    assert body["request_id"] == response.headers[api.REQUEST_ID_HEADER]
    assert runner.seen == [["How much can I pay into an ISA?"]]  # stripped
    # One log line, with the ids the client was given.
    [line] = [x for x in log_lines(log_stream) if x["event"] == "turn"]
    assert (line["session_id"], line["request_id"]) == (body["session_id"], body["request_id"])


def test_a_session_id_continues_the_conversation():
    runner = Runner()
    client, _ = client_for(runner)
    first = client.post("/chat", json={"message": "hello"}).json()
    second = client.post(
        "/chat", json={"message": "and pensions?", "session_id": first["session_id"]}
    )
    assert second.status_code == 200 and second.json()["turn"] == 2
    assert second.json()["session_id"] == first["session_id"]
    assert runner.seen[-1] == ["hello", "and pensions?"]  # the runner saw the whole conversation


def test_the_real_agent_answers_through_the_api(monkeypatch):
    monkeypatch.setattr(
        agent,
        "client",
        FakeClient(
            [
                tool_reply("look_up_client_query", {"reference": "FW-1001"}),
                text_reply(f'Status for FW-1001: "{STATUS}"'),
            ]
        ),
    )
    app = api.create_app(Settings(), warm_index=False)  # no runner: the real agent
    body = TestClient(app).post("/chat", json={"message": "What's the status of FW-1001?"}).json()
    assert STATUS in body["answer"] and body["behaviour"] == "answer" and body["sources"] == []
    [session] = app.state.sessions._sessions.values()
    assert [m["role"] for m in session.messages] == ["user", "assistant", "user", "assistant"]


def test_a_request_id_the_client_sends_is_used_and_a_malformed_one_is_replaced():
    client, _ = client_for()
    given = client.post(
        "/chat", json={"message": "hi"}, headers={api.REQUEST_ID_HEADER: "client-req-0001"}
    )
    assert given.json()["request_id"] == "client-req-0001"
    assert given.headers[api.REQUEST_ID_HEADER] == "client-req-0001"
    bad = client.post("/chat", json={"message": "hi"}, headers={api.REQUEST_ID_HEADER: "x"})
    assert bad.json()["request_id"] != "x" and len(bad.json()["request_id"]) == 32


# --- refusals --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {},
        {"message": "hi", "session_id": "not-a-session-id"},
        {"message": "hi", "session_id": "../../etc/passwd"},
    ],
)
def test_malformed_requests_are_refused_before_any_model_call(payload):
    runner = Runner()
    client, _ = client_for(runner)
    assert client.post("/chat", json=payload).status_code == 422
    assert runner.seen == []


def test_an_over_long_message_is_refused_with_the_limit():
    runner = Runner()
    client, _ = client_for(runner, max_message_chars=20)
    response = client.post("/chat", json={"message": "x" * 21})
    assert response.status_code == 422 and "20 characters" in response.json()["detail"]
    assert runner.seen == []


def test_an_unknown_session_is_refused_not_silently_restarted():
    runner = Runner()
    client, _ = client_for(runner)
    response = client.post("/chat", json={"message": "hi", "session_id": "0" * 32})
    assert response.status_code == 404
    assert "omit session_id" in response.json()["detail"] and response.json()["request_id"]
    assert runner.seen == []


def test_a_conversation_that_reaches_the_turn_limit_must_be_restarted():
    client, _ = client_for(max_turns=2)
    sid = client.post("/chat", json={"message": "one"}).json()["session_id"]
    assert client.post("/chat", json={"message": "two", "session_id": sid}).status_code == 200
    third = client.post("/chat", json={"message": "three", "session_id": sid})
    assert third.status_code == 409 and "2 turns" in third.json()["detail"]


def test_a_failing_turn_is_a_502_that_leaks_nothing_and_leaves_the_conversation_as_it_was(
    log_stream,
):
    runner = Runner(error=RuntimeError("secret internal detail sk-ant-123"))
    client, app = client_for(runner)
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "secret" not in response.text and response.json()["request_id"]
    [session] = app.state.sessions._sessions.values()
    assert session.messages == [] and session.turns == 0  # the message was taken back off
    [line] = [x for x in log_lines(log_stream) if x["event"] == "turn"]
    assert line["behaviour"] == "error" and line["error"]["type"] == "RuntimeError"
    # The customer can simply try again in the same conversation.
    runner.error = None
    again = client.post("/chat", json={"message": "hi", "session_id": session.id})
    assert again.status_code == 200 and again.json()["turn"] == 1


# --- ending a conversation ---------------------------------------------------------------


def test_ending_a_conversation_drops_it():
    client, _ = client_for()
    sid = client.post("/chat", json={"message": "hi"}).json()["session_id"]
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.post("/chat", json={"message": "hi", "session_id": sid}).status_code == 404
    assert client.delete(f"/sessions/{sid}").status_code == 404
    assert client.delete("/sessions/nonsense").status_code == 404


# --- one turn at a time per conversation ---------------------------------------------------


def test_two_requests_for_one_conversation_are_answered_one_after_the_other():
    active, peak, guard = [0], [0], threading.Lock()

    def slow(messages, state):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return scripted_result()

    client, _ = client_for(slow)
    sid = client.post("/chat", json={"message": "first"}).json()["session_id"]
    peak[0] = 0
    threads = [
        threading.Thread(
            target=client.post,
            args=("/chat",),
            kwargs={"json": {"message": f"m{i}", "session_id": sid}},
        )
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


# --- the session store -------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_an_idle_session_expires():
    clock = Clock()
    store = api.SessionStore(ttl_seconds=60, max_sessions=10, clock=clock)
    session = store.create()
    clock.now += 59
    assert store.get(session.id) is session  # and that touch renews it
    clock.now += 59
    assert store.get(session.id) is session
    clock.now += 61
    with pytest.raises(api.SessionNotFound):
        store.get(session.id)
    assert len(store) == 0


def test_the_least_recently_used_session_goes_first_when_the_store_is_full():
    clock = Clock()
    store = api.SessionStore(ttl_seconds=600, max_sessions=2, clock=clock)
    a, b = store.create(), store.create()
    store.get(a.id)  # a is now the most recently used
    c = store.create()
    assert len(store) == 2
    with pytest.raises(api.SessionNotFound):
        store.get(b.id)
    assert store.get(a.id) is a and store.get(c.id) is c
    assert a.id != c.id and len(a.id) == 32


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SUPPORT_BOT_MAX_MESSAGE_CHARS", "0"),
        ("SUPPORT_BOT_SESSION_TTL_MINUTES", "0"),
        ("SUPPORT_BOT_MAX_SESSIONS", "0"),
        ("SUPPORT_BOT_MAX_TURNS", "0"),
    ],
)
def test_the_api_settings_are_validated(name, value):
    with pytest.raises(ValueError):
        Settings.from_env({name: value})


def test_starting_the_service_configures_logging_and_opens_the_index(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "get_retriever", lambda: calls.append("index"))
    monkeypatch.setattr(observability, "configure_logging", lambda s: calls.append("logging"))
    with TestClient(api.create_app(Settings(), runner=Runner())) as client:  # runs the lifespan
        assert calls == ["logging", "index"]
        assert client.get("/health").status_code == 200


def test_the_bare_address_leads_to_the_documentation():
    client, _ = client_for()
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/docs"
    assert client.get("/").status_code == 200  # followed, it is the docs page
    assert "/" not in client.get("/openapi.json").json()["paths"]  # not part of the API


# --- one JSON line per request ------------------------------------------------------------


def test_every_request_but_the_liveness_probe_is_logged_as_json(log_stream):
    client, _ = client_for()
    client.get("/health")
    ok = client.post(
        "/chat", json={"message": "hi"}, headers={api.REQUEST_ID_HEADER: "req-00000001"}
    )
    client.post("/chat", json={"message": ""})
    client.get("/ready")
    requests = [x for x in log_lines(log_stream) if x["event"] == "request"]
    assert [(r["method"], r["path"], r["status"]) for r in requests] == [
        ("POST", "/chat", 200),
        ("POST", "/chat", 422),
        ("GET", "/ready", requests[2]["status"]),
    ]
    assert requests[0]["request_id"] == "req-00000001" == ok.json()["request_id"]
    assert all(r["latency_ms"] >= 0 and r["ts"].endswith("Z") for r in requests)
    # The turn line and the request line of one call share the request id.
    turn = next(x for x in log_lines(log_stream) if x["event"] == "turn")
    assert turn["request_id"] == requests[0]["request_id"]


def test_the_text_format_shows_a_request_on_one_line():
    stream = io.StringIO()
    observability.configure_logging(Settings(log_format="text"), stream=stream)
    try:
        client, _ = client_for()
        client.post("/chat", json={"message": ""}, headers={api.REQUEST_ID_HEADER: "abcdef123456"})
    finally:
        observability.reset_logging()
    assert "INFO POST /chat 422" in stream.getvalue() and "request=abcdef12" in stream.getvalue()
