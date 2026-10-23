"""The turn log: one JSON line per turn, everything about what happened and
nothing of what was said, unless a deployment asks for the content."""

import hashlib
import io
import json
import logging

import pytest

import agent
import app
import observability
from config import Settings
from tests.conftest import FakeClient, text_reply, tool_reply

QUESTION = "Is FW-1001 done?"
STATUS = "Received 1 Aug. With your adviser. Response due by 6 Aug."


def status_result():
    trace = agent.Trace()
    trace.log(
        "usage", "u", role="main", model="claude-sonnet-5-5", input_tokens=2000, output_tokens=80
    )
    trace.log(
        "usage", "u", role="helper", model="claude-haiku-5-5", input_tokens=300, output_tokens=20
    )
    trace.log("citation_violation", "x", violations=["URL"])
    return agent.AgentResult(
        answer=f'Status: "{STATUS}"',
        notes=["rewritten after citation check"],
        evidence=[f"look_up_client_query -> STATUS_VERBATIM: {STATUS}"],
        trace=trace,
        escalated=False,
        tools_used=["look_up_client_query"],
        models={"main": "claude-sonnet-5-5", "helper": "claude-haiku-5-5"},
    )


@pytest.fixture
def conversation():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "user", "content": QUESTION},
    ]


@pytest.fixture
def captured():
    """The project logger, writing JSON to a buffer for the test to read."""
    stream = io.StringIO()
    observability.configure_logging(Settings(log_format="json"), stream=stream)
    yield stream
    observability.reset_logging()


def lines(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# --- the record ------------------------------------------------------------------------


def test_a_turn_record_says_what_happened_and_what_it_cost(conversation):
    record = observability.turn_record(
        status_result(),
        messages=conversation,
        request_id="r1",
        session_id="s1",
        seconds=1.2345,
        settings=Settings(),
    )
    assert record["event"] == "turn" and record["ts"].endswith("Z")
    assert (record["request_id"], record["session_id"], record["turn"]) == ("r1", "s1", 2)
    assert record["behaviour"] == "answer" and record["escalated"] is False
    assert record["tools"] == ["look_up_client_query"] and record["guards"] == ["citation_rewrite"]
    assert record["latency_ms"] == 1234 and record["evidence_parts"] == 1
    assert record["models"]["main"] == "claude-sonnet-5-5"
    expected = (2000 * 2 + 80 * 10 + 300 * 0.1 + 20 * 0.5) / 1e6
    assert record["cost"]["usd"] == round(expected, 6) and record["cost"]["calls"] == 2
    json.dumps(record)  # serialisable as it stands


def test_the_customers_words_are_not_logged_by_default(conversation):
    record = observability.turn_record(
        status_result(),
        messages=conversation,
        request_id="r",
        session_id="s",
        seconds=0,
        settings=Settings(),
    )
    text = json.dumps(record)
    assert QUESTION not in text and STATUS not in text
    assert record["message"] == {
        "chars": len(QUESTION),
        "sha256": hashlib.sha256(QUESTION.encode()).hexdigest()[:16],
    }
    assert set(record["answer"]) == {"chars", "sha256"}


def test_a_deployment_can_choose_to_log_the_content(conversation):
    record = observability.turn_record(
        status_result(),
        messages=conversation,
        request_id="r",
        session_id="s",
        seconds=0,
        settings=Settings(log_content=True),
    )
    assert record["message"]["text"] == QUESTION and STATUS in record["answer"]["text"]


def test_a_failed_turn_is_recorded_as_an_error_without_the_answer_fields(conversation):
    record = observability.turn_record(
        None,
        messages=conversation,
        request_id="r",
        session_id="s",
        seconds=0.5,
        settings=Settings(),
        error=RuntimeError("API down"),
    )
    assert record["behaviour"] == "error"
    assert record["error"] == {"type": "RuntimeError", "detail": "API down"}
    assert "cost" not in record and "answer" not in record


# --- running a turn --------------------------------------------------------------------


def test_a_logged_turn_writes_one_json_line_and_returns_the_result(conversation, captured):
    result, record = observability.run_logged_turn(
        conversation, {"unclear_turns": 0}, session_id="s1", runner=lambda m, s: status_result()
    )
    assert result.answer.startswith("Status:")
    [line] = lines(captured)
    assert line == dict(record, level="INFO")
    assert len(line["request_id"]) == 32  # a fresh uuid when the caller gives none


def test_a_turn_with_a_warning_is_logged_at_warning_level(conversation, captured):
    def runner(messages, state):
        result = status_result()
        result.notes.append("WARNING: unverified detail remains (1)")
        return result

    observability.run_logged_turn(conversation, {}, session_id="s", request_id="r", runner=runner)
    [line] = lines(captured)
    assert line["level"] == "WARNING" and line["warnings"] == [
        "WARNING: unverified detail remains (1)"
    ]


def test_a_turn_that_raises_is_logged_and_the_error_still_reaches_the_caller(
    conversation, captured
):
    def broken(messages, state):
        raise ConnectionError("no route to api.anthropic.com")

    with pytest.raises(ConnectionError):
        observability.run_logged_turn(conversation, {}, session_id="s", runner=broken)
    [line] = lines(captured)
    assert line["level"] == "ERROR" and line["behaviour"] == "error"
    assert line["error"]["type"] == "ConnectionError"


def test_the_real_agent_turn_is_logged_end_to_end(monkeypatch, captured):
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
    messages = [{"role": "user", "content": QUESTION}]
    result, _ = observability.run_logged_turn(messages, {"unclear_turns": 0}, session_id="s")
    [line] = lines(captured)
    assert line["behaviour"] == "answer" and line["tools"] == ["look_up_client_query"]
    assert line["cost"] == {
        "usd": 0,  # the scripted model reports no usage
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "calls": 0,
        "unpriced_models": [],
    }
    assert STATUS in result.answer and STATUS not in json.dumps(line)


# --- configuration ---------------------------------------------------------------------


def test_configuring_twice_does_not_write_every_line_twice(conversation):
    stream = io.StringIO()
    observability.configure_logging(Settings(), stream=stream)
    observability.configure_logging(Settings(), stream=stream)
    observability.run_logged_turn(
        conversation, {}, session_id="s", runner=lambda m, s: status_result()
    )
    assert len(lines(stream)) == 1
    logger = logging.getLogger(observability.LOGGER_NAME)
    ours = [h for h in logger.handlers if getattr(h, observability._HANDLER_MARK, False)]
    assert logger.propagate is False and len(ours) == 1  # pytest adds its own capture handlers
    observability.reset_logging()


def test_the_turn_log_file_gets_json_even_when_the_console_gets_text(conversation, tmp_path):
    path = tmp_path / "logs" / "turns.jsonl"
    console = io.StringIO()
    observability.configure_logging(
        Settings(log_format="text", turn_log_path=str(path)), stream=console
    )
    try:
        observability.run_logged_turn(
            conversation, {}, session_id="abcdef123456", runner=lambda m, s: status_result()
        )
    finally:
        observability.reset_logging()
    [record] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert record["behaviour"] == "answer"
    text = console.getvalue()
    assert (
        text.startswith("INFO turn 2 answer") and "session=abcdef12" in text and "$0.0048" in text
    )


def test_other_messages_on_the_project_logger_are_json_too(captured):
    logging.getLogger("support_bot.app").warning("index rebuilt")
    [line] = lines(captured)
    assert (
        line["event"] == "log" and line["level"] == "WARNING" and line["message"] == "index rebuilt"
    )


@pytest.mark.parametrize(
    ("name", "value"), [("SUPPORT_BOT_LOG_LEVEL", "LOUD"), ("SUPPORT_BOT_LOG_FORMAT", "xml")]
)
def test_logging_settings_are_validated(name, value):
    with pytest.raises(ValueError):
        Settings.from_env({name: value})


# --- the web interface logs its turns --------------------------------------------------


def test_the_app_logs_each_turn_under_one_session_id(monkeypatch, captured):
    monkeypatch.setattr(agent, "run_agent", lambda messages, state: status_result())
    state = {"unclear_turns": 0}
    history, conversation = [], []
    for question in ("hello", QUESTION):
        history, _, conversation, state, *_ = app.respond(question, history, conversation, state)
    first, second = lines(captured)
    assert first["session_id"] == second["session_id"] == state["session_id"]
    assert (first["turn"], second["turn"]) == (1, 2)
    # Reset gives a new state, so the next conversation is a new session.
    assert "session_id" not in app.reset()[2]


def test_a_rewrite_in_an_earlier_turn_does_not_inflate_the_turn_count(monkeypatch, captured):
    # The citation guard's instruction is sent in the user role; before it was
    # written as content blocks it was counted as a customer turn.
    monkeypatch.setattr(
        agent,
        "client",
        FakeClient(
            [
                tool_reply("look_up_client_query", {"reference": "FW-1001"}),
                text_reply("Status: received. Call 0800 999 9999."),
                text_reply(f'Status for FW-1001: "{STATUS}"'),
                tool_reply("look_up_client_query", {"reference": "FW-1001"}, call_id="t2"),
                text_reply(f'Status for FW-1001: "{STATUS}"'),
            ]
        ),
    )
    messages, state = [{"role": "user", "content": QUESTION}], {"unclear_turns": 0}
    first, _ = observability.run_logged_turn(messages, state, session_id="s")
    assert "rewritten after citation check" in first.notes
    messages.append({"role": "user", "content": "And FW-1001 again?"})
    observability.run_logged_turn(messages, state, session_id="s")
    assert [line["turn"] for line in lines(captured)] == [1, 2]


def test_an_agent_event_outside_a_turn_joins_the_structured_stream(captured):
    agent.log("index", "index ready: 129 chunks", chunks=129)
    [line] = lines(captured)
    assert line["event"] == "index" and line["chunks"] == 129 and line["ts"].endswith("Z")


def test_an_agent_event_outside_a_turn_is_printed_when_nothing_is_configured(capsys):
    observability.reset_logging()
    agent.log("index", "index ready")
    assert "[index] index ready" in capsys.readouterr().out
