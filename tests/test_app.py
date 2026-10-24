"""The web layer only presents an AgentResult; its status chips are pure logic."""

import re

import app
from agent import AgentResult, Trace


def make_result(kinds=(), tools=(), notes=(), rerank=None):
    trace = Trace()
    for kind in kinds:
        trace.log(kind, kind)
    if rerank:
        trace.log("rerank", "rerank dropped", dropped=rerank[0], total=rerank[1])
    return AgentResult(
        answer="x",
        notes=list(notes),
        evidence=[],
        trace=trace,
        escalated=False,
        tools_used=list(tools),
        models={},
    )


def labels(html):
    return re.findall(r'class="chip chip-(\w+)">([^<]+)</span>', html)


def test_clean_answer_shows_tool_and_both_checks_green():
    chips = labels(
        app.build_status(
            make_result(kinds=["citations_clean", "verified"], tools=["look_up_client_query"])
        )
    )
    assert ("tool", "Looked up case status") in chips
    assert ("ok", "Figures verified") in chips
    assert ("ok", "Grounded in evidence") in chips


def test_citation_rewrite_that_still_fails_is_marked_bad():
    chips = labels(
        app.build_status(
            make_result(
                kinds=["citation_violation", "citations_remaining"],
                notes=["WARNING: unverified detail remains (1)"],
            )
        )
    )
    assert ("bad", "Unsupported figure removed") in chips
    assert ("bad", "unverified detail remains (1)") in chips


def test_policy_escalation_and_rerank_are_reported():
    chips = labels(
        app.build_status(
            make_result(kinds=["policy_escalate"], tools=["search_knowledge_base"], rerank=(2, 5))
        )
    )
    assert ("warn", "Escalated by policy") in chips
    assert ("tool", "Reranked, 2 of 5 discarded") in chips


def test_greeting_shows_the_no_tools_chip():
    assert labels(app.build_status(make_result())) == [("tool", "Answered without tools")]


def test_a_declined_turn_shows_the_decline_and_what_was_not_run():
    trace_kinds = ["tool", "tool_skipped", "declined"]
    result = make_result(tools=["look_up_client_query", "decline_out_of_scope"])
    for kind in trace_kinds:
        result.trace.log(kind, kind, name="look_up_client_query" if kind != "declined" else "")
    chips = labels(app.build_status(result))
    assert ("tool", "Declined: outside scope") in chips
    assert ("none", "Not run: Looked up case status") in chips
    assert ("tool", "Looked up case status") not in chips


def test_a_failed_turn_puts_the_conversation_back_and_shows_no_error_text(monkeypatch):
    # The agent extends the conversation in place: here it has already
    # recorded a tool call when the model fails.
    def failing(messages, state):
        messages.append(
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x"}]}
        )
        state["unclear_turns"] = 1
        raise RuntimeError("secret internal detail sk-ant-123")

    monkeypatch.setattr(app.agent, "run_agent", failing)
    state = {"unclear_turns": 0}
    history, _, conversation, state, status, _, trace = app.respond("ISA?", [], [], state)
    # Put back as it was, keeping the session id, so a retry joins the same session.
    assert conversation == [] and set(state) == {"unclear_turns", "session_id"}
    assert state["unclear_turns"] == 0
    assert history[-1]["content"] == app.REQUEST_FAILED_REPLY
    assert "Request failed" in status
    assert "secret" not in history[-1]["content"] and "secret" not in trace
    assert "RuntimeError" in trace  # the type only, for whoever runs the app


def test_a_reply_replaced_by_the_last_line_check_says_so():
    chips = labels(app.build_status(make_result(kinds=["citation_fallback"])))
    assert ("warn", "Replaced with a fixed reply") in chips
