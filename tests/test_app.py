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
