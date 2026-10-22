"""Opt-in tests against the real API. They cost money and are skipped by default.

Run them with:  pytest -m live
They also skip themselves when no API key is configured. Each case states what
a correct run looks like at the level of tools used, wording and guard notes,
so a change in model behaviour is caught as a failure rather than noticed by
a customer.
"""

import os

import pytest

import agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY set"),
]

CASES = [
    pytest.param(
        "What's the status of FW-1002?",
        ["look_up_client_query"],
        ["Documents verified. Awaiting your signature to proceed."],
        False,
        id="verbatim-status",
    ),
    pytest.param(
        "Do you have a Manchester office?",
        ["search_knowledge_base"],
        ["Manchester"],
        False,
        id="knowledge-base-lookup",
    ),
    pytest.param(
        "How much can I pay into an ISA this tax year?",
        ["search_knowledge_base"],
        ["£20,000"],
        False,
        id="corpus-fact-by-meaning",
    ),
    pytest.param(
        "Whose number is 0161 496 0170?",
        ["search_knowledge_base"],
        ["Manchester"],
        False,
        id="corpus-fact-by-exact-string",
    ),
    pytest.param(
        "Should I move my pension into equities?",
        ["escalate_to_adviser"],
        ["adviser", agent.SWITCHBOARD],
        True,
        id="advice-is-escalated",
    ),
    pytest.param(
        # Nothing in the corpus covers mortgages as a service (Phase 1 used
        # "Can I visit a branch on Sunday?", which the corpus now answers).
        "Do you offer mortgage advice?",
        [],
        ["adviser"],
        True,
        id="unanswerable-reaches-a-human",
    ),
]


@pytest.mark.parametrize(("question", "expect_tools", "expect_text", "expect_escalated"), CASES)
def test_live_question(question, expect_tools, expect_text, expect_escalated):
    messages, state = [{"role": "user", "content": question}], {"unclear_turns": 0}
    result = agent.run_agent(messages, state)

    for tool in expect_tools:
        assert tool in result.tools_used, f"used {result.tools_used}"
    for phrase in expect_text:
        assert phrase.lower() in result.answer.lower(), result.answer[:200]
    assert result.escalated is expect_escalated, result.trace.render()
    assert not any(note.startswith("WARNING") for note in result.notes), result.notes


def test_live_escalation_wording_is_not_rewritten_by_the_fact_checker():
    # "I can't advise on that" is the assistant stating its own limit, not a
    # factual claim. An earlier verifier prompt flagged it and forced a rewrite
    # that deleted the one sentence telling the customer why they were being
    # passed on.
    messages, state = (
        [{"role": "user", "content": "Should I move my pension into equities?"}],
        {"unclear_turns": 0},
    )
    result = agent.run_agent(messages, state)
    assert "rewritten after fact-check" not in result.notes, result.trace.render()


def test_live_greeting_uses_no_tools():
    result = agent.run_agent([{"role": "user", "content": "hello"}], {"unclear_turns": 0})
    assert result.tools_used == []
    assert result.evidence == []
