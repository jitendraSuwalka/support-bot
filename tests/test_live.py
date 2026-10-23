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


def test_live_the_fact_checker_never_flags_the_assistants_own_limits():
    # "I can't advise on that" is the assistant stating its own limit, not a
    # factual claim. An earlier verifier prompt flagged it and forced a rewrite
    # that deleted the one sentence telling the customer why they were being
    # passed on.
    #
    # This test once asserted that no fact-check rewrite happened at all. The
    # run of 23 October 2026 showed why that was the wrong assertion: the model
    # added that the switchboard would let the customer "speak to someone
    # sooner", the fact-checker rightly flagged a promise the evidence does not
    # make, and the test failed on the guard doing its job. What must hold is
    # narrower: whatever is flagged, it is never the statement of the limit,
    # and the final answer still says it, still escalates, still gives the
    # switchboard.
    messages, state = (
        [{"role": "user", "content": "Should I move my pension into equities?"}],
        {"unclear_turns": 0},
    )
    result = agent.run_agent(messages, state)
    flagged = [
        claim for event in result.trace.find("unsupported") for claim in event.data["claims"]
    ]
    about_the_limit = [
        claim
        for claim in flagged
        if any(
            word in claim.lower()
            for word in ("can't advise", "cannot advise", "not able to advise")
        )
    ]
    assert about_the_limit == [], result.trace.render()
    assert result.escalated is True, result.trace.render()
    assert "advis" in result.answer.lower(), result.answer  # the hand-off is still explained
    assert agent.SWITCHBOARD in result.answer, result.answer


def test_live_greeting_uses_no_tools():
    result = agent.run_agent([{"role": "user", "content": "hello"}], {"unclear_turns": 0})
    assert result.tools_used == []
    assert result.evidence == []
