"""Opt-in tests against the real API. They cost money and are skipped by default.

Run them with:  pytest -m live
They also skip themselves when no API key is configured.
"""

import os

import pytest

import agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY set"),
]

CASES = [
    ("Where is your Bristol office?", "search_knowledge_base", ["Victoria Street"]),
    ("What's the status of EP-1002?", "look_up_client_query", ["Documents verified"]),
    ("Should I move my pension into equities?", "escalate_to_adviser", ["adviser"]),
    ("Can I visit a branch on Sunday?", None, ["adviser"]),
]


@pytest.mark.parametrize(("question", "expect_tool", "expect_text"), CASES)
def test_live_question(question, expect_tool, expect_text):
    messages, state = [{"role": "user", "content": question}], {"unclear_turns": 0}
    answer = agent.run_agent(messages, state)

    used = [
        b.name
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if getattr(b, "type", None) == "tool_use"
    ]
    if expect_tool:
        assert expect_tool in used, f"used {used}"
    for phrase in expect_text:
        assert phrase.lower() in answer.lower(), answer[:150]
