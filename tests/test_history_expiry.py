"""Volatile tool results expire between turns; static ones survive."""

import agent
from tests.conftest import Block


def make_history():
    return [
        {
            "role": "assistant",
            "content": [
                Block("tool_use", id="a", name="look_up_client_query"),
                Block("tool_use", id="b", name="search_knowledge_base"),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "Closed 28 July."},
                {"type": "tool_result", "tool_use_id": "b", "content": "Complaints info"},
            ],
        },
    ]


def test_volatile_result_is_wiped():
    messages = make_history()
    agent.expire_stale_results(messages)
    blocks = {b["tool_use_id"]: b["content"] for b in messages[1]["content"]}
    assert blocks["a"] == agent.EXPIRED_NOTICE


def test_static_result_survives():
    messages = make_history()
    agent.expire_stale_results(messages)
    blocks = {b["tool_use_id"]: b["content"] for b in messages[1]["content"]}
    assert blocks["b"] == "Complaints info"


def test_exactly_one_wipe_reported():
    assert agent.expire_stale_results(make_history()) == 1


def test_rerunning_wipes_nothing():
    messages = make_history()
    agent.expire_stale_results(messages)
    assert agent.expire_stale_results(messages) == 0
