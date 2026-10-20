"""Reranker failure modes: it must degrade, never break."""

import agent

CANDIDATES = [
    (0.4, set(), {"topic": "offices", "content": "Bristol office", "source": "u1"}),
    (0.2, set(), {"topic": "callback", "content": "not branch opening hours", "source": "u2"}),
]


def topics(result):
    return [entry["topic"] for _, _, entry in result]


def test_keeps_only_what_the_model_judged_relevant(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: {"relevant": [0]})
    assert topics(agent.rerank("q", CANDIDATES)) == ["offices"]


def test_rejecting_everything_yields_no_results(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: {"relevant": []})
    assert agent.rerank("q", CANDIDATES) == []


def test_falls_back_to_lexical_order_when_the_helper_is_down(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: None)
    assert len(agent.rerank("q", CANDIDATES)) == 2


def test_malformed_indexes_are_ignored_safely(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: {"relevant": [99, -1, "x", 1]})
    assert topics(agent.rerank("q", CANDIDATES)) == ["callback"]


def test_empty_candidate_list_is_handled():
    assert agent.rerank("q", []) == []
