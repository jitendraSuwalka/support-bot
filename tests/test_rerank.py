"""Reranker failure modes: it must degrade, never break."""

import agent
from rag.retrieval import RetrievedChunk
from tests.conftest import make_chunk


def candidate(doc_id, section, text, lexical=0.4, dense=0.7):
    return RetrievedChunk(
        chunk=make_chunk(doc_id, section, text),
        score=0.03,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=lexical,
        dense_score=dense,
    )


CANDIDATES = [
    candidate("contact", "Our offices", "Bristol office"),
    candidate("appointments", "Callbacks", "not branch opening hours", lexical=0.2),
]


def sections(result):
    return [item.chunk.section for item in result]


def test_keeps_only_what_the_model_judged_relevant(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: {"relevant": [0]})
    assert sections(agent.rerank("q", CANDIDATES)) == ["Our offices"]


def test_rejecting_everything_yields_no_results(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: {"relevant": []})
    assert agent.rerank("q", CANDIDATES) == []


def test_falls_back_to_retrieval_order_when_the_helper_is_down(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: None)
    assert len(agent.rerank("q", CANDIDATES)) == 2


def test_malformed_indexes_are_ignored_safely(monkeypatch):
    monkeypatch.setattr(agent, "call_json", lambda *a, **k: {"relevant": [99, -1, "x", 1]})
    assert sections(agent.rerank("q", CANDIDATES)) == ["Callbacks"]


def test_empty_candidate_list_is_handled():
    assert agent.rerank("q", []) == []


def test_the_model_sees_each_passage_with_its_title_and_section(monkeypatch):
    seen = {}

    def spy(prompt, *a, **k):
        seen["prompt"] = prompt
        return {"relevant": [0]}

    monkeypatch.setattr(agent, "call_json", spy)
    agent.rerank("where are you", CANDIDATES)
    assert "[0] Contact › Our offices\nBristol office" in seen["prompt"]
    assert "[1] Appointments › Callbacks\nnot branch opening hours" in seen["prompt"]
    assert 'CUSTOMER QUESTION: "where are you"' in seen["prompt"]


def test_the_reranker_reason_is_recorded_when_it_drops_candidates(monkeypatch):
    monkeypatch.setattr(
        agent, "call_json", lambda *a, **k: {"relevant": [0], "reason": "entry 1 is a denial"}
    )
    trace = agent.Trace()
    token = agent._CURRENT_TRACE.set(trace)
    try:
        agent.rerank("q", CANDIDATES)
    finally:
        agent._CURRENT_TRACE.reset(token)
    event = trace.find("rerank")[0]
    assert event.data["reason"] == "entry 1 is a denial"
    assert event.data["dropped"] == 1
