"""The agent's search over the corpus: floors, expansion, reranking, the
evidence it hands the model, and what it logs. Runs on the in-memory retriever
from conftest (hash embedder), so the lexical side decides."""

import pytest

import agent
from rag.retrieval import RetrievedChunk, open_retriever
from rag.store import SyncReport
from tests.conftest import make_chunk


def result_with(lexical=None, dense=None):
    return RetrievedChunk(
        chunk=make_chunk("d", "S", "text"),
        score=0.01,
        lexical_rank=1 if lexical is not None else None,
        dense_rank=1 if dense is not None else None,
        lexical_score=lexical,
        dense_score=dense,
    )


@pytest.fixture
def helper(monkeypatch):
    """Script the helper model: expansion terms and the rerank verdict."""
    calls = {"expand": set(), "rerank": {"relevant": [0, 1, 2]}}

    def fake_call_json(prompt, *a, **k):
        if "expand" in prompt.lower():
            return sorted(calls["expand"])
        return calls["rerank"]

    monkeypatch.setattr(agent, "call_json", fake_call_json)
    return calls


@pytest.fixture
def trace():
    trace = agent.Trace()
    token = agent._CURRENT_TRACE.set(trace)
    yield trace
    agent._CURRENT_TRACE.reset(token)


# --- floors -----------------------------------------------------------------------------


def test_a_candidate_clears_the_floor_on_either_side():
    assert agent.clears_floor(result_with(lexical=agent.RELEVANCE_FLOOR))
    assert agent.clears_floor(result_with(dense=agent.DENSE_FLOOR))
    assert agent.clears_floor(result_with(lexical=0.01, dense=0.9))


def test_a_candidate_below_both_floors_is_dropped():
    assert not agent.clears_floor(result_with(lexical=0.1, dense=0.5))
    assert not agent.clears_floor(result_with(dense=0.59))
    assert not agent.clears_floor(result_with())


def test_the_dense_floor_sits_between_measured_off_topic_and_on_topic_scores():
    # Step 2.3 calibration with BAAI/bge-small-en-v1.5: off-topic questions
    # peaked at 0.529; on-topic chunks scored 0.681 and above.
    assert 0.53 < agent.DENSE_FLOOR < 0.68


def test_describe_hit_shows_both_methods_or_a_dash():
    both = agent.describe_hit(result_with(lexical=0.5, dense=0.7))
    assert both == "lexical #1 0.50 · dense #1 0.70"
    assert agent.describe_hit(result_with(dense=0.7)) == "lexical — · dense #1 0.70"


# --- search_knowledge_base ------------------------------------------------------------------


def test_evidence_carries_title_section_text_and_source_url(helper, trace):
    evidence = agent.search_knowledge_base("Do you have a Manchester office?")
    assert evidence.startswith("[Contact us and our offices › Our offices › Manchester]\n")
    assert "41 Mosley Street, Manchester, M2 3ZX" in evidence
    assert "Source: https://www.fernhillwealth.example/contact-us/" in evidence
    kept = trace.find("kept")
    assert kept and kept[0].data["section"] == "Our offices › Manchester"
    assert kept[0].data["found_by"] in ("both", "lexical")


def test_expansion_terms_are_logged_and_widen_the_lexical_search(helper, trace):
    helper["expand"] = {"complaint", "ombudsman"}
    evidence = agent.search_knowledge_base("I am unhappy with the service")
    assert "Complaints" in evidence or "complaint" in evidence.lower()
    [expanded] = trace.find("expanded")
    assert set(expanded.data["terms"]) >= {"complaint", "ombudsman"}


def test_off_topic_questions_return_no_match_and_log_why(helper, trace):
    helper["expand"] = set()
    evidence = agent.search_knowledge_base("quiz")
    assert evidence.startswith("NO_MATCH")
    [event] = trace.find("no_candidates")
    assert "best was" in event.message and "dense" in event.data["best"]


def test_the_reranker_can_reject_every_candidate(helper, trace):
    helper["rerank"] = {"relevant": [], "reason": "none answer it"}
    assert agent.search_knowledge_base("Manchester office").startswith("NO_MATCH")
    assert trace.find("rerank_rejected_all")


def test_no_more_than_top_k_chunks_are_kept(helper):
    helper["rerank"] = {"relevant": [0, 1, 2, 3, 4, 5]}
    evidence = agent.search_knowledge_base("ISA allowance transfers Lifetime ISA")
    assert evidence.count("\nSource: ") == agent.TOP_K


def test_unknown_expansion_words_cannot_sink_a_real_match(helper):
    # The Phase 1 live failure, now at corpus level: a typical expansion set
    # full of words no chunk contains must not push Manchester below the floor.
    helper["expand"] = {"branch", "location", "address", "city", "premises", "local", "visit"}
    evidence = agent.search_knowledge_base("Do you have a Manchester office?")
    assert "Our offices › Manchester" in evidence


# --- building the real thing ----------------------------------------------------------------


def test_open_retriever_syncs_the_index_and_honours_the_mode(tmp_path):
    settings = agent.Settings(
        embedder="hash", index_path=str(tmp_path / "idx"), retrieval_mode="lexical"
    )
    retriever, report = open_retriever(settings)
    try:
        assert report.added == report.total > 100 and report.removed == 0
        assert retriever.dense_weight == 0.0 and retriever.lexical_weight == 1.0
        assert retriever.search("0161 496 0170", k=1)[0].found_by == "lexical"
    finally:
        retriever.store.close()


class FakeStore:
    closed = False

    def close(self):
        self.closed = True


class FakeRetriever:
    def __init__(self):
        self.store = FakeStore()


def test_the_agent_builds_its_retriever_once_and_closes_it_at_exit(monkeypatch, trace):
    built, registered = [], []

    def fake_open(settings):
        built.append(settings)
        return FakeRetriever(), SyncReport(added=3, removed=0, unchanged=0)

    monkeypatch.setattr(agent, "_RETRIEVER", None)
    monkeypatch.setattr(agent, "open_retriever", fake_open)
    monkeypatch.setattr(agent.atexit, "register", registered.append)
    first = agent.get_retriever()
    assert agent.get_retriever() is first
    assert built == [agent.SETTINGS] and registered == [agent.close_retriever]
    [event] = trace.find("index")
    assert event.data == {"added": 3, "removed": 0, "unchanged": 0, "total": 3}

    agent.close_retriever()
    assert first.store.closed and agent._RETRIEVER is None
    agent.close_retriever()  # idempotent
