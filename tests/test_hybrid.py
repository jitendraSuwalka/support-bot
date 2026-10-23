"""Hybrid retrieval: reciprocal rank fusion, and the retriever that runs BM25 and
the vector store together and reports where each one ranked a chunk.

The vector side uses the hash embedder, so these tests prove the plumbing —
fusion, provenance, weights, pools — not semantic quality (see -m model)."""

import pytest

from rag.embeddings import HashEmbedder
from rag.lexical import BM25Index
from rag.retrieval import (
    RRF_K,
    HybridRetriever,
    RetrievedChunk,
    build_retriever,
    reciprocal_rank_fusion,
)
from rag.store import VectorStore
from tests.conftest import make_chunk

# --- reciprocal rank fusion ---------------------------------------------------------


def test_rrf_sums_one_over_k_plus_rank_across_rankings():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_agreement_beats_a_single_top_rank():
    # "a" is second in both lists; "b" is first in one and absent from the other.
    fused = reciprocal_rank_fusion([["b", "a"], ["c", "a"]])
    assert fused["a"] > fused["b"] == fused["c"]


def test_rrf_weights_scale_each_rankings_contribution():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0], k=10)
    assert fused["a"] == pytest.approx(2 / 11) and fused["b"] == pytest.approx(1 / 11)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_default_k_is_the_papers_sixty():
    assert RRF_K == 60


# --- the retriever ----------------------------------------------------------------------


@pytest.fixture
def chunks():
    return [
        make_chunk("contact", "Manchester", "41 Mosley Street, Manchester. Call 0161 496 0170.", 0),
        make_chunk("contact", "Bristol", "Fernhill House, Bristol. Call 0117 496 0100.", 1),
        make_chunk("isa-guide", "Allowance", "The ISA allowance is £20,000 a tax year.", 0),
        make_chunk("pensions", "Annual allowance", "The pension annual allowance is £60,000.", 0),
        make_chunk("complaints", "How", "Write to the complaints team or call 0117 496 0355.", 0),
    ]


@pytest.fixture
def store(chunks):
    vs = VectorStore(HashEmbedder(dimension=64))
    vs.sync(chunks)
    return vs


@pytest.fixture
def retriever(chunks, store):
    return build_retriever(chunks, store)


def test_results_carry_both_methods_ranks_and_scores(retriever, chunks):
    [top] = retriever.search("0161 496 0170 Manchester", k=1)
    assert isinstance(top, RetrievedChunk)
    assert top.chunk == chunks[0]
    assert top.lexical_rank == 1 and top.dense_rank == 1 and top.found_by == "both"
    assert 0.0 < top.lexical_score <= 1.0 and -1.0 <= top.dense_score <= 1.0
    assert top.score == pytest.approx(2 / (RRF_K + 1))


def test_a_chunk_only_one_method_returns_is_still_a_result(retriever):
    # Lexical search returns only chunks sharing a term; dense returns the nearest
    # k whatever the distance. So a query whose words appear nowhere is dense-only.
    results = retriever.search("zzz qqq", k=2)
    assert results and all(r.found_by == "dense" and r.lexical_rank is None for r in results)
    assert all(r.lexical_score is None and r.dense_score is not None for r in results)


def test_results_are_sorted_by_fused_score_and_limited_to_k(retriever):
    results = retriever.search("allowance 0117 496", k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    with pytest.raises(ValueError):
        retriever.search("x", k=0)


def test_extra_terms_widen_the_lexical_query_only(chunks, store):
    lexical_only = HybridRetriever(BM25Index(chunks), store, dense_weight=0.0)
    without = lexical_only.search("where is your northern office", k=5)
    assert all(r.chunk.section != "Manchester" for r in without)
    with_expansion = lexical_only.search(
        "where is your northern office", k=5, extra_terms=["manchester"]
    )
    assert with_expansion[0].chunk.section == "Manchester"


def test_zero_weight_switches_a_method_off(chunks, store):
    lexical_only = HybridRetriever(BM25Index(chunks), store, dense_weight=0.0)
    results = lexical_only.search("pension annual allowance", k=3)
    assert all(r.found_by == "lexical" for r in results)
    assert results[0].chunk.doc_id == "pensions"

    dense_only = HybridRetriever(BM25Index(chunks), store, lexical_weight=0.0)
    results = dense_only.search("pension annual allowance", k=3)
    assert all(r.found_by == "dense" for r in results)
    assert results[0].chunk.doc_id == "pensions"


def test_weights_must_be_sensible(chunks, store):
    with pytest.raises(ValueError):
        HybridRetriever(BM25Index(chunks), store, lexical_weight=0.0, dense_weight=0.0)
    with pytest.raises(ValueError):
        HybridRetriever(BM25Index(chunks), store, lexical_weight=-1.0)


def test_the_pool_is_wider_than_k_so_fusion_has_something_to_fuse(chunks, store, monkeypatch):
    seen = {}
    real_search = store.search

    def spy(query, k=5, doc_id=None):
        seen["dense_k"] = k
        return real_search(query, k=k, doc_id=doc_id)

    monkeypatch.setattr(store, "search", spy)
    retriever = build_retriever(chunks, store)
    retriever.search("ISA", k=2)
    assert seen["dense_k"] == 20  # max(4 * k, MIN_POOL)
    retriever.search("ISA", k=2, pool=3)
    assert seen["dense_k"] == 3


def test_an_empty_corpus_gives_no_results():
    empty = HybridRetriever(BM25Index([]), VectorStore(HashEmbedder(dimension=32)))
    assert empty.search("anything") == []


# --- BM25's first choice is guaranteed a place -------------------------------------------
# Found by the evaluation (exact-06): asked about a missed call from 0161 496
# 0170, BM25 put the Manchester office first and the vector index never
# returned it, so chunks both methods ranked moderately outvoted the one
# chunk with the answer.


def dense_order(store, chunks, order):
    """Make the vector store return `order` (indexes into chunks) whatever the query."""
    from rag.store import Hit

    def fake_search(query, k=5, doc_id=None):
        return [Hit(chunk=chunks[i], score=0.9 - 0.1 * n) for n, i in enumerate(order[:k])]

    store.search = fake_search


def test_bm25s_first_choice_is_guaranteed_a_place_among_the_candidates(chunks, store):
    dense_order(store, chunks, [1, 4, 2, 3, 0])  # Manchester last for the vector side
    query = "missed call 0161 496 0170"
    plain = HybridRetriever(BM25Index(chunks), store).search(query, k=2)
    assert all(
        r.chunk != chunks[0] for r in plain
    )  # outvoted: both methods like Bristol/complaints

    guarded = HybridRetriever(BM25Index(chunks), store, lexical_first_within=2).search(query, k=2)
    assert guarded[1].chunk == chunks[0] and guarded[1].lexical_rank == 1
    assert len(guarded) == 2 and guarded[0] == plain[0]  # the rest of the order is untouched

    first = HybridRetriever(BM25Index(chunks), store, lexical_first_within=1).search(query, k=3)
    assert first[0].chunk == chunks[0]


def test_the_guarantee_changes_nothing_when_fusion_already_agrees(chunks, store):
    dense_order(store, chunks, [0, 1, 2, 3, 4])
    guarded = HybridRetriever(BM25Index(chunks), store, lexical_first_within=3)
    plain = HybridRetriever(BM25Index(chunks), store)
    query = "0161 496 0170"
    assert [r.chunk for r in guarded.search(query, k=3)] == [
        r.chunk for r in plain.search(query, k=3)
    ]


def test_the_guarantee_only_applies_when_both_methods_ran(chunks, store):
    dense_order(store, chunks, [1, 4, 2, 3, 0])
    lexical_only = HybridRetriever(BM25Index(chunks), store, dense_weight=0, lexical_first_within=1)
    assert lexical_only.search("call 0161 496 0170", k=2)[0].chunk == chunks[0]  # its own order
    dense_only = HybridRetriever(BM25Index(chunks), store, lexical_weight=0, lexical_first_within=1)
    assert dense_only.search("call 0161 496 0170", k=2)[0].chunk == chunks[1]  # untouched
    nothing_lexical = HybridRetriever(BM25Index(chunks), store, lexical_first_within=1)
    assert nothing_lexical.search("zzz qqq", k=2)[0].chunk == chunks[1]  # no BM25 hit to promote


def test_the_guarantee_depth_must_be_at_least_one(chunks, store):
    with pytest.raises(ValueError, match="lexical_first_within"):
        HybridRetriever(BM25Index(chunks), store, lexical_first_within=0)
