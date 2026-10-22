"""The retrieval metrics: the arithmetic, the per-mode report, the calibration
figures, and the CLI. Runs on the hash embedder in memory; the real numbers
come from the real model."""

import json

import pytest

from config import Settings
from eval import retrieval as metrics
from eval.dataset import load_cases
from eval.retrieval import CaseOutcome, evaluate_mode, quantiles, rank_of_gold, summarise_ranks
from rag.corpus import chunk_corpus
from rag.embeddings import HashEmbedder
from rag.lexical import BM25Index
from rag.retrieval import HybridRetriever, weights_for
from rag.store import VectorStore

# --- arithmetic ---------------------------------------------------------------------


def outcome(rank, kind="factual"):
    return CaseOutcome(id=f"c{rank}", kind=kind, gold_rank=rank, gold_dense=None, top_dense=None)


def test_recall_and_mrr_from_ranks():
    summary = summarise_ranks([outcome(1), outcome(2), outcome(None), outcome(7)])
    assert summary["n"] == 4
    assert summary["recall@1"] == 0.25
    assert summary["recall@3"] == 0.5
    assert summary["recall@6"] == 0.5
    assert summary["mrr"] == pytest.approx((1 + 1 / 2 + 0 + 1 / 7) / 4, abs=0.001)


def test_no_outcomes_is_reported_as_n_zero():
    assert summarise_ranks([]) == {"n": 0}


def test_quantiles_cover_the_range():
    q = quantiles([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert q["n"] == 10 and q["min"] == 0.1 and q["max"] == 1.0
    assert q["p10"] <= q["median"] <= q["p90"]
    assert quantiles([]) == {"n": 0}
    assert quantiles([0.5])["median"] == 0.5


def test_rank_of_gold_finds_the_first_gold_chunk():
    class R:
        def __init__(self, cid, dense):
            self.chunk = type("C", (), {"chunk_id": cid})()
            self.dense_score = dense

    results = [R("a", 0.9), R("gold-2", 0.8), R("gold-1", 0.7)]
    assert rank_of_gold(results, frozenset({"gold-1", "gold-2"})) == (2, 0.8)
    assert rank_of_gold(results, frozenset({"zzz"})) == (None, None)


# --- a mode over the real set --------------------------------------------------------


@pytest.fixture(scope="module")
def parts():
    chunks = chunk_corpus()
    store = VectorStore(HashEmbedder())
    store.sync(chunks)
    return chunks, store, load_cases()


def test_a_mode_report_has_every_part(parts):
    chunks, store, cases = parts
    retriever = HybridRetriever(BM25Index(chunks), store, **weights_for("hybrid"))
    report = evaluate_mode(retriever, cases, pool=10)
    with_gold = [case for case in cases if case.gold]
    assert report["n"] == len(with_gold) and report["pool"] == 10
    assert 0.0 <= report["recall@1"] <= report["recall@3"] <= report["recall@6"] <= 1.0
    assert sum(group["n"] for group in report["by_kind"].values()) == report["n"]
    assert set(report["misses"]) <= {case.id for case in with_gold}
    calibration = report["dense_calibration"]
    assert calibration["gold_chunk_cosine"]["n"] > 0
    assert calibration["best_cosine_without_gold"]["n"] == len(cases) - len(with_gold)


def test_lexical_mode_reports_no_dense_calibration(parts):
    chunks, store, cases = parts
    retriever = HybridRetriever(BM25Index(chunks), store, **weights_for("lexical"))
    report = evaluate_mode(retriever, cases[:20], pool=10)
    assert "dense_calibration" not in report


def test_exact_strings_are_found_lexically(parts):
    chunks, store, cases = parts
    retriever = HybridRetriever(BM25Index(chunks), store, **weights_for("lexical"))
    report = evaluate_mode(retriever, [c for c in cases if c.kind == "exact-string"], pool=10)
    assert report["recall@1"] == 1.0


# --- the whole run and the CLI --------------------------------------------------------


def test_run_evaluates_every_mode_on_one_index(tmp_path):
    settings = Settings(embedder="hash", index_path=str(tmp_path / "idx"))
    report = metrics.run(settings, pool=5)
    assert set(report["modes"]) == {"hybrid", "lexical", "dense"}
    assert report["embedder"] == "hash-256" and report["cases"] == report["cases_with_gold"] + sum(
        1 for case in load_cases() if not case.gold
    )
    assert report["dense_floor"] == settings.dense_floor
    text = metrics.render(report)
    assert text.count("\n") >= 4 and "lexical" in text and "dense floor" in text


def test_the_cli_writes_the_report_where_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SUPPORT_BOT_EMBEDDER", "hash")
    monkeypatch.setenv("SUPPORT_BOT_INDEX_PATH", str(tmp_path / "idx"))
    out = tmp_path / "reports" / "retrieval.json"
    assert metrics.main(["--mode", "lexical", "--pool", "6", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert list(report["modes"]) == ["lexical"] and report["modes"]["lexical"]["pool"] == 6
    assert "report written to" in capsys.readouterr().out
