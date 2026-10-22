"""
Retrieval metrics over the labelled set, for each search mode, offline.

For every case with gold sections, each mode (hybrid, lexical, dense) is asked
for its top results and the rank of the first gold chunk is recorded. From
those ranks: recall@k (did a gold chunk appear in the top k) for k = 1, 3, 6,
and MRR (the mean of 1/rank, 0 for a miss). The misses are listed by id, so a
number is never the end of the story.

No model call is made: the metrics are about retrieval alone, before query
expansion and before the reranker, and they take seconds. They are also the
calibration for the dense floor: for cases with gold, the cosine score of
the gold chunk; for cases without (advice, out of scope, greetings, case
numbers), the best cosine any chunk reached. The floor should sit between
those two distributions, and the report shows whether it does.

    python -m eval.retrieval                      # all modes, report to reports/retrieval.json
    python -m eval.retrieval --mode lexical       # one mode
    python -m eval.retrieval --sweep              # also try fusion weights and rank constants
    python -m eval.retrieval --out somewhere.json

The sweep answers "are the fusion weights right?": it re-runs the hybrid mode
with the dense side weighted from half to three times the lexical side, and
with the rank constant at 20 and 60, and tabulates recall@6 and MRR for each.

The numbers that matter are the ones from the real embedding model; CI runs
this with the hash embedder only to prove the harness itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from config import Settings
from eval.dataset import Case, load_cases
from rag.corpus import chunk_corpus
from rag.embeddings import build_embedder
from rag.lexical import BM25Index
from rag.retrieval import HybridRetriever, weights_for
from rag.store import open_store

MODES = ("hybrid", "lexical", "dense")
KS = (1, 3, 6)
DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "reports" / "retrieval.json"
SWEEP_DENSE_WEIGHTS = (0.5, 1.0, 1.5, 2.0, 3.0)
SWEEP_RRF_KS = (20, 60)


@dataclass(frozen=True)
class CaseOutcome:
    id: str
    kind: str
    gold_rank: int | None  # 1-based rank of the first gold chunk; None if not in the top pool
    gold_dense: float | None  # cosine of that chunk, when the dense side returned it
    top_dense: float | None  # best cosine of any result


def rank_of_gold(results, gold_ids: frozenset[str]) -> tuple[int | None, float | None]:
    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in gold_ids:
            return rank, result.dense_score
    return None, None


def evaluate_case(retriever: HybridRetriever, case: Case, pool: int) -> CaseOutcome:
    results = retriever.search(case.question, k=pool)
    rank, gold_dense = rank_of_gold(results, case.gold_chunk_ids) if case.gold else (None, None)
    dense_scores = [r.dense_score for r in results if r.dense_score is not None]
    return CaseOutcome(
        id=case.id,
        kind=case.kind,
        gold_rank=rank,
        gold_dense=gold_dense,
        top_dense=max(dense_scores) if dense_scores else None,
    )


def summarise_ranks(outcomes: list[CaseOutcome]) -> dict:
    """recall@k and MRR over outcomes that have gold (gold_rank is None for a miss)."""
    if not outcomes:
        return {"n": 0}
    summary: dict = {"n": len(outcomes)}
    for k in KS:
        hits = sum(1 for o in outcomes if o.gold_rank is not None and o.gold_rank <= k)
        summary[f"recall@{k}"] = round(hits / len(outcomes), 3)
    summary["mrr"] = round(
        sum(1 / o.gold_rank for o in outcomes if o.gold_rank is not None) / len(outcomes), 3
    )
    return summary


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=10) if len(ordered) >= 2 else [ordered[0]] * 9
    return {
        "n": len(ordered),
        "min": round(ordered[0], 3),
        "p10": round(q[0], 3),
        "median": round(statistics.median(ordered), 3),
        "p90": round(q[8], 3),
        "max": round(ordered[-1], 3),
    }


def evaluate_mode(retriever: HybridRetriever, cases: list[Case], pool: int) -> dict:
    with_gold = [case for case in cases if case.gold]
    without_gold = [case for case in cases if not case.gold]
    outcomes = [evaluate_case(retriever, case, pool) for case in with_gold]
    by_kind: dict[str, list[CaseOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_kind[outcome.kind].append(outcome)

    report = {
        **summarise_ranks(outcomes),
        "pool": pool,
        "by_kind": {kind: summarise_ranks(group) for kind, group in sorted(by_kind.items())},
        "misses": sorted(o.id for o in outcomes if o.gold_rank is None),
        "outcomes": [asdict(o) for o in outcomes],
    }
    if retriever.dense_weight > 0:
        no_gold = [evaluate_case(retriever, case, pool) for case in without_gold]
        report["dense_calibration"] = {
            "gold_chunk_cosine": quantiles([o.gold_dense for o in outcomes if o.gold_dense]),
            "best_cosine_without_gold": quantiles([o.top_dense for o in no_gold if o.top_dense]),
        }
        report["outcomes"] += [asdict(o) for o in no_gold]
    return report


def sweep(lexical: BM25Index, store, cases: list[Case], pool: int) -> list[dict]:
    """Hybrid retrieval under each fusion setting: dense weight and rank constant."""
    rows = []
    for rrf_k in SWEEP_RRF_KS:
        for dense_weight in SWEEP_DENSE_WEIGHTS:
            retriever = HybridRetriever(
                lexical, store, rrf_k=rrf_k, lexical_weight=1.0, dense_weight=dense_weight
            )
            with_gold = [case for case in cases if case.gold]
            outcomes = [evaluate_case(retriever, case, pool) for case in with_gold]
            summary = summarise_ranks(outcomes)
            rows.append(
                {
                    "rrf_k": rrf_k,
                    "dense_weight": dense_weight,
                    "recall@1": summary["recall@1"],
                    "recall@6": summary["recall@6"],
                    "mrr": summary["mrr"],
                    "misses": sum(1 for o in outcomes if o.gold_rank is None),
                }
            )
    return rows


def run(
    settings: Settings, modes: tuple[str, ...] = MODES, pool: int = 10, with_sweep: bool = False
) -> dict:
    """Evaluate every mode on one shared index and corpus; return the report."""
    started = time.perf_counter()
    cases = load_cases()
    chunks = chunk_corpus()
    embedder = build_embedder(settings)
    store = open_store(settings, embedder)
    try:
        store.sync(chunks)
        lexical = BM25Index(chunks)
        results = {
            mode: evaluate_mode(
                HybridRetriever(lexical, store, rrf_k=settings.rrf_k, **weights_for(mode)),
                cases,
                pool,
            )
            for mode in modes
        }
        sweep_rows = sweep(lexical, store, cases, pool) if with_sweep else None
    finally:
        store.close()
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": embedder.name,
        "cases": len(cases),
        "cases_with_gold": sum(1 for case in cases if case.gold),
        "chunks": len(chunks),
        "dense_floor": settings.dense_floor,
        "relevance_floor": settings.relevance_floor,
        "rrf_k": settings.rrf_k,
        "modes": results,
        "seconds": round(time.perf_counter() - started, 1),
    }
    if sweep_rows is not None:
        report["sweep"] = sweep_rows
    return report


def render(report: dict) -> str:
    lines = [
        f"{report['cases_with_gold']} cases with gold of {report['cases']}, "
        f"{report['chunks']} chunks, embedder {report['embedder']}, fusion k {report['rrf_k']}",
        f"{'mode':8} {'recall@1':>9} {'recall@3':>9} {'recall@6':>9} {'MRR':>6}  misses",
    ]
    for mode, summary in report["modes"].items():
        lines.append(
            f"{mode:8} {summary['recall@1']:>9.3f} {summary['recall@3']:>9.3f} "
            f"{summary['recall@6']:>9.3f} {summary['mrr']:>6.3f}  {len(summary['misses'])}"
        )
    hybrid = report["modes"].get("hybrid") or report["modes"].get("dense")
    if hybrid and "dense_calibration" in hybrid:
        gold = hybrid["dense_calibration"]["gold_chunk_cosine"]
        none = hybrid["dense_calibration"]["best_cosine_without_gold"]
        lines.append(
            f"dense floor {report['dense_floor']}: gold chunks score "
            f"min {gold.get('min')} / p10 {gold.get('p10')} / median {gold.get('median')}; "
            f"questions without gold peak at p90 {none.get('p90')} / max {none.get('max')}"
        )
    if "sweep" in report:
        lines.append(
            f"\nsweep (lexical weight 1.0):\n{'rrf_k':>6} {'dense_w':>8} "
            f"{'recall@1':>9} {'recall@6':>9} {'MRR':>6}  misses"
        )
        for row in report["sweep"]:
            lines.append(
                f"{row['rrf_k']:>6} {row['dense_weight']:>8.2f} {row['recall@1']:>9.3f} "
                f"{row['recall@6']:>9.3f} {row['mrr']:>6.3f}  {row['misses']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.retrieval", description=__doc__)
    parser.add_argument("--mode", choices=MODES, help="one mode instead of all three")
    parser.add_argument("--pool", type=int, default=10, help="results examined per question")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="where to write the JSON")
    parser.add_argument(
        "--sweep", action="store_true", help="also try fusion weights and rank constants"
    )
    args = parser.parse_args(argv)

    modes = (args.mode,) if args.mode else MODES
    report = run(Settings.from_env(), modes=modes, pool=args.pool, with_sweep=args.sweep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(render(report))
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
