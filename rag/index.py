"""
Build the vector index from the corpus, or search it.

    python -m rag.index                                  # sync the index with corpus/
    python -m rag.index --query "ISA limit"              # hybrid search: BM25 + vectors
    python -m rag.index --query "ISA limit" --mode dense # one method on its own

Settings come from the environment (SUPPORT_BOT_*; see config.py). The first
run with the default "local" embedder downloads the model, about 70 MB, to
the model cache; later runs reuse it. Syncing re-embeds only chunks whose
text changed since the last run.

A search result line reads, for example,
``1. rrf 0.0325  lexical #1 (0.74)  dense #2 (0.713)  Title › Section``:
the fused score, then where each method ranked the chunk and that method's
own score (BM25 share of ceiling; cosine similarity). A dash means that
method did not return the chunk at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from config import Settings
from rag.corpus import chunk_corpus, load_corpus
from rag.embeddings import build_embedder
from rag.retrieval import HybridRetriever, RetrievedChunk, build_retriever
from rag.store import VectorStore, open_store

MODES = ("hybrid", "lexical", "dense")


def sync_index(settings: Settings, store: VectorStore | None = None) -> dict:
    """Load the corpus, chunk it, bring the index up to date; return a report."""
    started = time.perf_counter()
    documents = load_corpus()
    chunks = chunk_corpus(documents)
    owned = store is None
    if owned:
        store = open_store(settings, build_embedder(settings))
    try:
        report = store.sync(chunks)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedder": store.embedder.name,
            "collection": store.collection,
            "indexed": store.count(),
            **report.as_dict(),
            "seconds": round(time.perf_counter() - started, 1),
        }
    finally:
        if owned:
            store.close()


def search_index(
    settings: Settings,
    query: str,
    k: int,
    mode: str = "hybrid",
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Search with both methods, or one of them, over the live corpus and index."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    owned = store is None
    if owned:
        store = open_store(settings, build_embedder(settings))
    try:
        if store.count() == 0:
            return []
        retriever: HybridRetriever = build_retriever(
            chunk_corpus(load_corpus()),
            store,
            lexical_weight=0.0 if mode == "dense" else 1.0,
            dense_weight=0.0 if mode == "lexical" else 1.0,
        )
        return retriever.search(query, k=k)
    finally:
        if owned:
            store.close()


def _method(label: str, rank: int | None, score: float | None, digits: int) -> str:
    if rank is None:
        return f"{label} —"
    return f"{label} #{rank} ({score:.{digits}f})"


def format_hit(position: int, hit: RetrievedChunk) -> str:
    chunk = hit.chunk
    preview = " ".join(chunk.text.split())[:160]
    methods = (
        f"{_method('lexical', hit.lexical_rank, hit.lexical_score, 2)}  "
        f"{_method('dense', hit.dense_rank, hit.dense_score, 3)}"
    )
    return (
        f"{position}. rrf {hit.score:.4f}  {methods}  {chunk.title} › {chunk.section}\n"
        f"   {chunk.url}\n   {preview}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.index", description=__doc__)
    parser.add_argument("--query", "-q", help="search the index instead of syncing it")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to show (default 5)")
    parser.add_argument(
        "--mode", choices=MODES, default="hybrid", help="which method(s) to search with"
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.query:
        hits = search_index(settings, args.query, args.k, args.mode)
        if not hits:
            print("The index is empty. Run `python -m rag.index` first.")
            return 1
        print("\n".join(format_hit(rank, hit) for rank, hit in enumerate(hits, start=1)))
        return 0

    print(json.dumps(sync_index(settings), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
