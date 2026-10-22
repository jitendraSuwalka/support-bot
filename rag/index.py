"""
Build the vector index from the corpus, or search it.

    python -m rag.index                      # sync the index with corpus/
    python -m rag.index --query "ISA limit"  # show the closest chunks

Settings come from the environment (SUPPORT_BOT_*; see config.py). The first
run with the default "local" embedder downloads the model, about 70 MB, to
the model cache; later runs reuse it. Syncing re-embeds only chunks whose
text changed since the last run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from config import Settings
from rag.corpus import chunk_corpus, load_corpus
from rag.embeddings import build_embedder
from rag.store import VectorStore, open_store


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


def search_index(settings: Settings, query: str, k: int, store: VectorStore | None = None) -> list:
    owned = store is None
    if owned:
        store = open_store(settings, build_embedder(settings))
    try:
        return store.search(query, k=k)
    finally:
        if owned:
            store.close()


def format_hit(rank: int, hit) -> str:
    chunk = hit.chunk
    preview = " ".join(chunk.text.split())[:160]
    return f"{rank}. {hit.score:.3f}  {chunk.title} › {chunk.section}\n   {chunk.url}\n   {preview}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.index", description=__doc__)
    parser.add_argument("--query", "-q", help="search the index instead of syncing it")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to show (default 5)")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.query:
        hits = search_index(settings, args.query, args.k)
        if not hits:
            print("The index is empty. Run `python -m rag.index` first.")
            return 1
        print("\n".join(format_hit(rank, hit) for rank, hit in enumerate(hits, start=1)))
        return 0

    print(json.dumps(sync_index(settings), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
