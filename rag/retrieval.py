"""
Hybrid retrieval: lexical and dense search fused by reciprocal rank.

Each method is good at what the other is bad at. Dense search (the vector
index) matches meaning, so "how much can I put in my ISA" finds "the ISA
allowance is £20,000". Lexical search (BM25) matches strings, so a phone
number, a client reference or a postcode finds the one chunk that contains
it. Running both and merging the two rankings gives the strengths of each.

The merge is reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009):
a chunk's fused score is the sum over rankings of 1 / (k + rank). Only the
ranks are used, never the raw scores, so the two methods' incomparable
scales (a BM25 share of ceiling; a cosine) never have to be reconciled, and
a chunk that both methods rank highly rises above one that only one method
likes. k = 60 is the value from the paper; it damps the advantage of rank 1
over rank 2 so that agreement between methods matters more than either
method's top pick.

The result keeps each method's rank and score, so a caller (and a reader of
the trace) can see why a chunk was retrieved.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag.corpus import Chunk, chunk_corpus, load_corpus
from rag.embeddings import build_embedder
from rag.lexical import BM25Index, tokenize
from rag.store import SyncReport, VectorStore, open_store

RRF_K = 60
MIN_POOL = 20  # how many candidates each method contributes, at least


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused reciprocal-rank score
    lexical_rank: int | None  # 1-based; None if lexical search did not return it
    dense_rank: int | None
    lexical_score: float | None  # share of the query's BM25 ceiling, 0..1
    dense_score: float | None  # cosine similarity

    @property
    def found_by(self) -> str:
        if self.lexical_rank is not None and self.dense_rank is not None:
            return "both"
        return "lexical" if self.lexical_rank is not None else "dense"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse ranked lists of ids. Each list contributes weight / (k + rank) per id."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused


class HybridRetriever:
    """BM25 and the vector store searched side by side and merged."""

    def __init__(
        self,
        lexical: BM25Index,
        store: VectorStore,
        *,
        rrf_k: int = RRF_K,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight == 0:
            raise ValueError("weights must be non-negative and not both zero")
        self.lexical = lexical
        self.store = store
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        extra_terms: Iterable[str] = (),
        pool: int | None = None,
    ) -> list[RetrievedChunk]:
        """The ``k`` best chunks for the query.

        ``extra_terms`` are added to the lexical query only (query expansion
        belongs to the lexical side; the dense side already sees meaning).
        Each method contributes up to ``pool`` candidates before fusion.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        pool = pool or max(4 * k, MIN_POOL)

        lexical_hits = []
        if self.lexical_weight > 0:
            terms = set(tokenize(query)) | set(tokenize(" ".join(extra_terms)))
            lexical_hits = self.lexical.search(terms, k=pool)
        dense_hits = self.store.search(query, k=pool) if self.dense_weight > 0 else []

        lexical_by_id = {h.chunk.chunk_id: (rank, h) for rank, h in enumerate(lexical_hits, 1)}
        dense_by_id = {h.chunk.chunk_id: (rank, h) for rank, h in enumerate(dense_hits, 1)}
        fused = reciprocal_rank_fusion(
            [list(lexical_by_id), list(dense_by_id)],
            weights=[self.lexical_weight, self.dense_weight],
            k=self.rrf_k,
        )

        results = []
        for chunk_id, score in fused.items():
            lexical = lexical_by_id.get(chunk_id)
            dense = dense_by_id.get(chunk_id)
            chunk = (lexical or dense)[1].chunk
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    lexical_rank=lexical[0] if lexical else None,
                    dense_rank=dense[0] if dense else None,
                    lexical_score=lexical[1].score if lexical else None,
                    dense_score=dense[1].score if dense else None,
                )
            )
        results.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return results[:k]


def build_retriever(chunks: Iterable[Chunk], store: VectorStore, **kwargs) -> HybridRetriever:
    """A retriever over an index that is already synced with ``chunks``."""
    return HybridRetriever(BM25Index(chunks), store, **kwargs)


def weights_for(mode: str) -> dict[str, float]:
    """Constructor weights for a retrieval mode: hybrid, lexical or dense."""
    if mode == "lexical":
        return {"lexical_weight": 1.0, "dense_weight": 0.0}
    if mode == "dense":
        return {"lexical_weight": 0.0, "dense_weight": 1.0}
    if mode == "hybrid":
        return {"lexical_weight": 1.0, "dense_weight": 1.0}
    raise ValueError(f"unknown retrieval mode {mode!r}")


def open_retriever(settings) -> tuple[HybridRetriever, SyncReport]:
    """The retriever the settings describe, over an index brought up to date
    with the corpus. Returns the sync report too, so the caller can log it."""
    chunks = chunk_corpus(load_corpus())
    store = open_store(settings, build_embedder(settings))
    report = store.sync(chunks)
    retriever = HybridRetriever(
        BM25Index(chunks), store, rrf_k=settings.rrf_k, **weights_for(settings.retrieval_mode)
    )
    return retriever, report
