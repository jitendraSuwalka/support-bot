"""
The vector index over the corpus chunks, on Qdrant.

Qdrant runs here in *local mode*: the index is a folder on disk (or memory,
for tests) and no server is involved. The client API is the same one a Qdrant
server speaks, so moving to a server is a change of settings, not of code.
Local mode is single-process: one program opens the index at a time.

The index is kept in step with the corpus by ``sync``. Chunk ids are content
hashes (see ``rag.corpus``), so the comparison is a set difference: chunks
whose id is already stored are left alone, new ids are embedded and added,
ids that no longer exist in the corpus are removed. Editing one sentence in
one document therefore re-embeds one chunk, not the corpus.

The collection is named after the embedder, so changing model or dimension
cannot mix incompatible vectors: a new embedder gets a new, empty collection.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models

from rag.corpus import Chunk
from rag.embeddings import Embedder

COLLECTION_PREFIX = "fernhill"
SCROLL_PAGE = 256
UPSERT_BATCH = 64


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, -1..1; higher is closer


@dataclass(frozen=True)
class SyncReport:
    added: int
    removed: int
    unchanged: int

    @property
    def total(self) -> int:
        return self.added + self.unchanged

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "total": self.total}


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be integers or UUIDs; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def collection_name(embedder: Embedder, prefix: str = COLLECTION_PREFIX) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", embedder.name.lower()).strip("-")
    return f"{prefix}-{slug}"


def chunk_to_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "url": chunk.url,
        "text": chunk.text,
        "position": chunk.position,
        "metadata": dict(chunk.metadata),
    }


def payload_to_chunk(payload: dict) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        title=payload["title"],
        section=payload["section"],
        url=payload["url"],
        text=payload["text"],
        position=int(payload["position"]),
        metadata=dict(payload.get("metadata") or {}),
    )


class VectorStore:
    """Chunks and their vectors, searchable by meaning."""

    def __init__(
        self,
        embedder: Embedder,
        client: QdrantClient | None = None,
        *,
        path: str | None = None,
        url: str | None = None,
        prefix: str = COLLECTION_PREFIX,
    ):
        """Open (or create) the index.

        Exactly one of ``client``, ``path`` or ``url`` selects where the index
        lives; none of them means in memory, which is what tests use.
        """
        if sum(x is not None for x in (client, path, url)) > 1:
            raise ValueError("pass at most one of client, path or url")
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        self.embedder = embedder
        self.collection = collection_name(embedder, prefix)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=embedder.dimension, distance=models.Distance.COSINE
                ),
            )

    # --- contents -----------------------------------------------------------

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def stored_ids(self) -> set[str]:
        """Every chunk id in the index, read page by page without the vectors."""
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            ids.update(point.payload["chunk_id"] for point in points)
            if offset is None:
                return ids

    def sync(self, chunks: list[Chunk]) -> SyncReport:
        """Make the index match ``chunks`` exactly, embedding only what is new."""
        wanted = {chunk.chunk_id: chunk for chunk in chunks}
        if len(wanted) != len(chunks):
            raise ValueError("duplicate chunk ids")
        stored = self.stored_ids()

        stale = sorted(stored - wanted.keys())
        if stale:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=[point_id(c) for c in stale]),
            )

        fresh = [chunk for chunk in chunks if chunk.chunk_id not in stored]
        for start in range(0, len(fresh), UPSERT_BATCH):
            batch = fresh[start : start + UPSERT_BATCH]
            vectors = self.embedder.embed_documents([chunk.with_context() for chunk in batch])
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(chunk.chunk_id),
                        vector=vector.tolist(),
                        payload=chunk_to_payload(chunk),
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
        return SyncReport(added=len(fresh), removed=len(stale), unchanged=len(chunks) - len(fresh))

    # --- search ---------------------------------------------------------------

    def search(self, query: str, k: int = 5, doc_id: str | None = None) -> list[Hit]:
        """The ``k`` chunks closest to the query, optionally within one document."""
        if k < 1:
            raise ValueError("k must be at least 1")
        vector = self.embedder.embed_queries([query])[0]
        query_filter = None
        if doc_id is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            Hit(chunk=payload_to_chunk(point.payload), score=float(point.score))
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()


def open_store(settings, embedder: Embedder) -> VectorStore:
    """The store the settings describe: a Qdrant server if qdrant_url is set,
    otherwise the on-disk index at index_path."""
    if settings.qdrant_url:
        return VectorStore(embedder, url=settings.qdrant_url)
    return VectorStore(embedder, path=settings.index_path)
