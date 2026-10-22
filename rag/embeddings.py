"""
Text embeddings behind one small interface.

An embedder turns text into unit-length float32 vectors so that cosine
similarity is a dot product. Two implementations share the interface:

- ``LocalEmbedder`` runs a sentence-embedding model on the CPU through ONNX
  Runtime (via fastembed). No PyTorch, no GPU, no API key; the model is
  downloaded once to a cache folder. This is what the product uses.
- ``HashEmbedder`` is a deterministic stand-in with no model at all: feature
  hashing of words and word pairs. It is not semantic, but it is fast, needs
  no download, and gives the same vector on every machine, which is what the
  tests and CI need. It must never be used to answer customers.

Queries and documents are embedded by separate methods because some models
expect a short instruction in front of a query (BGE does). The hash embedder
treats both the same.

Why fastembed rather than sentence-transformers: measured against this
project's lockfile, sentence-transformers adds 37 packages including PyTorch
and the CUDA libraries; fastembed adds 12, the largest being ONNX Runtime.
The model (BAAI/bge-small-en-v1.5, 384 dimensions, MIT licence) is the same
either way.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

TOKEN = re.compile(r"[a-z0-9£%]+")

# BGE v1.5 models recommend this instruction on the query side for short
# query -> long passage retrieval; documents are embedded without it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": BGE_QUERY_PREFIX,
}


@runtime_checkable
class Embedder(Protocol):
    """What the vector store needs from an embedding model."""

    @property
    def name(self) -> str:
        """Identifies the model and its dimension; part of the index's identity."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length float32 rows, one per text, shape (len(texts), dimension)."""

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def unit_rows(rows: Iterable[Sequence[float]] | np.ndarray, dimension: int) -> np.ndarray:
    """Stack rows into a float32 matrix and scale each non-zero row to length 1."""
    matrix = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.float32)
    if matrix.size == 0:
        return np.zeros((0, dimension), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"expected rows of {dimension} values, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# =========================================================
# HASH EMBEDDER (tests, CI)
# =========================================================


class HashEmbedder:
    """Deterministic feature-hashing embeddings: words and adjacent word pairs are
    hashed to a fixed number of buckets with a sign, counted and normalised.

    Texts that share words score higher than texts that do not; that is all it
    promises. It exists so retrieval, indexing and syncing can be tested exactly,
    on any machine, without a model download.
    """

    def __init__(self, dimension: int = 256):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hash-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        tokens = TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            # blake2b, not hash(): Python salts str hashes per process.
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return unit_rows([self._vector(text) for text in texts], self._dimension)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents(texts)


# =========================================================
# LOCAL ONNX EMBEDDER (the product)
# =========================================================


def supported_models() -> dict[str, int]:
    """Model name -> embedding dimension for every model fastembed can run.

    Imported lazily so that the hash embedder, and therefore the tests, never
    need fastembed or ONNX Runtime loaded.
    """
    from fastembed import TextEmbedding

    return {entry["model"]: int(entry["dim"]) for entry in TextEmbedding.list_supported_models()}


class LocalEmbedder:
    """A sentence-embedding model run locally through ONNX Runtime.

    The model is loaded on first use, not at construction, so settings can be
    validated and the index opened without paying the load time; the dimension
    is known from fastembed's model table before anything is downloaded.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path = ".models",
        query_prefix: str | None = None,
        model_factory: Callable[..., object] | None = None,
        dimension: int | None = None,
    ):
        """``model_factory`` and ``dimension`` exist for tests, which substitute a
        fake model so the wrapper's behaviour can be checked without a download.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.query_prefix = (
            QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix
        )
        self._factory = model_factory
        self._model = None
        if dimension is None:
            table = supported_models()
            if model_name not in table:
                raise ValueError(
                    f"unknown embedding model {model_name!r}; fastembed supports: "
                    + ", ".join(sorted(table))
                )
            dimension = table[model_name]
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"{self.model_name}-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self):
        if self._model is None:
            if self._factory is not None:
                self._model = self._factory(model_name=self.model_name, cache_dir=self.cache_dir)
            else:
                from fastembed import TextEmbedding

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(
                    model_name=self.model_name, cache_dir=str(self.cache_dir)
                )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return unit_rows(self.model.embed(list(texts)), self._dimension)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        prefixed = [self.query_prefix + text for text in texts]
        return unit_rows(self.model.embed(prefixed), self._dimension)


# =========================================================
# FACTORY
# =========================================================


def build_embedder(settings) -> Embedder:
    """The embedder the settings ask for. Loads no model: that happens on first use."""
    if settings.embedder == "hash":
        return HashEmbedder()
    return LocalEmbedder(model_name=settings.embedding_model, cache_dir=settings.model_cache)
