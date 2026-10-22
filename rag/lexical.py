"""
Lexical retrieval: BM25 over the corpus chunks.

Dense retrieval finds meaning; it is poor at exact strings. A customer who
quotes a phone number, a client reference or a postcode wants the chunk
that contains those characters, and BM25 is the right tool for that. The
hybrid retriever (``rag.retrieval``) runs both and merges them.

Scores are normalised to the query's *ceiling*: the score a chunk would get
if every query term the corpus knows appeared in it with unbounded
frequency. A score is therefore a share, 0 to 1, and a relevance floor means
the same thing for a two-word query as for a ten-word one. Query terms the
corpus has never seen are left out of the ceiling, so an expansion term
such as "premises" that no chunk contains cannot drag a good match below
the floor (the Manchester incident, Phase 1).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from rag.corpus import Chunk

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "i", "im", "my", "me", "we", "you", "your", "how", "do", "does", "did",
    "what", "when", "where", "which", "who", "can", "could", "would", "should",
    "for", "in", "on", "at", "it", "this", "that", "with", "about", "from",
    "have", "has", "had", "be", "been", "get", "got", "any", "please", "thanks",
    "long", "take", "takes", "make", "made", "need", "want", "know", "tell",
    "help", "like", "just", "also", "there", "their", "am", "if", "so", "but",
    "not", "no", "yes", "all", "out", "up", "by", "as",
}  # fmt: skip

WORD = re.compile(r"[a-z0-9@.]+")

K1 = 1.2  # term-frequency saturation
B = 0.75  # length normalisation


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords removed and plurals folded.

    Returns a list, not a set: BM25 needs to know how often a term occurs in
    a chunk, not only whether it occurs. Digits survive, so "0161 496 0170"
    becomes three tokens that only the chunk containing that number matches.
    Dots inside a token are kept (email addresses, domains); a full stop at
    the end of a sentence is not part of the word before it.
    """
    tokens = []
    for word in WORD.findall(text.lower()):
        word = word.strip(".")
        if word in STOPWORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


@dataclass(frozen=True)
class LexicalHit:
    chunk: Chunk
    score: float  # share of the query's ceiling, 0..1
    matched: frozenset[str]  # the query terms this chunk contains


class BM25Index:
    """BM25 (Robertson/Spärck Jones idf, tf saturation, length normalisation)
    over the text of each chunk, headed by its title and section."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = K1, b: float = B):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(chunk.with_context())) for chunk in self.chunks]
        self._length = [sum(tf.values()) for tf in self._tf]
        self.avg_length = (sum(self._length) / len(self._length)) if self._length else 1.0
        total = len(self.chunks)
        document_frequency = Counter(term for tf in self._tf for term in tf)
        self.idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, terms: Iterable[str], k: int | None = None) -> list[LexicalHit]:
        """Chunks containing at least one known query term, best first.

        ``terms`` are already tokenized (so the caller can add expansion terms).
        Ties break on chunk id, so the order is the same on every run.
        """
        known = {term for term in terms if term in self.idf}
        if not known:
            return []
        ceiling = sum(self.idf[term] * (self.k1 + 1) for term in known)
        hits = []
        for chunk, tf, length in zip(self.chunks, self._tf, self._length, strict=True):
            matched = known & tf.keys()
            if not matched:
                continue
            length_norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            score = sum(
                self.idf[term] * tf[term] * (self.k1 + 1) / (tf[term] + length_norm)
                for term in matched
            )
            hits.append(LexicalHit(chunk=chunk, score=score / ceiling, matched=frozenset(matched)))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits[:k] if k is not None else hits
