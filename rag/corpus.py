"""
Load the corpus and split it into retrieval chunks.

A document is a Markdown file with a YAML front-matter block (title, url,
version, last_reviewed, owner). Chunking follows the document's own structure:
each `##` section becomes a chunk, and a section longer than the word budget
is split into overlapping windows so that no sentence is stranded at a
boundary. Every chunk carries the document title, section heading and source
URL, so an answer built from it can be traced to the page it came from.

Chunk ids are a hash of the document id, the section and the text, so they
are stable across runs and change only when the content does. That matters
for the vector store: a changed chunk is re-embedded, an unchanged one is not.

Run `python -m rag.corpus` for a summary of the corpus.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
REQUIRED_FIELDS = ("title", "url", "version", "last_reviewed", "owner")

MAX_WORDS = 220  # per chunk; a section longer than this is windowed
OVERLAP_WORDS = 40  # carried from the end of one window into the next

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$", re.M)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class CorpusError(ValueError):
    """A document is malformed. Raised at load time so a bad file cannot reach the index."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    url: str
    version: str
    last_reviewed: str
    owner: str
    body: str
    path: Path

    @property
    def word_count(self) -> int:
        return len(self.body.split())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    text: str
    position: int  # order within the document, from 0
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def with_context(self) -> str:
        """The text as it is embedded and shown to the model: headed by its source."""
        return f"{self.title} › {self.section}\n{self.text}"


# =========================================================
# LOADING
# =========================================================


def parse_front_matter(raw: str, path: Path) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(raw)
    if not match:
        raise CorpusError(f"{path.name}: missing front matter block")
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise CorpusError(f"{path.name}: bad front matter line {line!r}")
        fields[key.strip()] = value.strip().strip("'\"")
    missing = [name for name in REQUIRED_FIELDS if not fields.get(name)]
    if missing:
        raise CorpusError(f"{path.name}: front matter missing {', '.join(missing)}")
    return fields, raw[match.end() :]


def load_document(path: Path) -> Document:
    fields, body = parse_front_matter(path.read_text(encoding="utf-8"), path)
    return Document(
        doc_id=path.stem,
        title=fields["title"],
        url=fields["url"],
        version=fields["version"],
        last_reviewed=fields["last_reviewed"],
        owner=fields["owner"],
        body=body.strip(),
        path=path,
    )


def load_corpus(directory: Path | str = CORPUS_DIR) -> list[Document]:
    """Every Markdown file in the folder except the README, in name order."""
    directory = Path(directory)
    paths = sorted(p for p in directory.glob("*.md") if p.name.lower() != "readme.md")
    if not paths:
        raise CorpusError(f"no documents found in {directory}")
    return [load_document(path) for path in paths]


# =========================================================
# CHUNKING
# =========================================================


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a body into (heading, text) pairs along its `##` and `###` headings.

    Text before the first `##` (the introduction under the `#` title) becomes a
    section called "Overview". `###` sub-headings become their own sections,
    named "Parent › Child" so the chunk still says where it sits.
    """
    sections: list[tuple[str, str]] = []
    current_h2 = "Overview"
    current_name = "Overview"
    buffer: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append((current_name, text))
        buffer.clear()

    for line in body.splitlines():
        match = HEADING.match(line)
        if match and len(match.group(1)) == 1:
            continue  # the document title; the front matter already carries it
        if match:
            flush()
            level, name = len(match.group(1)), match.group(2)
            if level == 2:
                current_h2, current_name = name, name
            else:
                current_name = f"{current_h2} › {name}"
            continue
        buffer.append(line)
    flush()
    return sections


def window_words(text: str, max_words: int, overlap: int) -> list[str]:
    """Split text into windows of at most `max_words`, breaking at sentence ends.

    Each window after the first starts with the last `overlap` words of the one
    before it, so a fact that straddles a boundary is whole in at least one.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    windows: list[str] = []
    current: list[str] = []

    def emit():
        if current:
            windows.append(" ".join(current))

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current) + len(sentence_words) > max_words and current:
            emit()
            current = current[-overlap:] if overlap else []
        if len(sentence_words) > max_words:
            # A single sentence longer than the budget: hard-split it.
            for start in range(0, len(sentence_words), max_words - overlap):
                piece = sentence_words[start : start + max_words]
                emit_piece = current + piece
                windows.append(" ".join(emit_piece))
                current = []
            continue
        current.extend(sentence_words)
    emit()
    return windows


def chunk_id_for(doc_id: str, section: str, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}\n{section}\n{text}".encode()).hexdigest()
    return f"{doc_id}:{digest[:12]}"


def chunk_document(
    document: Document, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, text in split_sections(document.body):
        for window in window_words(text, max_words, overlap):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(document.doc_id, section, window),
                    doc_id=document.doc_id,
                    title=document.title,
                    section=section,
                    url=document.url,
                    text=window,
                    position=len(chunks),
                    metadata={
                        "version": document.version,
                        "last_reviewed": document.last_reviewed,
                        "owner": document.owner,
                    },
                )
            )
    return chunks


def chunk_corpus(documents: list[Document] | None = None, **kwargs) -> list[Chunk]:
    documents = load_corpus() if documents is None else documents
    chunks = [chunk for document in documents for chunk in chunk_document(document, **kwargs)]
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise CorpusError("duplicate chunk ids — two chunks have identical text in one section")
    return chunks


# =========================================================
# SUMMARY
# =========================================================


def summarise(documents: list[Document], chunks: list[Chunk]) -> dict:
    words = [chunk.word_count for chunk in chunks]
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "corpus_words": sum(document.word_count for document in documents),
        "chunk_words_min": min(words),
        "chunk_words_max": max(words),
        "chunk_words_mean": round(sum(words) / len(words), 1),
        "sections_per_document": round(len(chunks) / len(documents), 1),
    }


if __name__ == "__main__":
    import json

    documents = load_corpus()
    chunks = chunk_corpus(documents)
    print(json.dumps(summarise(documents, chunks), indent=2))
    for chunk in chunks[:3]:
        print(f"\n--- {chunk.chunk_id} ({chunk.word_count} words)\n{chunk.with_context()[:300]}")
