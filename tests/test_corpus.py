"""Loading and chunking the corpus: structure, limits, provenance and stability."""

from pathlib import Path

import pytest

import agent
from rag import corpus
from rag.corpus import (
    Chunk,
    CorpusError,
    chunk_corpus,
    chunk_document,
    load_corpus,
    load_document,
    split_sections,
    window_words,
)

# --- the real corpus -------------------------------------------------------


@pytest.fixture(scope="module")
def documents():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(documents):
    return chunk_corpus(documents)


def test_the_corpus_loads_every_document_except_the_readme(documents):
    names = {document.doc_id for document in documents}
    assert "readme" not in {name.lower() for name in names}
    assert len(documents) >= 12
    assert {"fees-and-charges", "complaints-procedure", "contact-and-offices"} <= names


def test_every_document_has_complete_provenance(documents):
    for document in documents:
        assert document.title and document.url.startswith("https://www.fernhillwealth.example/")
        assert document.version and document.last_reviewed and document.owner


def test_every_chunk_is_within_the_word_budget(chunks):
    assert all(chunk.word_count <= corpus.MAX_WORDS for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_ids_are_unique_and_prefixed_with_their_document(chunks):
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk.chunk_id.startswith(chunk.doc_id + ":") for chunk in chunks)


def test_every_chunk_carries_title_section_and_url(chunks):
    for chunk in chunks:
        assert chunk.title and chunk.section and chunk.url
        assert chunk.with_context().startswith(f"{chunk.title} › {chunk.section}\n")


def test_positions_count_up_within_each_document(chunks):
    by_doc: dict[str, list[int]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.position)
    for positions in by_doc.values():
        assert positions == list(range(len(positions)))


def test_chunk_ids_are_stable_across_runs(documents):
    first = [chunk.chunk_id for chunk in chunk_corpus(documents)]
    second = [chunk.chunk_id for chunk in chunk_corpus(documents)]
    assert first == second


def test_the_facts_the_agent_relies_on_are_in_the_corpus(chunks):
    text = " ".join(chunk.text for chunk in chunks)
    for fact in ("0117 496 0100", "0800 023 4567", "22 Harbourside Walk", "£20,000", "£60,000"):
        assert fact in text, fact
    # The escalation routes give these teams' numbers and addresses.
    for fact in (
        agent.CLIENT_RESOLUTION,
        agent.LIFE_EVENTS_TEAM,
        "complaints@fernhillwealth.example",
        "lifeevents@fernhillwealth.example",
        "Fraud and Security",
    ):
        assert fact in text, fact


# --- sections ----------------------------------------------------------------


def test_text_before_the_first_heading_becomes_the_overview():
    body = "# Title\n\nIntro sentence.\n\n## First\n\nBody one.\n"
    assert split_sections(body) == [("Overview", "Intro sentence."), ("First", "Body one.")]


def test_sub_headings_are_named_under_their_parent():
    body = "## Offices\n\nSix offices.\n\n### Bristol\n\nHead office.\n\n### London\n\nCity.\n"
    assert [name for name, _ in split_sections(body)] == [
        "Offices",
        "Offices › Bristol",
        "Offices › London",
    ]


def test_a_document_without_headings_is_one_overview_section():
    assert split_sections("Just a paragraph.\n\nAnother.") == [
        ("Overview", "Just a paragraph.\n\nAnother.")
    ]


# --- windows -------------------------------------------------------------------


def test_short_text_is_a_single_window():
    assert window_words("One. Two. Three.", max_words=10, overlap=2) == ["One. Two. Three."]


def test_long_text_is_split_at_sentence_ends_with_overlap():
    text = " ".join(f"Sentence number {i} ends here." for i in range(12))  # 60 words
    windows = window_words(text, max_words=25, overlap=5)
    assert len(windows) >= 3
    assert all(len(window.split()) <= 25 for window in windows)
    # Overlap: the first words of window 2 are the last words of window 1.
    tail = windows[0].split()[-5:]
    assert windows[1].split()[:5] == tail


def test_every_word_survives_windowing():
    text = " ".join(f"w{i}." for i in range(100))
    windows = window_words(text, max_words=30, overlap=5)
    seen = {word for window in windows for word in window.split()}
    assert seen == set(text.split())


def test_a_single_sentence_longer_than_the_budget_is_hard_split():
    text = " ".join(f"w{i}" for i in range(50))  # no sentence ends at all
    windows = window_words(text, max_words=20, overlap=4)
    assert all(len(window.split()) <= 20 for window in windows)
    assert {word for window in windows for word in window.split()} == set(text.split())


# --- front matter and errors ----------------------------------------------------


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


GOOD_FRONT = (
    "---\ntitle: T\nurl: https://www.fernhillwealth.example/t/\nversion: 2026-10\n"
    "last_reviewed: 2026-10-01\nowner: Test\n---\n"
)


def test_a_valid_document_parses(tmp_path):
    document = load_document(write(tmp_path, "t.md", GOOD_FRONT + "# T\n\n## A\n\nBody."))
    assert document.doc_id == "t" and document.title == "T" and document.body.startswith("# T")


def test_missing_front_matter_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="missing front matter"):
        load_document(write(tmp_path, "t.md", "# No front matter\n\nBody."))


def test_missing_required_field_is_rejected(tmp_path):
    front = GOOD_FRONT.replace("owner: Test\n", "")
    with pytest.raises(CorpusError, match="owner"):
        load_document(write(tmp_path, "t.md", front + "Body."))


def test_an_empty_folder_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="no documents"):
        load_corpus(tmp_path)


def test_duplicate_text_in_one_section_is_rejected(tmp_path):
    document = load_document(write(tmp_path, "t.md", GOOD_FRONT + "## A\n\nSame.\n\n## A\n\nSame."))
    with pytest.raises(CorpusError, match="duplicate chunk ids"):
        chunk_corpus([document])


def test_chunk_metadata_carries_version_and_review_date(tmp_path):
    document = load_document(write(tmp_path, "t.md", GOOD_FRONT + "## A\n\nBody."))
    [chunk] = chunk_document(document)
    assert isinstance(chunk, Chunk)
    assert chunk.metadata == {"version": "2026-10", "last_reviewed": "2026-10-01", "owner": "Test"}
