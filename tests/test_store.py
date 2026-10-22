"""The vector store: syncing re-embeds only what changed, search returns whole
chunks with provenance, the index persists, and embedders cannot be mixed.

Everything runs on Qdrant in memory with the hash embedder, so these tests
need no model, no server and no network."""

from pathlib import Path

import pytest

from config import Settings
from rag.corpus import Chunk, chunk_id_for
from rag.embeddings import HashEmbedder
from rag.store import (
    Hit,
    VectorStore,
    chunk_to_payload,
    collection_name,
    open_store,
    payload_to_chunk,
    point_id,
)


def make_chunk(doc_id: str, section: str, text: str, position: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id_for(doc_id, section, text),
        doc_id=doc_id,
        title=doc_id.replace("-", " ").title(),
        section=section,
        url=f"https://www.fernhillwealth.example/{doc_id}/",
        text=text,
        position=position,
        metadata={"version": "2026-10", "last_reviewed": "2026-10-20", "owner": "Test"},
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("isa-guide", "Allowance", "The ISA allowance is £20,000 each tax year.", 0),
        make_chunk("isa-guide", "Lifetime ISA", "A Lifetime ISA takes up to £4,000 a year.", 1),
        make_chunk("contact", "Manchester", "Peterloo Buildings, 41 Mosley Street, Manchester.", 0),
        make_chunk("pensions", "Annual allowance", "The pension annual allowance is £60,000.", 0),
    ]


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(HashEmbedder(dimension=64))


# --- syncing -------------------------------------------------------------------------


def test_the_first_sync_adds_everything(store, chunks):
    report = store.sync(chunks)
    assert report.as_dict() == {"added": 4, "removed": 0, "unchanged": 4 - 4, "total": 4}
    assert store.count() == 4


def test_a_second_sync_with_nothing_changed_embeds_nothing(store, chunks):
    store.sync(chunks)
    report = store.sync(chunks)
    assert (report.added, report.removed, report.unchanged) == (0, 0, 4)


def test_editing_one_chunk_replaces_only_that_chunk(store, chunks):
    store.sync(chunks)
    edited = list(chunks)
    edited[0] = make_chunk("isa-guide", "Allowance", "The ISA allowance is £20,000 a tax year.", 0)
    report = store.sync(edited)
    assert (report.added, report.removed, report.unchanged) == (1, 1, 3)
    assert store.count() == 4
    assert store.stored_ids() == {chunk.chunk_id for chunk in edited}


def test_removing_a_chunk_from_the_corpus_removes_it_from_the_index(store, chunks):
    store.sync(chunks)
    report = store.sync(chunks[:-1])
    assert (report.added, report.removed, report.unchanged) == (0, 1, 3)
    assert store.count() == 3


def test_duplicate_chunk_ids_are_rejected(store, chunks):
    with pytest.raises(ValueError, match="duplicate"):
        store.sync(chunks + [chunks[0]])


def test_sync_pages_through_a_large_index(chunks):
    """stored_ids reads the index a page at a time; prove it with more than one page."""
    from rag import store as store_module

    big = [make_chunk("doc", "S", f"Sentence {i} about topic {i % 7}.", i) for i in range(600)]
    vs = VectorStore(HashEmbedder(dimension=32))
    assert len(big) > store_module.SCROLL_PAGE
    vs.sync(big)
    assert len(vs.stored_ids()) == 600
    assert vs.sync(big).unchanged == 600


# --- searching -------------------------------------------------------------------------


def test_search_returns_whole_chunks_with_provenance(store, chunks):
    store.sync(chunks)
    [hit] = store.search("Peterloo Buildings Mosley Street Manchester", k=1)
    assert isinstance(hit, Hit)
    assert hit.chunk == chunks[2]  # every field survives the round trip, metadata included
    assert -1.0 <= hit.score <= 1.0


def test_the_chunk_sharing_the_most_words_ranks_first(store, chunks):
    store.sync(chunks)
    hits = store.search("pension annual allowance £60,000", k=4)
    assert hits[0].chunk.doc_id == "pensions"
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_k_limits_the_results_and_must_be_positive(store, chunks):
    store.sync(chunks)
    assert len(store.search("ISA", k=2)) == 2
    with pytest.raises(ValueError):
        store.search("ISA", k=0)


def test_search_can_be_restricted_to_one_document(store, chunks):
    store.sync(chunks)
    hits = store.search("allowance", k=4, doc_id="isa-guide")
    assert {hit.chunk.doc_id for hit in hits} == {"isa-guide"} and len(hits) == 2


def test_searching_an_empty_index_returns_nothing(store):
    assert store.search("anything") == []


# --- persistence and identity -----------------------------------------------------------


def test_the_index_persists_on_disk_and_is_not_re_embedded_on_reopen(tmp_path, chunks):
    first = VectorStore(HashEmbedder(dimension=64), path=str(tmp_path / "index"))
    first.sync(chunks)
    first.close()

    second = VectorStore(HashEmbedder(dimension=64), path=str(tmp_path / "index"))
    assert second.count() == 4
    assert second.sync(chunks).added == 0
    assert second.search("Lifetime ISA £4,000", k=1)[0].chunk == chunks[1]
    second.close()


def test_each_embedder_gets_its_own_collection(chunks):
    """Vectors from different models must never be compared, so they never share a collection."""
    small, large = HashEmbedder(dimension=32), HashEmbedder(dimension=64)
    assert collection_name(small) != collection_name(large)
    client_store = VectorStore(small)
    client_store.sync(chunks)
    sibling = VectorStore(large, client=client_store.client)
    assert sibling.count() == 0 and client_store.count() == 4


def test_collection_names_are_safe_for_any_embedder_name():
    class Odd:
        name = "BAAI/bge-small-en-v1.5-384"
        dimension = 384

    assert collection_name(Odd()) == "fernhill-baai-bge-small-en-v1-5-384"


def test_point_ids_are_stable_uuids():
    assert point_id("isa-guide:abc") == point_id("isa-guide:abc")
    assert point_id("isa-guide:abc") != point_id("isa-guide:abd")
    assert len(point_id("x")) == 36


def test_payload_round_trip_preserves_every_field(chunks):
    assert payload_to_chunk(chunk_to_payload(chunks[0])) == chunks[0]


def test_only_one_location_may_be_given():
    with pytest.raises(ValueError, match="at most one"):
        VectorStore(HashEmbedder(dimension=32), path="x", url="http://localhost:6333")


def test_open_store_uses_the_index_path_from_settings(tmp_path: Path):
    settings = Settings(embedder="hash", index_path=str(tmp_path / "idx"))
    vs = open_store(settings, HashEmbedder(dimension=32))
    try:
        assert vs.count() == 0
        assert (tmp_path / "idx").is_dir()
    finally:
        vs.close()
