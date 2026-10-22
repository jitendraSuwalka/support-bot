"""Embedders: the contract (unit-length float32 rows), the hash stand-in, and
the local-model wrapper checked against a fake model so nothing is downloaded."""

import numpy as np
import pytest

from config import Settings
from rag.embeddings import (
    BGE_QUERY_PREFIX,
    Embedder,
    HashEmbedder,
    LocalEmbedder,
    build_embedder,
    unit_rows,
)

# --- the contract ---------------------------------------------------------------


def test_unit_rows_normalises_each_row_and_keeps_float32():
    rows = unit_rows([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dimension=2)
    assert rows.dtype == np.float32 and rows.shape == (3, 2)
    assert np.allclose(np.linalg.norm(rows, axis=1), [1.0, 0.0, 1.0])


def test_unit_rows_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="expected rows of 3"):
        unit_rows([[1.0, 2.0]], dimension=3)


def test_unit_rows_of_nothing_is_an_empty_matrix_of_the_right_width():
    assert unit_rows([], dimension=5).shape == (0, 5)


# --- hash embedder ----------------------------------------------------------------


@pytest.fixture
def hasher():
    return HashEmbedder(dimension=64)


def test_hash_embedder_satisfies_the_protocol(hasher):
    assert isinstance(hasher, Embedder)
    assert hasher.name == "hash-64" and hasher.dimension == 64


def test_hash_vectors_are_unit_length_float32(hasher):
    rows = hasher.embed_documents(["ISA allowance is £20,000", "pension annual allowance"])
    assert rows.shape == (2, 64) and rows.dtype == np.float32
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_hash_vectors_are_deterministic_across_instances_and_batches(hasher):
    alone = hasher.embed_documents(["Lifetime ISA bonus"])[0]
    in_batch = HashEmbedder(dimension=64).embed_documents(["other text", "Lifetime ISA bonus"])[1]
    assert np.array_equal(alone, in_batch)


def test_texts_sharing_words_are_closer_than_texts_that_do_not(hasher):
    isa, isa_again, office = hasher.embed_documents(
        [
            "How much can I pay into an ISA this tax year?",
            "ISA payments this tax year",
            "Nearest tram stop to the Manchester office",
        ]
    )
    assert float(isa @ isa_again) > float(isa @ office)


def test_queries_and_documents_embed_identically_for_the_hash_embedder(hasher):
    text = ["same text"]
    assert np.array_equal(hasher.embed_queries(text), hasher.embed_documents(text))


def test_empty_text_is_a_zero_vector_and_no_texts_is_no_rows(hasher):
    assert not hasher.embed_documents([""]).any()
    assert hasher.embed_documents([]).shape == (0, 64)


def test_hash_dimension_has_a_floor():
    with pytest.raises(ValueError):
        HashEmbedder(dimension=4)


# --- local embedder, with a fake model -------------------------------------------


class FakeModel:
    """Stands in for fastembed's TextEmbedding: records what it was asked to embed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.seen: list[list[str]] = []

    def embed(self, texts):
        self.seen.append(list(texts))
        for text in texts:
            yield np.array([len(text), 1.0, 2.0], dtype=np.float64)


@pytest.fixture
def local(tmp_path):
    made = []

    def factory(**kwargs):
        made.append(FakeModel(**kwargs))
        return made[-1]

    embedder = LocalEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=tmp_path / "models",
        model_factory=factory,
        dimension=3,
    )
    return embedder, made


def test_local_embedder_loads_the_model_lazily(local):
    embedder, made = local
    assert embedder.name == "BAAI/bge-small-en-v1.5-3" and embedder.dimension == 3
    assert made == []  # nothing loaded until the first embedding
    embedder.embed_documents(["x"])
    assert len(made) == 1
    embedder.embed_queries(["y"])
    assert len(made) == 1  # and only once


def test_local_embedder_prefixes_queries_but_not_documents(local):
    embedder, made = local
    embedder.embed_documents(["a passage"])
    embedder.embed_queries(["a question"])
    assert made[0].seen == [["a passage"], [BGE_QUERY_PREFIX + "a question"]]


def test_local_embedder_output_meets_the_contract(local):
    embedder, _ = local
    rows = embedder.embed_documents(["one", "three"])
    assert rows.dtype == np.float32 and rows.shape == (2, 3)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)
    assert embedder.embed_queries([]).shape == (0, 3)


def test_local_embedder_passes_model_name_and_cache_dir_to_the_model(local, tmp_path):
    embedder, made = local
    embedder.embed_documents(["x"])
    assert made[0].kwargs == {
        "model_name": "BAAI/bge-small-en-v1.5",
        "cache_dir": tmp_path / "models",
    }


def test_a_custom_query_prefix_and_no_prefix_are_both_honoured(tmp_path):
    def factory(**kwargs):
        return FakeModel(**kwargs)

    custom = LocalEmbedder("any", query_prefix="Q: ", model_factory=factory, dimension=3)
    custom.embed_queries(["z"])
    assert custom.model.seen == [["Q: z"]]
    plain = LocalEmbedder("any", query_prefix="", model_factory=factory, dimension=3)
    plain.embed_queries(["z"])
    assert plain.model.seen == [["z"]]


def test_an_unknown_model_name_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="unknown embedding model"):
        LocalEmbedder("not-a-real/model")


# --- factory ------------------------------------------------------------------------


def test_build_embedder_follows_the_settings(tmp_path):
    assert isinstance(build_embedder(Settings(embedder="hash")), HashEmbedder)
    local = build_embedder(Settings(model_cache=str(tmp_path)))
    assert isinstance(local, LocalEmbedder)
    assert local.model_name == "BAAI/bge-small-en-v1.5" and local.dimension == 384
    assert local.cache_dir == tmp_path


# --- the real model: opt in with `pytest -m model` ------------------------------------


@pytest.mark.model
def test_the_real_model_puts_meaning_before_wording():
    embedder = LocalEmbedder()
    assert embedder.dimension == 384
    query = embedder.embed_queries(["How much can I pay into an ISA each year?"])[0]
    about_isa, about_offices = embedder.embed_documents(
        [
            "The overall ISA allowance for the 2026/27 tax year is £20,000.",
            "Our Manchester office is at 41 Mosley Street, near St Peter's Square tram stop.",
        ]
    )
    assert np.isclose(np.linalg.norm(query), 1.0)
    assert float(query @ about_isa) > float(query @ about_offices) + 0.1
