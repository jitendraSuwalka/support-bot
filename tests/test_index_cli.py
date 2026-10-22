"""`python -m rag.index` builds the index from the real corpus and searches it,
driven entirely by settings, with the hash embedder so no model is needed."""

import json

import pytest

from rag import index


@pytest.fixture
def offline_index(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_BOT_EMBEDDER", "hash")
    monkeypatch.setenv("SUPPORT_BOT_INDEX_PATH", str(tmp_path / "index"))
    monkeypatch.delenv("SUPPORT_BOT_QDRANT_URL", raising=False)
    return tmp_path / "index"


def test_sync_builds_the_whole_corpus_then_finds_nothing_to_do(offline_index, capsys):
    assert index.main([]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["documents"] >= 18 and first["chunks"] == first["indexed"] == first["added"]
    assert first["embedder"] == "hash-256" and first["collection"] == "fernhill-hash-256"
    assert first["removed"] == 0 and first["unchanged"] == 0

    assert index.main([]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["added"] == 0 and second["unchanged"] == first["chunks"]
    assert offline_index.is_dir()


def test_query_prints_ranked_hits_with_their_sources(offline_index, capsys):
    index.main([])
    capsys.readouterr()
    assert index.main(["--query", "Peterloo Buildings Mosley Street Manchester", "-k", "3"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("1. rrf ") and "Contact us and our offices" in lines[0]
    assert "lexical #1 (" in lines[0] and "dense #" in lines[0]
    assert "https://www.fernhillwealth.example/contact-us/" in lines[1]
    assert sum(line[:2] in ("1.", "2.", "3.") for line in lines) == 3


def test_each_mode_searches_with_that_method_alone(offline_index, capsys):
    index.main([])
    capsys.readouterr()
    assert index.main(["-q", "0161 496 0170", "-k", "2", "--mode", "lexical"]) == 0
    lexical = capsys.readouterr().out
    assert "dense —" in lexical and "lexical #1 (" in lexical
    assert "Our offices › Manchester" in lexical.splitlines()[0]

    assert index.main(["-q", "0161 496 0170", "-k", "2", "--mode", "dense"]) == 0
    dense = capsys.readouterr().out
    assert "lexical —" in dense and "dense #1 (" in dense

    with pytest.raises(ValueError, match="mode must be one of"):
        index.search_index(index.Settings.from_env(), "x", 1, mode="magic")


def test_querying_an_empty_index_says_so(offline_index, capsys):
    assert index.main(["-q", "anything"]) == 1
    assert "Run `python -m rag.index` first" in capsys.readouterr().out
