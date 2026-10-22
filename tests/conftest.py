"""Shared test doubles for the support agent.

The agent's behaviour depends on a non-deterministic model, so most tests
replace the Anthropic client with a scripted fake. Scripting exactly what the
model returns lets every branch be forced on demand -- including the ones a
well-behaved model would never take. No test in this directory calls the real
API unless it is marked ``live``.
"""

from __future__ import annotations

import types

import pytest

import agent
from rag.corpus import Chunk, chunk_corpus, chunk_id_for
from rag.embeddings import HashEmbedder
from rag.lexical import BM25Index
from rag.retrieval import HybridRetriever
from rag.store import VectorStore


class Block:
    """Mimics an SDK content block closely enough for the agent."""

    def __init__(self, type, text=None, id=None, name=None, input=None):  # noqa: A002
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class Response:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    """Returns scripted responses instead of calling the API.

    Main-model calls are served from a queue, in order. Helper-model calls
    (expansion, reranking, fact-checking) are answered from ``helper``, keyed
    by the kind of prompt, so tests stay deterministic and cost nothing.
    """

    def __init__(self, script, helper=None):
        self.script = list(script)
        self.helper = helper or {}
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        model = kwargs.get("model")
        prompt = ""
        if kwargs.get("messages"):
            first = kwargs["messages"][0].get("content")
            if isinstance(first, str):
                prompt = first

        if model == agent.HELPER_MODEL:
            if "expand" in prompt.lower():
                payload = self.helper.get("expand", "[]")
            elif "relevant" in prompt.lower():
                payload = self.helper.get("rerank", '{"relevant": [0]}')
            else:
                payload = self.helper.get("verify", '{"grounded": true, "unsupported": []}')
            return Response([Block("text", text=payload)], "end_turn")

        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("fake model ran out of scripted responses")
        return self.script.pop(0)


def text_reply(text):
    return Response([Block("text", text=text)], "end_turn")


def tool_reply(name, tool_input, call_id="t1", text=None):
    blocks = []
    if text:
        blocks.append(Block("text", text=text))
    blocks.append(Block("tool_use", id=call_id, name=name, input=tool_input))
    return Response(blocks, "tool_use")


@pytest.fixture(scope="session")
def corpus_retriever():
    """A hybrid retriever over the real corpus with the hash embedder, in memory.

    Built once per test session: no model download, no index on disk, and the
    same vectors on every machine. Its dense scores are weak (hashing is not
    meaning), so retrieval in tests is decided by the lexical side.
    """
    chunks = chunk_corpus()
    store = VectorStore(HashEmbedder())
    store.sync(chunks)
    return HybridRetriever(BM25Index(chunks), store)


@pytest.fixture(autouse=True)
def offline_retriever(request, monkeypatch, corpus_retriever):
    """Give the agent the in-memory retriever, except in live tests, which
    must exercise the real model and the real index."""
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(agent, "_RETRIEVER", corpus_retriever)


@pytest.fixture
def run_scenario(monkeypatch):
    """Run user turns through the real ``run_agent`` against a scripted model.

    Returns one AgentResult per user turn, the policy state, and the fake client.
    """

    def _run(script, user_messages, helper=None, state=None):
        fake = FakeClient(script, helper)
        monkeypatch.setattr(agent, "client", fake)
        messages = []
        state = state if state is not None else {"unclear_turns": 0}
        results = []
        for text in user_messages:
            messages.append({"role": "user", "content": text})
            results.append(agent.run_agent(messages, state))
        return results, state, fake

    return _run


def make_chunk(doc_id: str, section: str, text: str, position: int = 0) -> Chunk:
    """A corpus chunk with realistic provenance, for retrieval tests that build their own corpus."""
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
