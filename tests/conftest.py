"""Shared test doubles for the support agent.

The agent's behaviour depends on a non-deterministic model, so most tests
replace the Anthropic client with a scripted fake. Scripting exactly what the
model returns lets every branch be forced on demand -- including the ones a
well-behaved model would never take. No test in this directory calls the real
API unless it is marked ``live``.
"""

from __future__ import annotations

import socket
import types

import pytest

import agent
import agent_graph
import observability
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
        self.helper_calls = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        model = kwargs.get("model")
        prompt = ""
        if kwargs.get("messages"):
            first = kwargs["messages"][0].get("content")
            if isinstance(first, str):
                prompt = first

        if model == agent.HELPER_MODEL:
            self.helper_calls.append(prompt)
            # The fact-check prompt carries the customer's words, which may say
            # "expand" or "relevant": it is told apart by how it begins.
            if prompt.startswith("You are a strict fact-checker"):
                payload = self.helper.get("verify", '{"grounded": true, "unsupported": []}')
            elif "expand" in prompt.lower():
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


class NetworkUsed(RuntimeError):
    """A test tried to reach another machine."""


_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """The suite says it needs no network; this makes that a checked fact.
    A socket connected to anything but this machine fails the test that
    connected it. (Name lookups and unconnected datagrams are not watched.)
    Live and model tests are exempt: they call the API and download the
    embedding model by design."""
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("model"):
        return
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def local(sock, address) -> bool:
        if sock.family == getattr(socket, "AF_UNIX", None):
            return True
        return isinstance(address, tuple) and address[0] in _LOCAL_HOSTS

    def connect(sock, address):
        if not local(sock, address):
            raise NetworkUsed(f"a test tried to connect to {address!r}")
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if not local(sock, address):
            raise NetworkUsed(f"a test tried to connect to {address!r}")
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True)
def project_logging_left_as_found():
    """A test that configures the project's logging cannot leave it configured
    for the next one: agent events outside a turn go to the log when it is
    configured and are printed when it is not, and tests rely on both."""
    yield
    observability.reset_logging()


IMPLEMENTATIONS = {"loop": agent.run_loop_agent, "graph": agent_graph.run_graph_agent}


@pytest.fixture(params=sorted(IMPLEMENTATIONS))
def run_scenario(request, monkeypatch):
    """Run user turns through the real agent against a scripted model.

    Parametrised over both implementations, the hand-written loop and the
    LangGraph graph, so every scenario is proved on each. Returns one
    AgentResult per user turn, the policy state, and the fake client.
    """
    run_turn = IMPLEMENTATIONS[request.param]

    def _run(script, user_messages, helper=None, state=None):
        fake = FakeClient(script, helper)
        monkeypatch.setattr(agent, "client", fake)
        messages = []
        state = state if state is not None else {"unclear_turns": 0}
        results = []
        for text in user_messages:
            messages.append({"role": "user", "content": text})
            results.append(run_turn(messages, state))
        return results, state, fake

    _run.implementation = request.param
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
