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
