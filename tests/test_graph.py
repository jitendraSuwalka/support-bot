"""The LangGraph implementation: its shape, its equivalence to the loop, and
the paths only the graph's own wiring could get wrong."""

import pytest

import agent
import agent_graph
from tests.conftest import FakeClient, text_reply, tool_reply
from tests.test_agent_scenarios import (
    CRASH_SCRIPT,
    QUIZ_HELPER,
    QUIZ_SCRIPT,
    QUIZ_TURNS,
    STATUS_SCRIPT,
    TRUNCATED_URL_SCRIPT,
)

# --- shape -----------------------------------------------------------------------


def test_every_step_of_a_turn_is_a_node():
    nodes = set(agent_graph.GRAPH.get_graph().nodes)
    assert {
        "prepare",
        "call_model",
        "run_tools",
        "evidence_guard",
        "citation_guard",
        "groundedness_check",
        "escalation_policy",
        "finish",
        "give_up",
    } <= nodes


def test_the_guards_run_in_order_and_policy_runs_last():
    edges = {(e.source, e.target) for e in agent_graph.GRAPH.get_graph().edges}
    assert ("citation_guard", "groundedness_check") in edges
    assert ("groundedness_check", "escalation_policy") in edges
    assert ("escalation_policy", "finish") in edges
    assert ("call_model", "run_tools") in edges and ("run_tools", "call_model") in edges


def test_the_mermaid_diagram_is_generated_from_the_graph():
    source = agent_graph.mermaid()
    assert source.startswith("---") and "graph TD" in source
    for name in ("prepare", "evidence_guard", "escalation_policy", "give_up"):
        assert f"{name}({name})" in source


def test_the_cli_prints_the_diagram_or_the_explanation(capsys):
    assert agent_graph.main(["--mermaid"]) == 0
    assert "graph TD" in capsys.readouterr().out
    assert agent_graph.main([]) == 0
    assert "LangGraph state graph" in capsys.readouterr().out


# --- equivalence with the loop -------------------------------------------------------


def run_both(script, turns, helper=None, monkeypatch=None):
    outcomes = {}
    for name, run_turn in (("loop", agent.run_loop_agent), ("graph", agent_graph.run_graph_agent)):
        fake = FakeClient(list(script), helper)
        monkeypatch.setattr(agent, "client", fake)
        messages, state, results = [], {"unclear_turns": 0}, []
        for text in turns:
            messages.append({"role": "user", "content": text})
            results.append(run_turn(messages, state))
        outcomes[name] = (results, state, messages, fake.calls)
    return outcomes["loop"], outcomes["graph"]


@pytest.mark.parametrize(
    ("script", "turns", "helper"),
    [
        pytest.param(STATUS_SCRIPT, ["Is FW-1001 done?"], None, id="status"),
        pytest.param(
            TRUNCATED_URL_SCRIPT,
            ["Do you have a Manchester office?"],
            {"rerank": '{"relevant": [0]}'},
            id="citation-rewrite",
        ),
        pytest.param(QUIZ_SCRIPT, QUIZ_TURNS, QUIZ_HELPER, id="policy-escalation"),
        pytest.param(CRASH_SCRIPT, ["Check FW-1001"], None, id="tool-crash"),
        pytest.param(
            [
                text_reply("Complaints can be made by letter."),
                tool_reply("search_knowledge_base", {"query": "complaints"}),
                text_reply("Complaints can be made by letter."),
            ],
            ["How do I complain?"],
            None,
            id="evidence-guard-retry",
        ),
        pytest.param(
            [
                text_reply("Our Bristol office is on Queen Square."),
                text_reply("Our Bristol office is on Queen Square, near the harbour."),
                text_reply("Passed to an adviser. Call 0117 496 0100."),
            ],
            ["Where is your Bristol office?"],
            None,
            id="evidence-guard-failed",
        ),
        pytest.param([text_reply("Hi there!")], ["hello"], None, id="greeting"),
    ],
)
def test_loop_and_graph_produce_the_same_result(script, turns, helper, monkeypatch):
    (
        (loop_results, loop_state, loop_messages, loop_calls),
        (
            graph_results,
            graph_state,
            graph_messages,
            graph_calls,
        ),
    ) = run_both(script, turns, helper, monkeypatch)
    for loop, graph in zip(loop_results, graph_results, strict=True):
        assert graph.answer == loop.answer
        assert graph.notes == loop.notes
        assert graph.evidence == loop.evidence
        assert graph.escalated == loop.escalated
        assert graph.tools_used == loop.tools_used
        assert graph.trace.kinds() == loop.trace.kinds()
    assert graph_state == loop_state
    assert len(graph_messages) == len(loop_messages)  # the conversation grew identically
    assert len(graph_calls) == len(loop_calls)  # and the model was called the same number of times


# --- paths only the wiring could break --------------------------------------------------


def test_the_graph_extends_the_callers_conversation_in_place(monkeypatch):
    monkeypatch.setattr(agent, "client", FakeClient(list(STATUS_SCRIPT)))
    messages = [{"role": "user", "content": "Is FW-1001 done?"}]
    agent_graph.run_graph_agent(messages, {"unclear_turns": 0})
    assert len(messages) == 4  # user, assistant tool call, tool result, assistant answer
    assert messages[0]["content"] == "Is FW-1001 done?"


def test_a_turn_that_never_stops_calling_tools_ends_at_give_up(monkeypatch):
    endless = [
        tool_reply("look_up_client_query", {"reference": "FW-1001"}, call_id=f"t{i}")
        for i in range(agent.MAX_STEPS + 1)
    ]
    monkeypatch.setattr(agent, "client", FakeClient(endless))
    result = agent_graph.run_graph_agent(
        [{"role": "user", "content": "Is FW-1001 done?"}], {"unclear_turns": 0}
    )
    assert "step limit reached" in result.notes[0]
    assert "max_steps" in result.trace.kinds()
    assert len(result.tools_used) == agent.MAX_STEPS


def test_run_agent_dispatches_on_the_setting(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "run_loop_agent", lambda m, s: calls.append("loop"))
    monkeypatch.setattr(agent_graph, "run_graph_agent", lambda m, s: calls.append("graph"))
    monkeypatch.setattr(agent, "SETTINGS", agent.Settings(agent_impl="graph"))
    agent.run_agent([], {})
    monkeypatch.setattr(agent, "SETTINGS", agent.Settings(agent_impl="loop"))
    agent.run_agent([], {})
    assert calls == ["graph", "loop"]


def test_the_setting_rejects_unknown_implementations():
    with pytest.raises(ValueError, match="agent_impl"):
        agent.Settings.from_env({"SUPPORT_BOT_AGENT_IMPL": "actor"})
