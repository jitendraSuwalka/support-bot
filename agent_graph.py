"""
The same support agent as a LangGraph state graph.

Every model call, tool and guard is the code in agent.py; this module only
wires the steps differently. The hand-written loop in agent.py keeps its
control flow in `for`, `if` and `continue`; here each step is a node and
each decision is a conditional edge, so the structure of a turn is data a
tool can draw (`python agent_graph.py --mermaid`) rather than prose a
reader has to reconstruct. The scenario tests run every case through both,
and the two must agree.

What the graph adds: an explicit, inspectable state; a diagram that cannot
drift from the code because it is generated from it; and a place to attach
LangGraph's checkpointing or streaming later without touching the agent's
logic. What it does not change: the model is still called through the
Anthropic SDK directly, because the guards need the raw response blocks
and the request needs the effort setting, which a chat-model wrapper would
hide or lag behind.

The turn:

    prepare -> call_model -> [tool_use?] -> run_tools -> call_model ...
                          -> [no tools]  -> evidence_guard -> (retry) call_model
                                                           -> citation_guard
                                                           -> groundedness_check
                                                           -> escalation_policy
                                                           -> finish

A turn that keeps calling tools past max_steps ends at give_up, exactly as
the loop does.
"""

from __future__ import annotations

import sys
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

import agent
from agent import AgentResult, Trace


class TurnState(TypedDict, total=False):
    messages: list  # the conversation, extended in place
    policy: dict  # cross-turn policy memory (unclear_turns)
    trace: Trace
    is_greeting: bool
    evidence_parts: list[str]
    tools_used: list[str]
    model_escalated: bool
    guard_retries: int
    guard_verdict: str  # "pass", "retry" or "failed"
    steps: int
    response: Any  # the latest model response
    answer: str
    notes: list[str]
    forced: bool
    result: AgentResult


# =========================================================
# NODES — each one is a step from agent.py
# =========================================================


def prepare(state: TurnState) -> TurnState:
    return {
        "is_greeting": agent.begin_turn(state["messages"]),
        "evidence_parts": [],
        "tools_used": [],
        "model_escalated": False,
        "guard_retries": 0,
        "steps": 0,
        "notes": [],
        "forced": False,
    }


def call_model(state: TurnState) -> TurnState:
    response = agent.call_model(state["messages"])
    return {"response": response, "steps": state["steps"] + 1}


def run_tools(state: TurnState) -> TurnState:
    escalated = agent.run_tool_calls(
        state["response"], state["messages"], state["evidence_parts"], state["tools_used"]
    )
    return {"model_escalated": state["model_escalated"] or escalated}


def evidence_guard(state: TurnState) -> TurnState:
    answer = agent.text_of(state["response"])
    verdict = agent.evidence_guard(
        state["messages"], state["evidence_parts"], state["is_greeting"], state["guard_retries"]
    )
    update: TurnState = {"answer": answer, "guard_verdict": verdict}
    if verdict == "retry":
        update["guard_retries"] = 1
    return update


def citation_guard(state: TurnState) -> TurnState:
    answer, notes = agent.citation_guard(
        state["answer"], state["messages"], "\n\n".join(state["evidence_parts"])
    )
    return {"answer": answer, "notes": state["notes"] + notes}


def groundedness_check(state: TurnState) -> TurnState:
    answer, notes = agent.groundedness_check(
        state["answer"], state["messages"], "\n\n".join(state["evidence_parts"])
    )
    return {"answer": answer, "notes": state["notes"] + notes}


def escalation_policy(state: TurnState) -> TurnState:
    answer, notes, forced = agent.escalation_policy(
        state["answer"],
        state["messages"],
        state["evidence_parts"],
        state["model_escalated"],
        state["policy"],
    )
    return {"answer": answer, "notes": state["notes"] + notes, "forced": forced}


def finish(state: TurnState) -> TurnState:
    return {
        "result": AgentResult(
            answer=state["answer"],
            notes=state["notes"],
            evidence=list(state["evidence_parts"]),
            trace=state["trace"],
            escalated=state["model_escalated"] or state["forced"],
            tools_used=state["tools_used"],
            models=agent.models_in_use(),
        )
    }


def give_up(state: TurnState) -> TurnState:
    return {
        "result": agent.step_limit_result(
            state["evidence_parts"], state["tools_used"], state["trace"]
        )
    }


# =========================================================
# EDGES — the decisions
# =========================================================


def after_model(state: TurnState) -> Literal["run_tools", "evidence_guard"]:
    return "run_tools" if state["response"].stop_reason == "tool_use" else "evidence_guard"


def after_tools(state: TurnState) -> Literal["call_model", "give_up"]:
    return "call_model" if state["steps"] < agent.MAX_STEPS else "give_up"


def after_evidence_guard(state: TurnState) -> Literal["call_model", "citation_guard", "give_up"]:
    if state["guard_verdict"] != "retry":
        return "citation_guard"
    # The retry is one more model call, and it counts against max_steps as it
    # does in the loop.
    return "call_model" if state["steps"] < agent.MAX_STEPS else "give_up"


def build_graph():
    graph = StateGraph(TurnState)
    for name, node in (
        ("prepare", prepare),
        ("call_model", call_model),
        ("run_tools", run_tools),
        ("evidence_guard", evidence_guard),
        ("citation_guard", citation_guard),
        ("groundedness_check", groundedness_check),
        ("escalation_policy", escalation_policy),
        ("finish", finish),
        ("give_up", give_up),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "call_model")
    graph.add_conditional_edges("call_model", after_model)
    graph.add_conditional_edges("run_tools", after_tools)
    graph.add_conditional_edges("evidence_guard", after_evidence_guard)
    graph.add_edge("citation_guard", "groundedness_check")
    graph.add_edge("groundedness_check", "escalation_policy")
    graph.add_edge("escalation_policy", "finish")
    graph.add_edge("finish", END)
    graph.add_edge("give_up", END)
    return graph.compile()


GRAPH = build_graph()


def run_graph_agent(messages, state) -> AgentResult:
    """One customer turn through the graph; same contract as agent.run_loop_agent."""
    trace = Trace()
    token = agent._CURRENT_TRACE.set(trace)
    try:
        final = GRAPH.invoke(
            {"messages": messages, "policy": state, "trace": trace},
            # Each model call is up to three nodes; the limit must never be
            # what ends a turn, max_steps is.
            config={"recursion_limit": 4 * agent.MAX_STEPS + 10},
        )
        return final["result"]
    finally:
        agent._CURRENT_TRACE.reset(token)


def mermaid() -> str:
    """The graph as Mermaid source, for the README. Generated, so it cannot drift."""
    return GRAPH.get_graph().draw_mermaid()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(mermaid() if "--mermaid" in argv else __doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
