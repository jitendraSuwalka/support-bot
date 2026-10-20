"""The structured trace: events are recorded per turn and isolated between turns."""

import agent
from agent import Trace


def test_log_outside_a_run_prints_instead_of_failing(capsys):
    agent.log("demo", "hello")
    assert "[demo] hello" in capsys.readouterr().out


def test_log_inside_a_run_is_captured_on_the_trace():
    trace = Trace()
    token = agent._CURRENT_TRACE.set(trace)
    try:
        agent.log("demo", "captured", detail=42)
    finally:
        agent._CURRENT_TRACE.reset(token)
    assert trace.kinds() == ["demo"]
    assert trace.find("demo")[0].data == {"detail": 42}


def test_render_is_one_line_per_event():
    trace = Trace()
    trace.log("tool", "search_knowledge_base({'query': 'offices'})")
    trace.log("citations_clean", "citations: clean")
    assert trace.render().splitlines() == [
        "[tool] search_knowledge_base({'query': 'offices'})",
        "[citations_clean] citations: clean",
    ]


def test_each_turn_gets_its_own_trace(run_scenario):
    from tests.conftest import text_reply, tool_reply

    results, _, _ = run_scenario(
        script=[
            tool_reply("look_up_client_query", {"reference": "FW-1001"}),
            text_reply('"Received 1 Aug. With your adviser. Response due by 6 Aug."'),
            text_reply("You're welcome."),
        ],
        user_messages=["Is FW-1001 done?", "thanks"],
    )
    assert "tool" in results[0].trace.kinds()
    # The second turn only expired the first turn's volatile lookup; nothing
    # from turn one leaks into its trace.
    assert results[1].trace.kinds() == ["expired"]
