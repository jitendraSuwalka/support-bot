"""The code-enforced policy: clarify once, then escalate."""

import agent

UNRESOLVED = ["search_knowledge_base -> NO_MATCH: nothing found"]
RESOLVED = ["search_knowledge_base -> [complaints] real content"]


def test_first_unclear_turn_allows_one_clarification():
    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("Could you clarify?", UNRESOLVED, False, state)
    assert forced is False
    assert state["unclear_turns"] == 1


def test_second_unclear_turn_forces_escalation():
    state = {"unclear_turns": 1}
    _, forced = agent.enforce_escalation("Tell me more?", UNRESOLVED, False, state)
    assert forced is True


def test_counter_resets_after_forcing():
    state = {"unclear_turns": 1}
    agent.enforce_escalation("Tell me more?", UNRESOLVED, False, state)
    assert state["unclear_turns"] == 0


def test_no_double_escalation_when_the_model_already_escalated():
    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("An adviser will call.", UNRESOLVED, True, state)
    assert forced is False


def test_successful_turn_resets_the_counter():
    state = {"unclear_turns": 1}
    _, forced = agent.enforce_escalation("Here is your answer.", RESOLVED, False, state)
    assert forced is False
    assert state["unclear_turns"] == 0


def test_unclear_turn_with_no_question_escalates_immediately():
    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("I do not have that.", UNRESOLVED, False, state)
    assert forced is True


def test_no_record_also_counts_as_unresolved():
    state = {"unclear_turns": 0}
    evidence = ["look_up_client_query -> NO_RECORD: not found"]
    agent.enforce_escalation("Not found, check the number?", evidence, False, state)
    assert state["unclear_turns"] == 1


def test_greeting_is_recognised():
    assert "hello" in agent.PLEASANTRIES


def test_a_real_query_is_not_treated_as_a_greeting():
    assert "quiz" not in agent.PLEASANTRIES


# --- the clarify and decline tools ---------------------------------------------------
# When the model asks through ask_for_clarification the policy counts the
# question exactly instead of looking for a "?" in prose; when it declines,
# the turn is resolved.


def test_a_clarification_through_the_tool_is_allowed_once():
    state = {"unclear_turns": 0}
    answer, forced = agent.enforce_escalation("What do you mean?", [], False, state, "clarified")
    assert (answer, forced) == ("What do you mean?", False)
    assert state["unclear_turns"] == 1


def test_a_second_clarification_in_a_row_is_escalated_with_the_fixed_reply():
    state = {"unclear_turns": 1}
    answer, forced = agent.enforce_escalation("And what else?", [], False, state, "clarified")
    assert forced is True
    assert answer == agent.POLICY_ESCALATION_REPLY and agent.SWITCHBOARD in answer
    assert "?" not in answer
    assert state["unclear_turns"] == 0


def test_a_declined_request_resolves_the_turn():
    state = {"unclear_turns": 1}
    answer, forced = agent.enforce_escalation(agent.DECLINE_REPLY, [], False, state, "declined")
    assert (answer, forced) == (agent.DECLINE_REPLY, False)
    assert state["unclear_turns"] == 0


def test_the_tool_signal_outranks_the_prose_heuristic():
    # The tool says the turn ended with a clarification even though the text has no "?".
    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation(
        "Tell me more about it.", UNRESOLVED, False, state, "clarified"
    )
    assert forced is False and state["unclear_turns"] == 1
