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


# --- greetings with something attached ------------------------------------------


def test_a_greeting_with_a_pleasantry_attached_is_still_a_greeting():
    for text in (
        "Hi, hope you're well.",
        "Hello there!",
        "ok thanks",
        "Great, thank you and goodbye",
        "Thanks a lot - that's helpful",
        "good morning",
    ):
        assert agent.is_pleasantry(text) is True, text


def test_a_greeting_with_a_question_attached_is_not():
    for text in (
        "Hi, what's the ISA allowance?",
        "thanks, and what about pensions",
        "hello, FW-1001?",
        "allowance",
        "quiz",
        "yes",
        "no",
        "",
    ):
        assert agent.is_pleasantry(text) is False, text


# --- who takes an escalation -------------------------------------------------
# A complaint about an adviser was handed back to "a qualified adviser", and a
# customer who had just paid a scammer was not told to call their bank.


def test_a_complaint_goes_to_client_resolution():
    reply = agent.escalate_to_adviser("complaint", "Complaint about the adviser.")
    assert "Client Resolution" in reply and agent.CLIENT_RESOLUTION in reply
    assert "adviser will contact" not in reply


def test_a_scam_payment_goes_to_fraud_and_security_and_the_customers_bank():
    reply = agent.escalate_to_adviser("scam or missing money", "Paid a caller.")
    assert "Fraud and Security" in reply and agent.SWITCHBOARD in reply
    assert "own bank straight away" in reply
    # The documents say to call immediately: this route alone keeps its urgency.
    assert "immediately" in reply and "sooner" not in reply


def test_a_life_event_goes_to_the_life_events_team():
    reply = agent.escalate_to_adviser("life event", "Client diagnosed with dementia.")
    assert agent.LIFE_EVENTS_TEAM in reply


def test_anything_else_goes_to_an_adviser_with_the_switchboard():
    reasons = ("advice", "financial difficulty", "instruction", "not covered")
    for reason in (*reasons, "unresolved query (policy-enforced)"):
        reply = agent.escalate_to_adviser(reason, "x")
        assert "qualified adviser will contact" in reply and agent.SWITCHBOARD in reply


def test_a_reason_in_the_models_own_words_still_finds_its_team():
    assert agent.escalation_route("Complaint about my adviser") == "complaint"
    assert agent.escalation_route("suspected fraud") == "fraud"
    assert agent.escalation_route("bereavement") == "life event"
    assert agent.escalation_route("my husband passed away") == "life event"
    assert agent.escalation_route("pension advice") == "adviser"


def test_a_reason_that_could_go_two_ways_goes_where_the_customer_is_safest():
    assert agent.escalation_route("complaint about a scam") == "fraud"  # the bank must hear
    assert agent.escalation_route("missing document") == "adviser"  # paperwork, not fraud
    assert agent.escalation_route("not covered (missing details)") == "adviser"


def test_no_route_but_fraud_is_called_quicker():
    for reason in agent.ESCALATION_REASONS:
        reply = agent.escalate_to_adviser(reason, "x")
        if agent.escalation_route(reason) == "fraud":
            assert "sooner" not in reply
        else:
            assert "sooner" in reply  # the instruction not to say it


def test_the_tool_offers_every_reason_and_routes_each():
    [escalate] = [tool for tool in agent.tools if tool["name"] == "escalate_to_adviser"]
    offered = escalate["input_schema"]["properties"]["reason"]["enum"]
    assert offered == list(agent.ESCALATION_REASONS)
    routed = {agent.escalation_route(reason) for reason in offered}
    assert routed == set(agent.ESCALATION_ROUTES)
