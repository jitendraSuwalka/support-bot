"""Full ``run_agent`` runs against a scripted model.

These force branches a real model may avoid -- the policy rewrite, the
citation rewrite, the crashed tool, the answer from memory -- so the safety
net is known to work rather than hoped to. Every assertion reads the
structured AgentResult, never printed text.
"""

import agent
from tests.conftest import text_reply, tool_reply

# --- verbatim status quoting ---------------------------------------------

STATUS_SCRIPT = [
    tool_reply("look_up_client_query", {"reference": "FW-1001"}),
    text_reply('Status for FW-1001: "Received 1 Aug. With your adviser. Response due by 6 Aug."'),
]


def test_verbatim_status_survives_the_pipeline_unchanged(run_scenario):
    results, _, _ = run_scenario(STATUS_SCRIPT, ["Is FW-1001 done?"])
    assert "Received 1 Aug. With your adviser. Response due by 6 Aug." in results[0].answer


def test_evidence_is_attached_to_the_result(run_scenario):
    results, _, _ = run_scenario(STATUS_SCRIPT, ["Is FW-1001 done?"])
    assert results[0].evidence and results[0].evidence[0].startswith("look_up_client_query ->")
    assert results[0].tools_used == ["look_up_client_query"]


def test_a_clean_answer_carries_no_notes_and_a_clean_trace(run_scenario):
    results, _, _ = run_scenario(STATUS_SCRIPT, ["Is FW-1001 done?"])
    assert results[0].notes == []
    assert results[0].escalated is False
    assert "citations_clean" in results[0].trace.kinds()
    assert "verified" in results[0].trace.kinds()


def test_the_result_records_which_models_answered(run_scenario):
    results, _, _ = run_scenario(STATUS_SCRIPT, ["Is FW-1001 done?"])
    assert results[0].models == {"main": agent.MAIN_MODEL, "helper": agent.HELPER_MODEL}


# --- citation violation forces a rewrite --------------------------------

TRUNCATED_URL_SCRIPT = [
    tool_reply("search_knowledge_base", {"query": "offices"}),
    text_reply("We have offices. See fernhillwealth.example/offices for the list."),
    text_reply("We have offices across England and Scotland."),
]


def test_truncated_url_triggers_an_automatic_rewrite(run_scenario):
    results, _, _ = run_scenario(
        TRUNCATED_URL_SCRIPT,
        ["Do you have a Manchester office?"],
        helper={"rerank": '{"relevant": [0]}'},
    )
    assert "fernhillwealth.example/offices for the list" not in results[0].answer


def test_rewrite_is_recorded_in_notes_and_trace(run_scenario):
    results, _, _ = run_scenario(
        TRUNCATED_URL_SCRIPT,
        ["Do you have a Manchester office?"],
        helper={"rerank": '{"relevant": [0]}'},
    )
    assert "rewritten after citation check" in results[0].notes
    violation = results[0].trace.find("citation_violation")
    assert violation and "shorten" in violation[0].data["violations"][0]


# --- the branch a real model kept avoiding ------------------------------
# Turn 1: model asks a question. Turn 2: model asks ANOTHER question instead
# of escalating. Policy must escalate in code and rewrite the reply.

QUIZ_SCRIPT = [
    tool_reply("search_knowledge_base", {"query": "quiz"}),
    text_reply("I'm not sure what you mean by quiz — could you clarify?"),
    tool_reply("search_knowledge_base", {"query": "quiz tool"}, call_id="t2"),
    text_reply("Still unclear — can you tell me more about where you saw it?"),
    text_reply(
        "I don't have information about that. It has been passed to a "
        "qualified adviser who will contact you. You can call 0117 496 0100."
    ),
]
QUIZ_TURNS = ["quiz", "just the quiz thing"]
QUIZ_HELPER = {"rerank": '{"relevant": []}'}


def test_turn_one_clarification_is_allowed_once(run_scenario):
    results, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "clarify" in results[0].answer.lower()
    assert results[0].escalated is False
    assert "policy_clarify" in results[0].trace.kinds()


def test_turn_two_policy_escalates_in_code(run_scenario):
    results, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert results[1].escalated is True
    assert "escalated by policy" in results[1].notes
    assert "policy_escalate" in results[1].trace.kinds()


def test_turn_two_answer_is_rewritten_not_appended(run_scenario):
    results, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "can you tell me more" not in results[1].answer.lower()


def test_turn_two_no_dangling_question_after_escalation(run_scenario):
    results, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "?" not in results[1].answer


def test_turn_two_switchboard_number_is_present(run_scenario):
    results, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert agent.SWITCHBOARD in results[1].answer


# --- model escalates itself: policy must stand down ---------------------


def test_model_led_escalation_does_not_trigger_a_second_one(run_scenario):
    results, _, _ = run_scenario(
        script=[
            tool_reply("escalate_to_adviser", {"reason": "advice", "summary": "pension advice"}),
            text_reply("A qualified adviser will contact you. Call 0117 496 0100."),
        ],
        user_messages=["Should I move my pension into equities?"],
    )
    assert results[0].escalated is True
    assert "escalated by policy" not in results[0].notes
    assert "policy_escalate" not in results[0].trace.kinds()


# --- the evidence guard: no answering from memory -----------------------


def test_greeting_is_answered_without_any_tool(run_scenario):
    results, _, fake = run_scenario(
        script=[text_reply("Hi there — how can I help you today?")],
        user_messages=["hello"],
    )
    assert results[0].tools_used == []
    assert "evidence_guard" not in results[0].trace.kinds()
    assert len(fake.calls) == 1


def test_an_answer_from_memory_is_sent_back_for_a_lookup(run_scenario):
    results, _, fake = run_scenario(
        script=[
            text_reply("Complaints can be made by letter, telephone, email or in person."),
            tool_reply("search_knowledge_base", {"query": "complaints"}),
            text_reply("Complaints can be made by letter, telephone, email or in person."),
        ],
        user_messages=["How do I complain?"],
    )
    assert "evidence_guard" in results[0].trace.kinds()
    assert results[0].tools_used == ["search_knowledge_base"]
    assert results[0].escalated is False
    # The retry instruction was inserted as a user turn before the model tried again.
    assert any(m.get("content") == agent.EVIDENCE_GUARD_RETRY for m in fake.calls[1]["messages"])


def test_two_answers_from_memory_are_escalated_not_delivered(run_scenario):
    results, _, _ = run_scenario(
        script=[
            text_reply("Our Bristol office is on Queen Square."),
            text_reply("Our Bristol office is on Queen Square, near the harbour."),
            text_reply(
                "I do not have that information. It has been passed to a qualified "
                "adviser who will be in touch. You can call 0117 496 0100."
            ),
        ],
        user_messages=["Where is your Bristol office?"],
    )
    assert "evidence_guard_failed" in results[0].trace.kinds()
    assert results[0].escalated is True
    assert "escalated by policy" in results[0].notes
    assert "Queen Square" not in results[0].answer


# --- a broken tool must not kill the conversation -----------------------
# The fallback reply quotes the switchboard number, which is NOT in the
# evidence (the tool crashed), so the citation guard correctly fires and
# demands a rewrite. A hardcoded phone number in an answer is an unsupported
# claim like any other.

CRASH_SCRIPT = [
    tool_reply("look_up_client_query", {"wrong_argument": "x"}),
    text_reply("Sorry, I could not look that up. Please call 0117 496 0100."),
    text_reply("Sorry, I could not look that up. Please contact your adviser."),
]


def test_a_tool_crash_is_caught_and_the_agent_still_replies(run_scenario):
    results, _, _ = run_scenario(CRASH_SCRIPT, ["Check FW-1001"])
    assert results[0].answer
    assert "tool_failed" in results[0].trace.kinds()


def test_an_unsupported_fallback_number_is_stripped_by_the_guard(run_scenario):
    results, _, _ = run_scenario(CRASH_SCRIPT, ["Check FW-1001"])
    assert "0117 496 0100" not in results[0].answer
    assert "rewritten after citation check" in results[0].notes


# --- the terminal rendering keeps the audit trail ----------------------


def test_render_includes_answer_notes_and_evidence(run_scenario):
    results, _, _ = run_scenario(CRASH_SCRIPT, ["Check FW-1001"])
    rendered = results[0].render()
    assert results[0].answer in rendered
    assert "[rewritten after citation check]" in rendered
    assert "audit trail" in rendered
