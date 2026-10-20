"""Full ``run_agent`` runs against a scripted model.

These force branches a real model may avoid -- the policy rewrite, the
citation rewrite, the crashed tool -- so the safety net is known to work
rather than hoped to.
"""

from tests.conftest import text_reply, tool_reply


def customer_facing(answer):
    """Strip the notes and audit trail so assertions target what the customer sees."""
    return answer.split("--- audit")[0].split("\n\n  [")[0]


# --- verbatim status quoting ---------------------------------------------


def test_verbatim_status_survives_the_pipeline_unchanged(run_scenario):
    answers, _, _ = run_scenario(
        script=[
            tool_reply("look_up_client_query", {"reference": "FW-1001"}),
            text_reply(
                'Status for FW-1001: "Received 1 Aug. With your adviser. Response due by 6 Aug."'
            ),
        ],
        user_messages=["Is FW-1001 done?"],
    )
    assert "Received 1 Aug. With your adviser. Response due by 6 Aug." in answers[0]


def test_audit_trail_is_attached(run_scenario):
    answers, _, _ = run_scenario(
        script=[
            tool_reply("look_up_client_query", {"reference": "FW-1001"}),
            text_reply('Status for FW-1001: "Received 1 Aug."'),
        ],
        user_messages=["Is FW-1001 done?"],
    )
    assert "audit trail" in answers[0]


# --- citation violation forces a rewrite --------------------------------


def test_truncated_url_triggers_an_automatic_rewrite(run_scenario):
    answers, _, _ = run_scenario(
        script=[
            tool_reply("search_knowledge_base", {"query": "offices"}),
            text_reply("We have offices. See fernhillwealth.example/offices for the list."),
            text_reply("We have offices across the UK, Ireland and the Channel Islands."),
        ],
        user_messages=["Do you have a Manchester office?"],
        helper={"rerank": '{"relevant": [0]}'},
    )
    assert "fernhillwealth.example/offices for the list" not in answers[0]


def test_rewrite_is_flagged_to_the_reader(run_scenario):
    answers, _, _ = run_scenario(
        script=[
            tool_reply("search_knowledge_base", {"query": "offices"}),
            text_reply("We have offices. See fernhillwealth.example/offices for the list."),
            text_reply("We have offices across the UK, Ireland and the Channel Islands."),
        ],
        user_messages=["Do you have a Manchester office?"],
        helper={"rerank": '{"relevant": [0]}'},
    )
    assert "citation check" in answers[0]


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
    answers, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "clarify" in answers[0].lower()


def test_turn_two_policy_escalates_in_code(run_scenario):
    answers, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "escalated by policy" in answers[1]


def test_turn_two_answer_is_rewritten_not_appended(run_scenario):
    answers, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "can you tell me more" not in answers[1].lower()


def test_turn_two_no_dangling_question_after_escalation(run_scenario):
    answers, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "?" not in customer_facing(answers[1])


def test_turn_two_switchboard_number_is_present(run_scenario):
    answers, _, _ = run_scenario(QUIZ_SCRIPT, QUIZ_TURNS, helper=QUIZ_HELPER)
    assert "0117 496 0100" in answers[1]


# --- model escalates itself: policy must stand down ---------------------


def test_model_led_escalation_does_not_trigger_a_second_one(run_scenario):
    answers, _, _ = run_scenario(
        script=[
            tool_reply("escalate_to_adviser", {"reason": "advice", "summary": "pension advice"}),
            text_reply("A qualified adviser will contact you. Call 0117 496 0100."),
        ],
        user_messages=["Should I move my pension into equities?"],
    )
    assert "escalated by policy" not in answers[0]


# --- forced tool use, with the greeting exception -----------------------


def test_greeting_does_not_force_a_tool_call(run_scenario):
    _, _, fake = run_scenario(
        script=[text_reply("Hi there — how can I help you today?")],
        user_messages=["hello"],
    )
    assert fake.calls[0].get("tool_choice", {}).get("type") == "auto"


def test_a_real_question_does_force_a_tool_call(run_scenario):
    _, _, fake = run_scenario(
        script=[
            tool_reply("search_knowledge_base", {"query": "complaints"}),
            text_reply("Complaints can be made by letter, telephone, email or in person."),
        ],
        user_messages=["How do I complain?"],
    )
    assert fake.calls[0].get("tool_choice", {}).get("type") == "any"


# --- a broken tool must not kill the conversation -----------------------
# The fallback reply quotes 0117 496 0100, which is NOT in the evidence (the
# tool crashed), so the citation guard correctly fires and demands a rewrite.
# A hardcoded phone number in an answer is an unsupported claim like any other.

CRASH_SCRIPT = [
    tool_reply("look_up_client_query", {"wrong_argument": "x"}),
    text_reply("Sorry, I could not look that up. Please call 0117 496 0100."),
    text_reply("Sorry, I could not look that up. Please contact your adviser."),
]


def test_a_tool_crash_is_caught_and_the_agent_still_replies(run_scenario):
    answers, _, _ = run_scenario(CRASH_SCRIPT, ["Check FW-1001"])
    assert len(answers[0]) > 0


def test_an_unsupported_fallback_number_is_stripped_by_the_guard(run_scenario):
    answers, _, _ = run_scenario(CRASH_SCRIPT, ["Check FW-1001"])
    assert "0117 496 0100" not in answers[0].split("--- audit")[0]
