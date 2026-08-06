"""
evaluate.py — the test suite for the support agent.

Run:  python evaluate.py           fast, free, no API calls
      python evaluate.py --live    also runs real questions through the API

WHY THIS EXISTS
    Testing by hand stopped working the moment behaviour became
    non-deterministic. Asking "quiz" twice produced two different paths, so
    the policy-rewrite branch went untested purely by luck.

    The fix is a FAKE MODEL. By scripting exactly what the model returns, every
    branch can be forced on demand -- including the ones a well-behaved model
    would never take. That is the difference between hoping your safety net
    works and knowing it does.

LAYERS
    1. Unit tests      pure functions, deterministic, instant
    2. Agent scenarios full run_agent() against a scripted fake model
    3. Live tests      real API, opt-in, costs money
"""

import sys
import types

import step10 as agent


# =========================================================
# TINY TEST FRAMEWORK
# =========================================================

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    line = f"  [{mark}] {name}"
    if detail and not condition:
        line += f"\n         -> {detail}"
    print(line)


def section(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


# =========================================================
# FAKE MODEL
# =========================================================

class Block:
    """Mimics an SDK content block closely enough for the agent."""

    def __init__(self, type, text=None, id=None, name=None, input=None):
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
    (expansion, reranking, fact-checking) are answered from `helper`, keyed by
    the kind of prompt, so tests stay deterministic and cost nothing.
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


def run_scenario(script, user_messages, helper=None, state=None):
    """Run turns through the real run_agent() with a scripted model."""
    real_client = agent.client
    fake = FakeClient(script, helper)
    agent.client = fake
    try:
        messages, state = [], state if state is not None else {"unclear_turns": 0}
        answers = []
        for text in user_messages:
            messages.append({"role": "user", "content": text})
            answers.append(agent.run_agent(messages, state))
        return answers, state, fake
    finally:
        agent.client = real_client


# =========================================================
# LAYER 1 — UNIT TESTS
# =========================================================

def test_retrieval():
    section("1. RETRIEVAL")

    def top(query, extra=""):
        tokens = agent.tokenize(query) | agent.tokenize(extra)
        scored = [r for r in agent.score_documents(tokens) if r[0] >= agent.RELEVANCE_FLOOR]
        return [entry["topic"] for _, _, entry in scored[:2]]

    check("relevant query ranks the right entry first",
          "office locations bristol local offices" in (top("where is your bristol office")[:1] or [""]),
          f"got {top('where is your bristol office')}")

    check("complaint query beats the long financial-planning entry",
          top("how long does a complaint take")[:1] == ["how to make a complaint"],
          f"got {top('how long does a complaint take')}")

    check("off-topic query returns nothing", top("what is the weather in Paris") == [],
          f"got {top('what is the weather in Paris')}")

    check("expansion terms rescue a reworded query",
          any("complaint" in t for t in top("I am unhappy", "complaint resolution ombudsman")),
          f"got {top('I am unhappy', 'complaint resolution ombudsman')}")

    check("stopwords drop generic query words", "long" not in agent.tokenize("how long does it take"))
    check("plurals fold together", agent.tokenize("offices") == agent.tokenize("office"))

# A healthy config leaves room between real matches and the floor.
    # Testing found the complaint query scored 0.91 against a floor of 0.90 —
    # passing by 0.01, which is luck, not correctness.
    scored = agent.score_documents(agent.tokenize("how do I make a complaint"))
    best = scored[0][0] if scored else 0
    check("good matches clear the floor with margin, not by luck",
          best >= agent.RELEVANCE_FLOOR + 0.3,
          f"best {best:.2f} vs floor {agent.RELEVANCE_FLOOR}")

def test_citations():
    section("2. CITATION GUARD (deterministic)")

    evidence = ("There are 21 offices across the UK. Call 0800 023 4567. "
                "Source: https://www.evelyn.com/offices/bristol/")

    check("truncated URL is caught (the real bug from testing)",
          any("shorten" in v for v in agent.check_citations("See evelyn.com/offices", evidence)),
          str(agent.check_citations("See evelyn.com/offices", evidence)))

    check("invented URL is caught",
          agent.check_citations("Visit example.com/help", evidence) != [])

    check("invented phone number is caught",
          agent.check_citations("Call 0800 111 2222", evidence) != [])

    check("exact URL passes",
          agent.check_citations("See https://www.evelyn.com/offices/bristol/", evidence) == [])

    check("correct numbers pass",
          agent.check_citations("There are 21 offices. Call 0800 023 4567.", evidence) == [],
          str(agent.check_citations("There are 21 offices. Call 0800 023 4567.", evidence)))

    check("small numbers do not false-alarm",
          agent.check_citations("There are 3 options.", evidence) == [])


def test_json():
    section("3. JSON EXTRACTION")
    cases = [
        ("clean object", '{"a": 1}', True),
        ("trailing prose (the crash case)", '{"a": 1}\nAll good.', True),
        ("code fence", '```json\n{"a": 1}\n```', True),
        ("prose both sides", 'Verdict: {"a": 1} hope that helps', True),
        ("braces inside a string", '{"a": "has {braces}"}', True),
        ("array", '["x","y"]', True),
        ("no JSON present", 'I cannot do that.', False),
    ]
    for label, raw, should_parse in cases:
        got = agent.extract_json(raw)
        check(f"extract_json: {label}", (got is not None) == should_parse, repr(got))


def test_expiry():
    section("4. HISTORY EXPIRY")
    messages = [
        {"role": "assistant", "content": [Block("tool_use", id="a", name="look_up_client_query"),
                                          Block("tool_use", id="b", name="search_knowledge_base")]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "Closed 28 July."},
            {"type": "tool_result", "tool_use_id": "b", "content": "Complaints info"},
        ]},
    ]
    wiped = agent.expire_stale_results(messages)
    blocks = {b["tool_use_id"]: b["content"] for b in messages[1]["content"]}
    check("volatile result is wiped", blocks["a"] == agent.EXPIRED_NOTICE)
    check("static result survives", blocks["b"] == "Complaints info")
    check("exactly one wipe reported", wiped == 1, f"got {wiped}")
    check("re-running wipes nothing (idempotent)", agent.expire_stale_results(messages) == 0)


def test_policy_unit():
    section("5. ESCALATION POLICY (state machine)")
    unresolved = ["search_knowledge_base -> NO_MATCH: nothing found"]
    resolved = ["search_knowledge_base -> [complaints] real content"]

    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("Could you clarify?", unresolved, False, state)
    check("first unclear turn allows one clarification", forced is False and state["unclear_turns"] == 1)

    _, forced = agent.enforce_escalation("Tell me more?", unresolved, False, state)
    check("second unclear turn forces escalation", forced is True)
    check("counter resets after forcing", state["unclear_turns"] == 0)

    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("An adviser will call.", unresolved, True, state)
    check("no double-escalation when the model already escalated", forced is False)

    state = {"unclear_turns": 1}
    _, forced = agent.enforce_escalation("Here is your answer.", resolved, False, state)
    check("successful turn resets the counter", state["unclear_turns"] == 0 and forced is False)

    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("I do not have that.", unresolved, False, state)
    check("unclear turn with no question escalates immediately", forced is True)

    state = {"unclear_turns": 0}
    _, forced = agent.enforce_escalation("Not found, check the number?",
                                         ["look_up_client_query -> NO_RECORD: not found"], False, state)
    check("NO_RECORD also counts as unresolved", state["unclear_turns"] == 1)

    check("greeting is recognised", "hello" in agent.PLEASANTRIES)
    check("a real query is NOT treated as a greeting", "quiz" not in agent.PLEASANTRIES)


def test_rerank_unit():
    section("6. RERANKER FAILURE MODES")
    candidates = [
        (0.4, set(), {"topic": "offices", "content": "Bristol office", "source": "u1"}),
        (0.2, set(), {"topic": "callback", "content": "not branch opening hours", "source": "u2"}),
    ]
    original = agent.call_json
    try:
        agent.call_json = lambda *a, **k: {"relevant": [0]}
        check("keeps only what the model judged relevant",
              [e["topic"] for _, _, e in agent.rerank("q", candidates)] == ["offices"])

        agent.call_json = lambda *a, **k: {"relevant": []}
        check("rejecting everything yields no results", agent.rerank("q", candidates) == [])

        agent.call_json = lambda *a, **k: None
        check("falls back to lexical order when the helper is down",
              len(agent.rerank("q", candidates)) == 2)

        agent.call_json = lambda *a, **k: {"relevant": [99, -1, "x", 1]}
        check("malformed indexes are ignored safely",
              [e["topic"] for _, _, e in agent.rerank("q", candidates)] == ["callback"])

        check("empty candidate list is handled", agent.rerank("q", []) == [])
    finally:
        agent.call_json = original


# =========================================================
# LAYER 2 — AGENT SCENARIOS (scripted model, no API)
# =========================================================

def test_agent_scenarios():
    section("7. AGENT SCENARIOS (scripted model — forces branches a real model may avoid)")

    # --- verbatim status quoting -------------------------------------
    answers, _, _ = run_scenario(
        script=[
            tool_reply("look_up_client_query", {"reference": "EP-1001"}),
            text_reply('Status for EP-1001: "Received 1 Aug. With your adviser. Response due by 6 Aug."'),
        ],
        user_messages=["Is EP-1001 done?"],
    )
    check("verbatim status survives the pipeline unchanged",
          "Received 1 Aug. With your adviser. Response due by 6 Aug." in answers[0])
    check("audit trail is attached", "audit trail" in answers[0])

    # --- citation violation forces a rewrite -------------------------
    answers, _, fake = run_scenario(
        script=[
            tool_reply("search_knowledge_base", {"query": "offices"}),
            text_reply("We have offices. See evelyn.com/offices for the list."),
            text_reply("We have offices across the UK, Ireland and the Channel Islands."),
        ],
        user_messages=["Do you have a Manchester office?"],
        helper={"rerank": '{"relevant": [0]}'},
    )
    check("truncated URL triggers an automatic rewrite",
          "evelyn.com/offices for the list" not in answers[0],
          answers[0][:120])
    check("rewrite is flagged to the reader",
          "citation check" in answers[0], answers[0][-200:])

    # --- THE BRANCH A REAL MODEL KEPT AVOIDING -----------------------
    # Turn 1: model asks a question. Turn 2: model asks ANOTHER question
    # instead of escalating. Policy must escalate in code and rewrite.
    answers, state, _ = run_scenario(
        script=[
            tool_reply("search_knowledge_base", {"query": "quiz"}),
            text_reply("I'm not sure what you mean by quiz — could you clarify?"),
            tool_reply("search_knowledge_base", {"query": "quiz tool"}, call_id="t2"),
            text_reply("Still unclear — can you tell me more about where you saw it?"),
            text_reply("I don't have information about that. It has been passed to a "
                       "qualified adviser who will contact you. You can call 020 7131 4000."),
        ],
        user_messages=["quiz", "just the quiz thing"],
        helper={"rerank": '{"relevant": []}'},
    )
    check("turn 1: clarification is allowed once", "clarify" in answers[0].lower())
    check("turn 2: policy escalates in code", "escalated by policy" in answers[1],
          answers[1][-200:])
    check("turn 2: answer is REWRITTEN, not appended",
          "can you tell me more" not in answers[1].lower(), answers[1][:200])
    check("turn 2: no dangling question after escalation",
          "?" not in answers[1].split("--- audit")[0].split("[escalated")[0],
          answers[1][:200])
    check("turn 2: switchboard number is present", "020 7131 4000" in answers[1])

    # --- model escalates itself: policy must stand down --------------
    answers, _, _ = run_scenario(
        script=[
            tool_reply("escalate_to_adviser", {"reason": "advice", "summary": "pension advice"}),
            text_reply("A qualified adviser will contact you. Call 020 7131 4000."),
        ],
        user_messages=["Should I move my pension into equities?"],
    )
    check("model-led escalation does not trigger a second one",
          "escalated by policy" not in answers[0])

    # --- greeting skips the forced tool call -------------------------
    answers, _, fake = run_scenario(
        script=[text_reply("Hi there — how can I help you today?")],
        user_messages=["hello"],
    )
    check("greeting does not force a tool call",
          fake.calls[0].get("tool_choice", {}).get("type") == "auto",
          str(fake.calls[0].get("tool_choice")))

    answers, _, fake = run_scenario(
        script=[
            tool_reply("search_knowledge_base", {"query": "complaints"}),
            text_reply("Complaints can be made by letter, telephone, email or in person."),
        ],
        user_messages=["How do I complain?"],
    )
    check("a real question DOES force a tool call",
          fake.calls[0].get("tool_choice", {}).get("type") == "any",
          str(fake.calls[0].get("tool_choice")))

    # --- a broken tool must not kill the conversation ----------------
    # Note: the fallback reply quotes 020 7131 4000, which is NOT in the
    # evidence (the tool crashed), so the citation guard correctly fires and
    # demands a rewrite. Found by this very test -- a hardcoded phone number in
    # an answer is an unsupported claim like any other.
    answers, _, _ = run_scenario(
        script=[
            tool_reply("look_up_client_query", {"wrong_argument": "x"}),
            text_reply("Sorry, I could not look that up. Please call 020 7131 4000."),
            text_reply("Sorry, I could not look that up. Please contact your adviser."),
        ],
        user_messages=["Check EP-1001"],
    )
    check("a tool crash is caught and the agent still replies", len(answers[0]) > 0)
    check("an unsupported fallback number is stripped by the guard",
          "020 7131 4000" not in answers[0].split("--- audit")[0],
          answers[0][:150])


# =========================================================
# LAYER 3 — LIVE TESTS (opt-in, costs money)
# =========================================================

LIVE_CASES = [
    {"question": "Where is your Bristol office?",
     "expect_tool": "search_knowledge_base",
     "expect_text": ["Victoria Street"]},
    {"question": "What's the status of EP-1002?",
     "expect_tool": "look_up_client_query",
     "expect_text": ["Documents verified"]},
    {"question": "Should I move my pension into equities?",
     "expect_tool": "escalate_to_adviser",
     "expect_text": ["adviser"]},
    {"question": "Can I visit a branch on Sunday?",
     "expect_tool": None,
     "expect_text": ["adviser"]},
]


def test_live():
    section("8. LIVE TESTS (real API)")
    for case in LIVE_CASES:
        messages, state = [{"role": "user", "content": case["question"]}], {"unclear_turns": 0}
        try:
            answer = agent.run_agent(messages, state)
        except Exception as error:
            check(f"live: {case['question']}", False, str(error))
            continue

        used = [b.name for m in messages if isinstance(m.get("content"), list)
                for b in m["content"] if getattr(b, "type", None) == "tool_use"]

        if case["expect_tool"]:
            check(f"live: '{case['question']}' calls {case['expect_tool']}",
                  case["expect_tool"] in used, f"used {used}")
        for phrase in case["expect_text"]:
            check(f"live: '{case['question']}' mentions '{phrase}'",
                  phrase.lower() in answer.lower(), answer[:150])


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Evaluating the support agent\n")

    test_retrieval()
    test_citations()
    test_json()
    test_expiry()
    test_policy_unit()
    test_rerank_unit()
    test_agent_scenarios()

    if "--live" in sys.argv:
        test_live()
    else:
        print("\n(skipping live API tests — run with --live to include them)")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 62}")
    print(f"RESULT: {passed}/{total} passed")
    if passed < total:
        print("\nFailures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}")
                if detail:
                    print(f"      {detail}")
    print("=" * 62)

    sys.exit(0 if passed == total else 1)
