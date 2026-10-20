"""
Grounded support agent — retrieval, verification and escalation.

A customer support assistant for Fernhill Wealth, a fictional UK wealth
management firm. It answers only
what it can evidence, quotes volatile data verbatim, and hands to a human when
it cannot help.

Every hard guarantee here lives in Python rather than in the system prompt,
because a prompt is a strong suggestion and nothing more. The three guards,
in the order they run:

  CITATION GUARD — deterministic.
      Every URL and number in an answer must match the evidence exactly.
      Added after the agent derived "fernhillwealth.example/offices" by truncating a
      deeper URL it had been given: plausible, probably correct, not in the
      evidence, and waved through by the LLM checker. Code cannot be
      persuaded that something looks right.

  GROUNDEDNESS CHECK — an independent model call.
      Reads the finished answer back against its evidence and flags
      unsupported claims, then forces a rewrite. Separate from the model that
      wrote the answer, because a model asked to critique its own output
      defends it. Reports three states, including "unavailable" — a check
      that fails silently while reporting success is worse than no check.

  ESCALATION POLICY — deterministic, stateful.
      Clarify once, then escalate. Enforced in code, so an unanswerable
      question always reaches a human regardless of what the model decides.

Retrieval is two-stage: BM25 lexical scoring for recall, then an LLM rerank
for precision. Lexical scoring counts words and cannot see that "not branch
opening hours" is a denial; reranking reads.

Model non-determinism is irreducible. That is why the guarantees are in code
and why every answer carries an audit trail of its raw evidence.

Personal learning project. Fernhill Wealth is invented; see knowledge_base.py.
Run `python evaluate.py` for the test suite.
"""

import json
import math
import re

from anthropic import Anthropic
from dotenv import load_dotenv

from knowledge_base import KNOWLEDGE_BASE

load_dotenv()
client = Anthropic()

MAIN_MODEL = "claude-sonnet-5"
HELPER_MODEL = "claude-haiku-4-5-20251001"

MAX_STEPS = 8
RELEVANCE_FLOOR = 0.15  # deliberately loose: rerank does the real filtering
CANDIDATES = 5  # retrieve widely...
TOP_K = 2  # ...then keep only what survives reranking

VOLATILE_TOOLS = {"look_up_client_query"}
EXPIRED_NOTICE = (
    "[This earlier result has expired and may no longer be accurate. "
    "Call the tool again if you need this information.]"
)

FORCE_TOOL_FIRST = True
USE_RERANK = True
VERIFY_ANSWERS = True
CHECK_CITATIONS = True
SHOW_AUDIT = True


# =========================================================
# PARSING HELPERS
# =========================================================


def extract_json(text):
    """Pull the first complete JSON value out of a model response.

    Models append explanations, wrap output in fences, or emit a second object.
    json.loads() dies on all three. Never trust a model to return bare JSON
    just because you asked it to.
    """
    if not text:
        return None
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence) :]
    text = text.removesuffix("```").strip()

    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def call_json(prompt, max_tokens=600, attempts=2):
    """Call the helper model and parse JSON, with one retry. None on failure."""
    for attempt in range(attempts):
        try:
            response = client.messages.create(
                model=HELPER_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            payload = extract_json(text)
            if payload is None:
                raise ValueError(f"no JSON in {text[:70]!r}")
            return json.loads(payload)
        except Exception as error:
            if attempt == attempts - 1:
                print(f"  [helper call failed: {error}]")
                return None


# =========================================================
# WEAKNESS 1 — DETERMINISTIC CITATION CHECK
# =========================================================

URL_PATTERN = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s,;)\]\"']*)?", re.I)
NUMBER_PATTERN = re.compile(r"\d[\d\s.,:/-]*\d|\d")

# Numbers so generic that requiring evidence produces only false alarms.
NUMBER_ALLOWLIST = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}


def normalise_url(url):
    """Strip scheme, www and trailing slash so URLs compare fairly."""
    url = url.strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.rstrip("/")


def normalise_number(text):
    """Reduce to digits only, so '0800 023 4567' == '08000234567'."""
    return re.sub(r"\D", "", text)


def check_citations(answer, evidence):
    """Require every URL and number in the answer to appear in the evidence.

    This is the only guarantee in the whole system that cannot be talked out
    of. The LLM verifier passed 'fernhillwealth.example/offices' because it looked right
    and was a substring of a longer URL. This function demands an EXACT match
    after normalisation, so truncating a path to guess a parent is caught.

    Returns a list of violations. Empty list means clean.
    """
    violations = []

    evidence_urls = {normalise_url(u) for u in URL_PATTERN.findall(evidence)}
    for raw in URL_PATTERN.findall(answer):
        candidate = normalise_url(raw)
        if not candidate or "." not in candidate:
            continue
        if candidate in evidence_urls:
            continue
        # Was it derived by truncating or extending a real one?
        related = next(
            (u for u in evidence_urls if u.startswith(candidate) or candidate.startswith(u)),
            None,
        )
        if related:
            violations.append(
                f"URL '{raw}' was not given. Evidence has '{related}'. "
                "Do not shorten or extend a URL."
            )
        else:
            violations.append(f"URL '{raw}' does not appear in the evidence at all.")

    evidence_digits = normalise_number(evidence)
    for raw in NUMBER_PATTERN.findall(answer):
        digits = normalise_number(raw)
        if not digits or digits in NUMBER_ALLOWLIST:
            continue
        if digits not in evidence_digits:
            violations.append(f"Number '{raw.strip()}' does not appear in the evidence.")

    return violations


# =========================================================
# RETRIEVAL
# =========================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "and",
    "or",
    "i",
    "im",
    "my",
    "me",
    "we",
    "you",
    "your",
    "how",
    "do",
    "does",
    "did",
    "what",
    "when",
    "where",
    "which",
    "who",
    "can",
    "could",
    "would",
    "should",
    "for",
    "in",
    "on",
    "at",
    "it",
    "this",
    "that",
    "with",
    "about",
    "from",
    "have",
    "has",
    "had",
    "be",
    "been",
    "get",
    "got",
    "any",
    "please",
    "thanks",
    "long",
    "take",
    "takes",
    "make",
    "made",
    "need",
    "want",
    "know",
    "tell",
    "help",
    "like",
    "just",
    "also",
    "there",
    "their",
    "am",
    "if",
    "so",
    "but",
    "not",
    "no",
    "yes",
    "all",
    "out",
    "up",
    "by",
    "as",
}


def tokenize(text):
    words = re.findall(r"[a-z0-9@.]+", text.lower())
    tokens = set()
    for word in words:
        if word in STOPWORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.add(word)
    return tokens


def build_index():
    documents = []
    for entry in KNOWLEDGE_BASE:
        documents.append(
            {
                "entry": entry,
                "tokens": tokenize(entry["topic"] + " " + entry["content"]),
            }
        )
    total = len(documents)
    frequency = {}
    for document in documents:
        for token in document["tokens"]:
            frequency[token] = frequency.get(token, 0) + 1
    idf = {t: math.log((total + 1) / (c + 1)) + 1.0 for t, c in frequency.items()}
    return documents, idf


DOCUMENTS, IDF = build_index()
AVG_DOC_LENGTH = sum(len(d["tokens"]) for d in DOCUMENTS) / len(DOCUMENTS)
K1, B = 1.2, 0.75


def expand_query(query):
    """Widen the query so meaning survives rewording.

    Deliberately NOT shown the knowledge base topics: an earlier version was,
    and for "Manchester office" it suggested "bristol" because a KB topic
    mentioned Bristol. Retrieval then matched its own suggestion.
    """
    result = call_json(
        "You expand customer support search queries into keywords.\n\n"
        f'Customer query: "{query}"\n\n'
        "Return ONLY a JSON array of 4-10 lowercase single words: the customer's "
        "meaningful words plus close synonyms and generic related terms.\n"
        "Do NOT invent place names, brand names, product names, numbers or any "
        "specific detail the customer did not mention.",
        max_tokens=200,
    )
    if isinstance(result, list):
        return {str(t).lower() for t in result if isinstance(t, (str, int))}
    return set()


def score_documents(query_tokens):
    """IDF-weighted overlap with BM25 length normalisation."""
    weight_total = sum(IDF.get(t, 1.0) for t in query_tokens) or 1.0
    results = []
    for document in DOCUMENTS:
        matched = query_tokens & document["tokens"]
        if not matched:
            continue
        penalty = K1 * (1 - B + B * len(document["tokens"]) / AVG_DOC_LENGTH) + 1
        weight = (K1 + 1) / penalty
        score = sum(IDF.get(t, 1.0) * weight for t in matched) / weight_total
        results.append((score, matched, document["entry"]))
    results.sort(key=lambda row: row[0], reverse=True)
    return results


# =========================================================
# WEAKNESS 2 — RERANKING
# =========================================================


def rerank(query, candidates):
    """Have a model read each candidate and judge whether it answers the query.

    Lexical scoring counts word overlap; it cannot see that "not branch opening
    hours" is a denial, or that an entry shares words while answering a
    different question. Reranking reads. Retrieve widely, then filter by
    comprehension — the standard two-stage pattern in production search.

    Falls back to the lexical order if the model call fails: degraded, not broken.
    """
    if not candidates:
        return []

    listing = "\n\n".join(
        f"[{index}] topic: {entry['topic']}\ncontent: {entry['content']}"
        for index, (_, _, entry) in enumerate(candidates)
    )
    result = call_json(
        "Decide which knowledge base entries genuinely answer the customer's "
        "question. Judge meaning, not shared words. An entry that merely "
        "mentions the subject, or explicitly says it does NOT cover it, is not "
        "relevant.\n\n"
        f'CUSTOMER QUESTION: "{query}"\n\nENTRIES:\n{listing}\n\n'
        'Return ONLY: {"relevant": [0, 2]} listing the indexes that genuinely '
        'answer the question, best first. Return {"relevant": []} if none do.',
        max_tokens=300,
    )

    if not isinstance(result, dict) or "relevant" not in result:
        print("  [rerank unavailable — using lexical order]")
        return candidates[:TOP_K]

    kept = []
    for index in result["relevant"]:
        if isinstance(index, int) and 0 <= index < len(candidates):
            kept.append(candidates[index])
    dropped = len(candidates) - len(kept)
    if dropped:
        print(f"  [rerank dropped {dropped} of {len(candidates)} candidates]")
    return kept[:TOP_K]


def search_knowledge_base(query):
    raw_tokens = tokenize(query)
    expanded = expand_query(query)
    all_tokens = raw_tokens | tokenize(" ".join(expanded))

    added = sorted(all_tokens - raw_tokens)
    if added:
        print(f"  [expanded +{len(added)}: {', '.join(added[:8])}]")

    scored = score_documents(all_tokens)
    candidates = [row for row in scored if row[0] >= RELEVANCE_FLOOR][:CANDIDATES]

    if not candidates:
        best = f"{scored[0][0]:.2f}" if scored else "0.00"
        print(f"  [no candidates above floor {RELEVANCE_FLOOR} — best was {best}]")

    kept = rerank(query, candidates) if USE_RERANK else candidates[:TOP_K]

    if candidates and not kept:
        print(f"  [rerank rejected all {len(candidates)} candidates as irrelevant]")

    if not kept:
        return (
            "NO_MATCH: nothing in the knowledge base answers this question. "
            "Do not answer from your own knowledge and do not infer. Escalate."
        )

    blocks = []
    for score, _, entry in kept:
        print(f"  [kept {score:.2f}] {entry['topic']}")
        blocks.append(
            f"[{entry['topic']}] (relevance {score:.2f})\n"
            f"{entry['content']}\nSource: {entry['source']}"
        )
    return "\n\n".join(blocks)


# =========================================================
# TOOLS
# =========================================================

CLIENT_QUERIES = {
    "FW-1001": "Received 1 Aug. With your adviser. Response due by 6 Aug.",
    "FW-1002": "Documents verified. Awaiting your signature to proceed.",
    "FW-1003": "Closed 28 July. Summary letter posted to your address.",
    "FW-1004": "Escalated to the Client Resolution team on 4 Aug.",
}


def look_up_client_query(reference):
    reference = reference.upper().strip()
    status = CLIENT_QUERIES.get(reference)
    if status is None:
        return (
            f"NO_RECORD: no query found with reference {reference}. "
            "Ask the customer to check the number. Do not speculate."
        )
    return (
        f"reference: {reference}\n"
        f"STATUS_VERBATIM: {status}\n"
        "Reproduce STATUS_VERBATIM word for word inside quotation marks. Add no "
        "detail, no interpretation, no reassurance about what happens next."
    )


def escalate_to_adviser(reason, summary):
    print(f"\n  [!] ESCALATION LOGGED — {reason}")
    print(f"      {summary}")
    return (
        "Escalation logged. Tell the customer a qualified adviser will contact "
        "them, and give the switchboard number 0117 496 0100."
    )


TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "look_up_client_query": look_up_client_query,
    "escalate_to_adviser": escalate_to_adviser,
}

tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the company knowledge base for facts about services, offices, "
            "phone numbers, the client portal, complaints, appointments and account "
            "admin.\n\nRun one search per distinct topic. Pass the customer's own "
            "wording. If the result begins with NO_MATCH, escalate rather than "
            "improvising."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "One topic only"}},
            "required": ["query"],
        },
    },
    {
        "name": "look_up_client_query",
        "description": (
            "Look up the live status of a client query by reference, format FW-XXXX. "
            "This data changes: always call it fresh, even for a reference looked up "
            "earlier. The result contains STATUS_VERBATIM, which must be reproduced "
            "exactly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reference": {"type": "string", "description": "e.g. FW-1002"}},
            "required": ["reference"],
        },
    },
    {
        "name": "escalate_to_adviser",
        "description": (
            "Hand to a qualified human adviser. MUST be used for financial, "
            "investment, tax or pension advice; questions about a specific person's "
            "money; complaints the customer wants to raise now; bereavement, divorce "
            "or power of attorney; and anything returning NO_MATCH."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Short category"},
                "summary": {"type": "string", "description": "One sentence"},
            },
            "required": ["reason", "summary"],
        },
    },
]

SYSTEM_PROMPT = """You are a customer support assistant for Fernhill Wealth, a UK wealth
management firm.

FAITHFULNESS RULES — these override helpfulness:
1. Every fact you state must appear in a tool result from THIS turn.
2. Reproduce STATUS_VERBATIM word for word, inside quotation marks.
3. Add nothing. No "on file", no "covering the outcome", no reassurance the
   evidence does not contain. Terse and accurate beats warm and embellished.
4. Never turn a target into a promise. "Response due by 6 Aug" is not "you will
   hear by 6 Aug".
5. URLs, phone numbers and figures must be copied EXACTLY as given. Never
   shorten a URL to a parent path, never extend one, never reconstruct a
   website address from memory. If the evidence lacks a URL, name the page in
   words instead of guessing its address.
6. If evidence is missing, say what you do not know and escalate.

CONDUCT RULES:
7. You are NOT a financial adviser. Escalate all advice questions immediately.
8. Multi-part questions: call a tool for each part.
9. Never ask for account numbers, passwords, dates of birth, NI numbers or
   card details.
10. Be brief and plain. Two short paragraphs maximum.
"""


# =========================================================
# HISTORY EXPIRY
# =========================================================


def expire_stale_results(messages):
    id_to_tool = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                id_to_tool[block.id] = block.name

    wiped = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            volatile = id_to_tool.get(block.get("tool_use_id")) in VOLATILE_TOOLS
            if volatile and block["content"] != EXPIRED_NOTICE:
                block["content"] = EXPIRED_NOTICE
                wiped += 1
    return wiped


# =========================================================
# GROUNDEDNESS CHECK
# =========================================================


def verify_answer(answer, evidence):
    """Read the answer back against the evidence. Three honest states."""
    if not evidence.strip():
        return {"status": "unavailable", "unsupported": []}

    result = call_json(
        "You are a strict fact-checker. Compare the ANSWER against the EVIDENCE.\n\n"
        "Flag any claim not directly supported: added details, softened or "
        "strengthened wording, implied guarantees, invented reassurance. Ordinary "
        "politeness is fine and must not be flagged.\n\n"
        f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}\n\n"
        'Return ONLY: {"grounded": true, "unsupported": []}',
        max_tokens=500,
    )
    if not isinstance(result, dict):
        print("  [!] VERIFICATION UNAVAILABLE — answer NOT fact-checked")
        return {"status": "unavailable", "unsupported": []}

    unsupported = [str(c) for c in result.get("unsupported", [])]
    grounded = bool(result.get("grounded", True)) and not unsupported
    return {
        "status": "grounded" if grounded else "unsupported",
        "unsupported": unsupported,
    }


# =========================================================
# CODE-ENFORCED ESCALATION POLICY
# =========================================================

UNRESOLVED_MARKERS = ("NO_MATCH", "NO_RECORD")

PLEASANTRIES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "thanks a lot",
    "cheers",
    "ok",
    "okay",
    "bye",
    "goodbye",
    "great",
    "perfect",
    "no thanks",
    "that's all",
    "thats all",
}


def enforce_escalation(answer, evidence_parts, model_escalated, state):
    """Guarantee that an unanswerable question reaches a human.

    The prompt already says NO_MATCH must escalate. In testing the model broke
    that rule on the query "quiz" and asked a clarifying question instead --
    which was BETTER than the rule, because escalating one ambiguous word wastes
    an adviser's time. So the fix is not to force escalation harder; it is to
    encode the policy a good human agent actually follows:

        clarify once, then escalate.

    Turn 1 unanswerable -> one clarifying question is allowed.
    Turn 2 still unanswerable -> escalation happens in Python, whatever the
    model decided. A customer can never be left going round in circles.
    """
    unresolved = any(marker in part for part in evidence_parts for marker in UNRESOLVED_MARKERS)

    if not unresolved or model_escalated:
        state["unclear_turns"] = 0
        return answer, False

    asked_for_clarification = "?" in answer
    if asked_for_clarification and state["unclear_turns"] == 0:
        state["unclear_turns"] = 1
        print("  [policy: clarification allowed once]")
        return answer, False

    state["unclear_turns"] = 0
    print("  [!] POLICY: unresolved and not escalated — escalating in code")
    escalate_to_adviser(
        "unresolved query (policy-enforced)",
        "The knowledge base had no answer and the assistant did not escalate.",
    )
    # Deliberately NOT appending a sentence here. An earlier version did, and
    # produced a self-contradicting message: the model offered to escalate
    # ("or I can pass this to an adviser?") and the appended text then claimed
    # it already had. Bolting text onto finished prose does not work -- the
    # answer has to be rewritten knowing escalation has happened.
    return answer, True


# =========================================================
# THE AGENT LOOP
# =========================================================


def rewrite(messages, problems):
    """Send the answer back to be rewritten strictly from evidence."""
    messages.append(
        {
            "role": "user",
            "content": (
                "Your previous answer had these problems: "
                + "; ".join(problems)
                + ". Rewrite it using only what the tool results actually say. Remove "
                "every added detail. Copy URLs and numbers exactly or omit them "
                "entirely. Do not call any tools."
            ),
        }
    )
    response = client.messages.create(
        model=MAIN_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})
    return "".join(b.text for b in response.content if b.type == "text")


def rewrite_after_escalation(messages):
    """Rewrite the answer as a single coherent message, post-escalation.

    Escalation is now a fact, so the answer must state it as done: no offering
    to escalate, no dangling question inviting a reply the customer no longer
    needs to give.
    """
    messages.append(
        {
            "role": "user",
            "content": (
                "This query has now been escalated to a human adviser automatically, "
                "because the knowledge base could not answer it. Rewrite your last "
                "message as one short, coherent reply that: says plainly you do not "
                "have the information; states that it HAS BEEN passed to a qualified "
                "adviser who will be in touch; gives the switchboard number "
                "0117 496 0100. Do not offer to escalate, since it has already "
                "happened, and do not ask the customer any question. Do not call "
                "any tools."
            ),
        }
    )
    response = client.messages.create(
        model=MAIN_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})
    return "".join(b.text for b in response.content if b.type == "text")


def run_agent(messages, state):
    wiped = expire_stale_results(messages)
    if wiped:
        print(f"  [expired {wiped} old result(s)]")

    evidence_parts = []
    model_escalated = False

    # A greeting needs no evidence, so forcing a tool call just burns an API
    # call searching the knowledge base for "hello". Deterministic list only:
    # anything not on it is treated as a real question and still forced.
    last_message = next(
        (
            m["content"]
            for m in reversed(messages)
            if m["role"] == "user" and isinstance(m["content"], str)
        ),
        "",
    )
    is_greeting = last_message.lower().strip(" .!?,") in PLEASANTRIES

    for step in range(MAX_STEPS):
        force = FORCE_TOOL_FIRST and step == 0 and not is_greeting
        tool_choice = {"type": "any"} if force else {"type": "auto"}

        response = client.messages.create(
            model=MAIN_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            evidence = "\n\n".join(evidence_parts)
            notes = []

            # Deterministic check first: it is certain, and it is cheap.
            if CHECK_CITATIONS and evidence:
                violations = check_citations(answer, evidence)
                if violations:
                    print("\n  [!] CITATION VIOLATIONS (deterministic):")
                    for problem in violations:
                        print(f"      - {problem}")
                    answer = rewrite(messages, violations)
                    notes.append("rewritten after citation check")

                    remaining = check_citations(answer, evidence)
                    if remaining:
                        print("  [!] STILL UNSUPPORTED after rewrite — flagging")
                        notes.append(f"WARNING: unverified detail remains ({len(remaining)})")
                    else:
                        print("  [citations: clean after rewrite]")
                else:
                    print("  [citations: clean]")

            # Then the judgement-based check.
            if VERIFY_ANSWERS and evidence:
                verdict = verify_answer(answer, evidence)
                if verdict["status"] == "unsupported":
                    print("\n  [!] UNSUPPORTED CLAIMS:")
                    for claim in verdict["unsupported"]:
                        print(f"      - {claim}")
                    answer = rewrite(messages, verdict["unsupported"])
                    notes.append("rewritten after fact-check")
                elif verdict["status"] == "grounded":
                    print("  [verified: grounded]")
                else:
                    notes.append("WARNING: fact-check unavailable")

            # Policy runs last, on the final wording, and cannot be overridden.
            answer, forced = enforce_escalation(answer, evidence_parts, model_escalated, state)
            if forced:
                answer = rewrite_after_escalation(messages)
                notes.append("escalated by policy")

            if notes:
                answer += "\n\n  [" + "; ".join(notes) + "]"
            if SHOW_AUDIT and evidence:
                answer += "\n\n  --- audit trail (raw evidence) ---\n  "
                answer += evidence.replace("\n", "\n  ")
            return answer

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({block.input})")
            if block.name == "escalate_to_adviser":
                model_escalated = True
            try:
                result = TOOL_FUNCTIONS[block.name](**block.input)
            except Exception as error:
                result = f"Tool failed: {error}"
            evidence_parts.append(f"{block.name} -> {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Sorry, I could not complete that. Please call 0117 496 0100."


if __name__ == "__main__":
    print("Support bot ready. Type 'quit' to exit.\n")
    conversation = []
    state = {"unclear_turns": 0}  # survives across turns; policy needs memory
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"quit", "exit"}:
            break
        if not user_input.strip():
            continue
        conversation.append({"role": "user", "content": user_input})
        print(f"\nBot: {run_agent(conversation, state)}\n")
