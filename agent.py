"""
Grounded support agent — retrieval, verification and escalation.

A customer support assistant for Fernhill Wealth, a fictional UK wealth
management firm. It answers only what it can evidence, quotes volatile data
verbatim, and hands to a human when it cannot help.

Every hard guarantee here lives in Python rather than in the system prompt,
because a prompt is a strong suggestion and nothing more. The guards, in the
order they run:

  EVIDENCE GUARD — deterministic.
      A real question must be answered from a tool result. If the model
      answers from memory it is told to look the question up; if it still
      does not, the turn is treated as unanswerable and escalated. Current
      models reject forced tool use (tool_choice "any"), so the guarantee is
      enforced after the fact in code instead of before the fact by the API.

  CITATION GUARD — deterministic.
      Every URL and number in an answer must match the evidence exactly.
      Added after the agent derived "fernhillwealth.example/offices" by
      truncating a deeper URL it had been given: plausible, probably correct,
      not in the evidence, and waved through by the LLM checker. Code cannot
      be persuaded that something looks right.

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
for precision. Lexical scoring counts words and cannot see that "not office
opening hours" is a denial; reranking reads.

Every run returns an AgentResult: the customer-facing answer, the notes about
what the guards did, the raw evidence, and a structured trace of every
pipeline event. Nothing is communicated through stdout.

Model non-determinism is irreducible. That is why the guarantees are in code
and why every answer carries its evidence.

Personal learning project. Fernhill Wealth is invented; see knowledge_base.py.
Run `pytest` for the test suite.
"""

from __future__ import annotations

import contextvars
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv

from config import Settings
from knowledge_base import KNOWLEDGE_BASE

load_dotenv()
client = Anthropic()

SETTINGS = Settings.from_env()

MAIN_MODEL = SETTINGS.main_model
HELPER_MODEL = SETTINGS.helper_model

MAX_STEPS = SETTINGS.max_steps
RELEVANCE_FLOOR = SETTINGS.relevance_floor
CANDIDATES = SETTINGS.candidates
TOP_K = SETTINGS.top_k

USE_RERANK = SETTINGS.use_rerank
VERIFY_ANSWERS = SETTINGS.verify_answers
CHECK_CITATIONS = SETTINGS.check_citations
REQUIRE_EVIDENCE = SETTINGS.require_evidence

VOLATILE_TOOLS = {"look_up_client_query"}
EXPIRED_NOTICE = (
    "[This earlier result has expired. Do not rely on it or repeat it. Look the "
    "reference up afresh only if the customer asks about it again.]"
)

SWITCHBOARD = "0117 496 0100"


# =========================================================
# TRACE — structured record of what the pipeline did
# =========================================================


@dataclass
class Event:
    kind: str
    message: str
    data: dict = field(default_factory=dict)


class Trace:
    """An ordered list of pipeline events for one turn.

    The web interface, the tests and the audit log all read this instead of
    parsing printed text, so wording changes cannot silently break them.
    """

    def __init__(self):
        self.events: list[Event] = []

    def log(self, kind: str, message: str, **data) -> None:
        self.events.append(Event(kind, message, data))

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]

    def find(self, kind: str) -> list[Event]:
        return [event for event in self.events if event.kind == kind]

    def render(self) -> str:
        """One line per event; structured data follows the message when present."""
        lines = []
        for event in self.events:
            line = f"[{event.kind}] {event.message}"
            if event.data:
                details = ", ".join(f"{key}={value!r}" for key, value in event.data.items())
                line += f"  ({details})"
            lines.append(line)
        return "\n".join(lines)


_CURRENT_TRACE: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "support_bot_trace", default=None
)


def log(kind: str, message: str, **data) -> None:
    """Record an event on the current turn's trace.

    Outside a run (for example a function called directly from the terminal)
    there is no trace, so the event is printed instead — nothing is lost.
    """
    trace = _CURRENT_TRACE.get()
    if trace is None:
        print(f"  [{kind}] {message}")
    else:
        trace.log(kind, message, **data)


@dataclass
class AgentResult:
    """What one turn produces. `answer` is the only part a customer sees."""

    answer: str
    notes: list[str]
    evidence: list[str]
    trace: Trace
    escalated: bool
    tools_used: list[str]
    models: dict

    def render(self) -> str:
        """Terminal view: the answer, then the notes and the raw evidence."""
        text = self.answer
        if self.notes:
            text += "\n\n  [" + "; ".join(self.notes) + "]"
        if self.evidence:
            text += "\n\n  --- audit trail (raw evidence) ---\n  "
            text += "\n\n".join(self.evidence).replace("\n", "\n  ")
        return text


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


def text_of(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def call_json(prompt, max_tokens=None, attempts=2):
    """Call the helper model and parse JSON, with one retry. None on failure."""
    for attempt in range(attempts):
        try:
            response = client.messages.create(
                model=HELPER_MODEL,
                max_tokens=max_tokens or SETTINGS.helper_max_tokens,
                output_config={"effort": SETTINGS.helper_effort},
                messages=[{"role": "user", "content": prompt}],
            )
            payload = extract_json(text_of(response))
            if payload is None:
                raise ValueError("no JSON in helper response")
            return json.loads(payload)
        except Exception as error:
            if attempt == attempts - 1:
                log("helper_failed", f"helper call failed: {error}", error=str(error))
                return None


# =========================================================
# CITATION GUARD — deterministic
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
    """Reduce to digits only, so '0117 496 0100' == '01174960100'."""
    return re.sub(r"\D", "", text)


def number_tokens(text):
    """Every number in a text, as a set of digit strings.

    Each number is a token in its own right. An earlier version joined all the
    evidence digits into one string and tested substrings against it, which
    let any figure pass if its digits happened to appear across the join of two
    unrelated numbers — "0119 98" against "FW-1001 ... 1998", for example.
    """
    return {normalise_number(match) for match in NUMBER_PATTERN.findall(text)} - {""}


def check_citations(answer, evidence):
    """Require every URL and number in the answer to appear in the evidence.

    This is the only guarantee in the whole system that cannot be talked out
    of. The LLM verifier passed 'fernhillwealth.example/offices' because it
    looked right and was a substring of a longer URL. This function demands an
    EXACT match after normalisation, so truncating a path to guess a parent is
    caught, and so is a number assembled from pieces of other numbers.

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

    evidence_numbers = number_tokens(evidence)
    for raw in NUMBER_PATTERN.findall(answer):
        digits = normalise_number(raw)
        if not digits or digits in NUMBER_ALLOWLIST:
            continue
        if digits not in evidence_numbers:
            violations.append(f"Number '{raw.strip()}' does not appear in the evidence.")

    return violations


# =========================================================
# RETRIEVAL — BM25, normalised to the query's ceiling
# =========================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "i", "im", "my", "me", "we", "you", "your", "how", "do", "does", "did",
    "what", "when", "where", "which", "who", "can", "could", "would", "should",
    "for", "in", "on", "at", "it", "this", "that", "with", "about", "from",
    "have", "has", "had", "be", "been", "get", "got", "any", "please", "thanks",
    "long", "take", "takes", "make", "made", "need", "want", "know", "tell",
    "help", "like", "just", "also", "there", "their", "am", "if", "so", "but",
    "not", "no", "yes", "all", "out", "up", "by", "as",
}  # fmt: skip


def tokenize(text):
    """Lower-case word tokens with stopwords removed and plurals folded.

    Returns a list, not a set: BM25 needs to know how often a term occurs in
    a document, not only whether it occurs.
    """
    tokens = []
    for word in re.findall(r"[a-z0-9@.]+", text.lower()):
        if word in STOPWORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


K1, B = 1.2, 0.75


def build_index():
    """Term frequencies per document, document lengths, and BM25 idf per term."""
    documents = []
    for entry in KNOWLEDGE_BASE:
        tokens = tokenize(entry["topic"] + " " + entry["content"])
        documents.append({"entry": entry, "tf": Counter(tokens), "length": len(tokens)})
    total = len(documents)
    document_frequency = Counter(term for document in documents for term in document["tf"])
    idf = {
        term: math.log(1 + (total - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }
    return documents, idf


DOCUMENTS, IDF = build_index()
AVG_DOC_LENGTH = sum(document["length"] for document in DOCUMENTS) / len(DOCUMENTS)


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
    )
    if isinstance(result, list):
        return {str(t).lower() for t in result if isinstance(t, (str, int))}
    return set()


def score_documents(query_tokens):
    """BM25 (Robertson/Spärck Jones idf, tf saturation, length normalisation).

    Scores are divided by the query's saturation ceiling — the score a document
    would get if every query term the corpus knows appeared in it with unbounded
    frequency — so they fall between 0 and 1 and the relevance floor means the
    same thing for a two-word query as for a ten-word one.

    Query terms the corpus has never seen are left out of the ceiling. Query
    expansion routinely adds words such as "premises" or "location" that no
    entry contains; an earlier version counted them at full weight, and the
    inflated ceiling pushed "Do you have a Manchester office?" below the
    relevance floor even though the offices entry names Manchester.
    """
    known_terms = {term for term in query_tokens if term in IDF}
    ceiling = sum(IDF[term] * (K1 + 1) for term in known_terms) or 1.0
    results = []
    for document in DOCUMENTS:
        matched = known_terms & document["tf"].keys()
        if not matched:
            continue
        length_norm = K1 * (1 - B + B * document["length"] / AVG_DOC_LENGTH)
        score = 0.0
        for term in matched:
            tf = document["tf"][term]
            score += IDF[term] * tf * (K1 + 1) / (tf + length_norm)
        results.append((score / ceiling, matched, document["entry"]))
    results.sort(key=lambda row: row[0], reverse=True)
    return results


# =========================================================
# RERANKING
# =========================================================


def rerank(query, candidates):
    """Have a model read each candidate and judge whether it answers the query.

    Lexical scoring counts word overlap; it cannot see that "not office opening
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
        "Decide which knowledge base entries help answer the customer's question. "
        "Judge meaning, not shared words.\n\n"
        "An entry is RELEVANT if it contains information that answers the question "
        "in whole or in part, even when the entry is mostly about something else. "
        "For example, an entry listing the firm's offices answers \"do you have an "
        'office in X" whenever X is on the list, even if the rest of the entry '
        "describes a different office.\n\n"
        "An entry is NOT relevant if it only shares vocabulary with the question, "
        'or explicitly says it does NOT cover what was asked (for example, "these '
        'are callback slots, not office opening hours" does not answer a question '
        "about opening hours).\n\n"
        f'CUSTOMER QUESTION: "{query}"\n\nENTRIES:\n{listing}\n\n'
        'Return ONLY: {"relevant": [0, 2], "reason": "<one sentence>"} listing the '
        "indexes of relevant entries, best first, and a one-sentence reason. "
        'Return {"relevant": [], "reason": "..."} if none are relevant.',
    )

    if not isinstance(result, dict) or "relevant" not in result:
        log("rerank_unavailable", "rerank unavailable — using lexical order")
        return candidates[:TOP_K]

    kept = []
    for index in result["relevant"]:
        if isinstance(index, int) and 0 <= index < len(candidates):
            kept.append(candidates[index])
    dropped = len(candidates) - len(kept)
    reason = str(result.get("reason", "")).strip()
    if dropped:
        log(
            "rerank",
            f"rerank dropped {dropped} of {len(candidates)} candidates",
            dropped=dropped,
            total=len(candidates),
            reason=reason,
        )
    return kept[:TOP_K]


def search_knowledge_base(query):
    raw_tokens = set(tokenize(query))
    expanded = expand_query(query)
    all_tokens = raw_tokens | set(tokenize(" ".join(expanded)))

    added = sorted(all_tokens - raw_tokens)
    if added:
        log("expanded", f"expanded +{len(added)}: {', '.join(added[:8])}", terms=added)

    scored = score_documents(all_tokens)
    candidates = [row for row in scored if row[0] >= RELEVANCE_FLOOR][:CANDIDATES]

    if not candidates:
        best = scored[0][0] if scored else 0.0
        log(
            "no_candidates",
            f"no candidates above floor {RELEVANCE_FLOOR} — best was {best:.2f}",
            best=best,
        )

    kept = rerank(query, candidates) if USE_RERANK else candidates[:TOP_K]

    if candidates and not kept:
        log("rerank_rejected_all", f"rerank rejected all {len(candidates)} candidates")

    if not kept:
        return (
            "NO_MATCH: nothing in the knowledge base answers this question. "
            "Do not answer from your own knowledge and do not infer. Escalate."
        )

    blocks = []
    for score, _, entry in kept:
        log("kept", f"kept {score:.2f} {entry['topic']}", score=score, topic=entry["topic"])
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
    log("escalation", f"ESCALATION LOGGED — {reason}: {summary}", reason=reason, summary=summary)
    return (
        "Escalation logged. Tell the customer a qualified adviser will contact "
        f"them, and give the switchboard number {SWITCHBOARD}."
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
1. Every fact you state must appear in a tool result from THIS turn. Always
   look a question up with a tool before answering it; never answer from
   memory, even when you think you know the answer.
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
8. Multi-part questions: call a tool for each part. Answer only the customer's
   latest message: do not repeat earlier answers or re-fetch references from
   earlier turns unless the customer asks about them again.
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
        "Flag any factual claim not directly supported: added details, softened or "
        "strengthened wording, implied guarantees, invented reassurance.\n\n"
        "Do NOT flag, because they are not factual claims about the evidence:\n"
        "- ordinary politeness and offers of further help;\n"
        "- statements about the assistant's own role or limits, such as that it "
        "cannot give financial, investment, tax or pension advice;\n"
        "- statements that the matter has been escalated or that an adviser will be "
        "in touch, when the evidence records an escalation.\n\n"
        f"EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}\n\n"
        'Return ONLY: {"grounded": true, "unsupported": []}',
    )
    if not isinstance(result, dict):
        log("verification_unavailable", "VERIFICATION UNAVAILABLE — answer NOT fact-checked")
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
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thanks a lot", "cheers", "ok", "okay", "bye",
    "goodbye", "great", "perfect", "no thanks", "that's all", "thats all",
}  # fmt: skip


def is_pleasantry(text):
    return text.lower().strip(" .!?,") in PLEASANTRIES


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
        log("policy_clarify", "policy: clarification allowed once")
        return answer, False

    state["unclear_turns"] = 0
    log("policy_escalate", "POLICY: unresolved and not escalated — escalating in code")
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


def _main_call(messages, use_tools):
    request = {
        "model": MAIN_MODEL,
        "max_tokens": SETTINGS.main_max_tokens,
        "output_config": {"effort": SETTINGS.main_effort},
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    if use_tools:
        request["tools"] = tools
    return client.messages.create(**request)


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
    response = _main_call(messages, use_tools=False)
    messages.append({"role": "assistant", "content": response.content})
    return text_of(response)


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
                f"{SWITCHBOARD}. Do not offer to escalate, since it has already "
                "happened, and do not ask the customer any question. Do not call "
                "any tools."
            ),
        }
    )
    response = _main_call(messages, use_tools=False)
    messages.append({"role": "assistant", "content": response.content})
    return text_of(response)


EVIDENCE_GUARD_RETRY = (
    "You answered without consulting any tool. Every fact must come from a tool "
    "result in this turn. Look the question up with the appropriate tool now, "
    "then answer from what it returns."
)
EVIDENCE_GUARD_NO_MATCH = (
    "evidence_guard -> NO_MATCH: the assistant answered twice without consulting "
    "any tool, so the answer cannot be evidenced. Escalate."
)


def run_agent(messages, state) -> AgentResult:
    """Run one customer turn.

    `messages` is the conversation so far (the newest user message last);
    `state` persists across turns because the escalation policy needs memory.
    """
    trace = Trace()
    token = _CURRENT_TRACE.set(trace)
    try:
        return _run_turn(messages, state, trace)
    finally:
        _CURRENT_TRACE.reset(token)


def _run_turn(messages, state, trace) -> AgentResult:
    wiped = expire_stale_results(messages)
    if wiped:
        log("expired", f"expired {wiped} old result(s)", count=wiped)

    evidence_parts: list[str] = []
    tools_used: list[str] = []
    model_escalated = False
    guard_retries = 0

    last_message = next(
        (
            m["content"]
            for m in reversed(messages)
            if m["role"] == "user" and isinstance(m["content"], str)
        ),
        "",
    )
    # A greeting needs no evidence. Deterministic list only: anything not on
    # it is treated as a real question and must be answered from a tool.
    is_greeting = is_pleasantry(last_message)

    models = {"main": MAIN_MODEL, "helper": HELPER_MODEL}

    for _step in range(MAX_STEPS):
        response = _main_call(messages, use_tools=True)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log("tool", f"{block.name}({block.input})", name=block.name, input=block.input)
                tools_used.append(block.name)
                if block.name == "escalate_to_adviser":
                    model_escalated = True
                try:
                    result = TOOL_FUNCTIONS[block.name](**block.input)
                except Exception as error:
                    result = f"Tool failed: {error}"
                    log("tool_failed", f"{block.name} failed: {error}", name=block.name)
                evidence_parts.append(f"{block.name} -> {result}")
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        answer = text_of(response)

        # EVIDENCE GUARD: a real question answered without any tool call is an
        # answer from memory. Tell the model once; if it still will not look
        # the question up, the turn is unanswerable and the policy escalates.
        if REQUIRE_EVIDENCE and not is_greeting and not evidence_parts:
            if guard_retries == 0:
                guard_retries = 1
                log("evidence_guard", "answered without a tool — asking for a lookup")
                messages.append({"role": "user", "content": EVIDENCE_GUARD_RETRY})
                continue
            log("evidence_guard_failed", "still no tool call — treating as unanswerable")
            evidence_parts.append(EVIDENCE_GUARD_NO_MATCH)

        return _finish(
            answer, messages, evidence_parts, tools_used, model_escalated, state, trace, models
        )

    log("max_steps", f"gave up after {MAX_STEPS} steps")
    return AgentResult(
        answer=f"Sorry, I could not complete that. Please call {SWITCHBOARD}.",
        notes=["WARNING: step limit reached"],
        evidence=evidence_parts,
        trace=trace,
        escalated=False,
        tools_used=tools_used,
        models=models,
    )


def _finish(answer, messages, evidence_parts, tools_used, model_escalated, state, trace, models):
    evidence = "\n\n".join(evidence_parts)
    notes = []

    # Deterministic check first: it is certain, and it is cheap.
    if CHECK_CITATIONS and evidence:
        violations = check_citations(answer, evidence)
        if violations:
            log("citation_violation", "CITATION VIOLATIONS (deterministic)", violations=violations)
            answer = rewrite(messages, violations)
            notes.append("rewritten after citation check")

            remaining = check_citations(answer, evidence)
            if remaining:
                log("citations_remaining", "STILL UNSUPPORTED after rewrite", violations=remaining)
                notes.append(f"WARNING: unverified detail remains ({len(remaining)})")
            else:
                log("citations_clean", "citations: clean after rewrite", after_rewrite=True)
        else:
            log("citations_clean", "citations: clean", after_rewrite=False)

    # Then the judgement-based check.
    if VERIFY_ANSWERS and evidence:
        verdict = verify_answer(answer, evidence)
        if verdict["status"] == "unsupported":
            log("unsupported", "UNSUPPORTED CLAIMS", claims=verdict["unsupported"])
            answer = rewrite(messages, verdict["unsupported"])
            notes.append("rewritten after fact-check")
        elif verdict["status"] == "grounded":
            log("verified", "verified: grounded")
        else:
            notes.append("WARNING: fact-check unavailable")

    # Policy runs last, on the final wording, and cannot be overridden.
    answer, forced = enforce_escalation(answer, evidence_parts, model_escalated, state)
    if forced:
        answer = rewrite_after_escalation(messages)
        notes.append("escalated by policy")

    return AgentResult(
        answer=answer,
        notes=notes,
        evidence=list(evidence_parts),
        trace=trace,
        escalated=model_escalated or forced,
        tools_used=tools_used,
        models=models,
    )


if __name__ == "__main__":
    print("Support bot ready. Type 'quit' to exit.\n")
    conversation = []
    policy_state = {"unclear_turns": 0}  # survives across turns; policy needs memory
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"quit", "exit"}:
            break
        if not user_input.strip():
            continue
        conversation.append({"role": "user", "content": user_input})
        result = run_agent(conversation, policy_state)
        print("\n" + result.trace.render())
        print(f"\nBot: {result.render()}\n")
