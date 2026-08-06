"""
Step 8 — the gold-standard agent.

Fixes two failure modes that survive every earlier guardrail:

  PROBLEM 1: Blind retrieval.
      "I'm unhappy with the service" shares no words with the complaints
      entry, so keyword search returns nothing.
      FIX: LLM query expansion + IDF-weighted scoring + a relevance floor.

  PROBLEM 2: Paraphrase drift.
      The tool returns "Summary letter posted to your address" and the bot
      writes "a summary letter covering the outcome... on file". Correct data,
      embellished output.
      FIX: verbatim quoting of volatile facts, plus an automated groundedness
      check that reads the final answer back against the evidence.

Personal learning project. Not affiliated with Evelyn Partners.
"""

import json
import math
import re

from dotenv import load_dotenv
from anthropic import Anthropic
from knowledge_base import KNOWLEDGE_BASE

load_dotenv()
client = Anthropic()

MAIN_MODEL = "claude-sonnet-5"                  # the agent
HELPER_MODEL = "claude-haiku-4-5-20251001"      # expansion + verification (cheap, fast)

MAX_STEPS = 8
RELEVANCE_FLOOR = 0.18      # below this, we return nothing rather than noise
TOP_K = 2

VOLATILE_TOOLS = {"look_up_client_query"}
EXPIRED_NOTICE = (
    "[This earlier result has expired and may no longer be accurate. "
    "Call the tool again if you need this information.]"
)

FORCE_TOOL_FIRST = True
VERIFY_ANSWERS = True       # run the groundedness check
SHOW_AUDIT = True           # print raw evidence under each answer


# =========================================================
# PART 1 — BETTER RETRIEVAL
# =========================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "i", "im", "my", "me", "we", "you", "your", "how", "do", "does", "did",
    "what", "when", "where", "which", "who", "can", "could", "would", "should",
    "for", "in", "on", "at", "it", "this", "that", "with", "about", "from",
    "have", "has", "had", "be", "been", "get", "got", "any", "please", "thanks",
    # Generic words that carry no topic information but happen to appear
    # inside knowledge base entries ("long-term care", "take the following").
    # Without these, "how long does a complaint take" matches the financial
    # planning entry on the word "long". A real false positive, found in testing.
    "long", "take", "takes", "make", "made", "need", "want", "know", "tell",
    "help", "like", "just", "also", "there", "their", "any", "any", "am",
    "if", "so", "but", "not", "no", "yes", "all", "out", "up", "by", "as",
}


def extract_json(text):
    """Pull the first complete JSON value out of a model response.

    Models append explanations, wrap output in code fences, or emit a second
    object. json.loads() dies on all three ('Extra data: line 2 column 1').
    We scan for the first balanced {...} or [...] and ignore everything else.
    Never trust a model to return bare JSON just because you asked.
    """
    if not text:
        return None
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
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
                return text[start:index + 1]
    return None


def tokenize(text):
    """Lowercase, strip punctuation, drop stopwords, crudely singularise."""
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
    """Pre-compute tokens and IDF weights once, at import time.

    IDF = inverse document frequency. A word appearing in every entry
    ('evelyn', 'client') tells us nothing. A word appearing in one entry
    ('ombudsman') is highly diagnostic. IDF encodes exactly that, so rare
    words dominate the score instead of every word counting equally.
    """
    documents = []
    for entry in KNOWLEDGE_BASE:
        tokens = tokenize(entry["topic"] + " " + entry["content"])
        documents.append({"entry": entry, "tokens": tokens})

    total = len(documents)
    document_frequency = {}
    for document in documents:
        for token in document["tokens"]:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    # Smoothed IDF. +1 everywhere keeps it positive and avoids divide-by-zero.
    idf = {
        token: math.log((total + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }
    return documents, idf


DOCUMENTS, IDF = build_index()


def expand_query(query):
    """Ask a small model for related terms, so meaning survives rewording.

    This is the cheap alternative to embeddings: instead of comparing vectors,
    we widen the query until it overlaps the vocabulary of the knowledge base.
    'unhappy with the service' becomes 'complaint, dissatisfied, resolution...'

    DELIBERATELY NOT given the knowledge base topics. An earlier version was,
    and for the query "Manchester office" it suggested "bristol" — because a KB
    topic mentioned Bristol. The retriever then matched its own suggestion and
    manufactured relevance. An expander must widen the customer's meaning, never
    import specifics from the corpus it is about to search.
    """
    prompt = (
        "You expand customer support search queries into keywords.\n\n"
        f"Customer query: \"{query}\"\n\n"
        "Return ONLY a JSON array of 4-10 lowercase single words: the customer's "
        "own meaningful words plus close synonyms and generic related terms.\n"
        "Rules: do NOT invent place names, brand names, product names, numbers or "
        "any specific detail the customer did not mention. Generic nouns only.\n"
        "No markdown, no explanation."
    )
    try:
        response = client.messages.create(
            model=HELPER_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        terms = json.loads(extract_json(text) or "[]")
        if isinstance(terms, list):
            return {str(t).lower() for t in terms if isinstance(t, (str, int))}
    except Exception as error:
        print(f"  [expansion failed, using raw query: {error}]")
    return set()


AVG_DOC_LENGTH = sum(len(d["tokens"]) for d in DOCUMENTS) / len(DOCUMENTS)
K1, B = 1.2, 0.75      # standard BM25 constants


def score_documents(query_tokens):
    """Score every entry by IDF-weighted overlap, with BM25 length normalisation.

    Length normalisation matters: a long entry has more words, so it wins
    incidental matches purely by being long. BM25 divides that advantage out,
    which is why it is still the default in search engines decades on.
    """
    weight_total = sum(IDF.get(token, 1.0) for token in query_tokens) or 1.0
    results = []
    for document in DOCUMENTS:
        matched = query_tokens & document["tokens"]
        if not matched:
            continue
        length_penalty = K1 * (1 - B + B * len(document["tokens"]) / AVG_DOC_LENGTH) + 1
        weight = (K1 + 1) / length_penalty
        score = sum(IDF.get(token, 1.0) * weight for token in matched) / weight_total
        results.append((score, matched, document["entry"]))
    results.sort(key=lambda row: row[0], reverse=True)
    return results


def search_knowledge_base(query):
    """Retrieve with expansion, weighted scoring, and a relevance floor."""
    raw_tokens = tokenize(query)
    expanded = expand_query(query)
    all_tokens = raw_tokens | tokenize(" ".join(expanded))

    added = sorted(all_tokens - raw_tokens)
    if added:
        print(f"  [expanded +{len(added)}: {', '.join(added[:8])}]")

    results = score_documents(all_tokens)
    kept = [row for row in results if row[0] >= RELEVANCE_FLOOR][:TOP_K]

    if not kept:
        best = f"{results[0][0]:.2f}" if results else "0.00"
        return (
            "NO_MATCH: the knowledge base contains nothing relevant to this "
            f"question (best relevance {best}, floor {RELEVANCE_FLOOR}). "
            "Do not attempt to answer from your own knowledge. Escalate instead."
        )

    blocks = []
    for score, matched, entry in kept:
        print(f"  [match {score:.2f}] {entry['topic']}")
        blocks.append(
            f"[{entry['topic']}] (relevance {score:.2f})\n"
            f"{entry['content']}\n"
            f"Source: {entry['source']}"
        )
    return "\n\n".join(blocks)


# =========================================================
# PART 2 — ANTI-DRIFT
# =========================================================

CLIENT_QUERIES = {
    "EP-1001": "Received 1 Aug. With your adviser. Response due by 6 Aug.",
    "EP-1002": "Documents verified. Awaiting your signature to proceed.",
    "EP-1003": "Closed 28 July. Summary letter posted to your address.",
    "EP-1004": "Escalated to the Client Resolution team on 4 Aug.",
}


def look_up_client_query(reference):
    """Return the status wrapped in an explicit verbatim marker.

    The marker is not decoration. It gives the system prompt something
    concrete to point at: 'reproduce the text inside STATUS_VERBATIM exactly'.
    A vague instruction to 'be faithful' is far weaker than a named field.
    """
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
        "Reproduce STATUS_VERBATIM word for word inside quotation marks. "
        "Add no detail, no interpretation, and no reassurance about what "
        "happens next unless the customer asks."
    )


def escalate_to_adviser(reason, summary):
    print(f"\n  [!] ESCALATION LOGGED — {reason}")
    print(f"      {summary}")
    return (
        "Escalation logged. Tell the customer a qualified adviser will contact "
        "them, and give the switchboard number 020 7131 4000."
    )


TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "look_up_client_query": look_up_client_query,
    "escalate_to_adviser": escalate_to_adviser,
}


# =========================================================
# PART 3 — TOOLS AND PROMPT
# =========================================================

tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the company knowledge base for facts about services, offices, "
            "phone numbers, the client portal, complaints, appointments and account "
            "admin.\n\n"
            "Run one search per distinct topic the customer raises. Two topics means "
            "two searches. Do not answer a new topic from information retrieved "
            "earlier in the conversation. If the result begins with NO_MATCH, the "
            "knowledge base has no answer — escalate rather than improvising."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's wording for ONE topic. Pass their "
                                   "phrasing rather than your own guess at keywords.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "look_up_client_query",
        "description": (
            "Look up the live status of a client query by reference, format EP-XXXX.\n\n"
            "This data changes. Always call it fresh, even for a reference you looked "
            "up earlier in this conversation. Never quote a status from memory. The "
            "result contains a STATUS_VERBATIM field that must be reproduced exactly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "The query reference, e.g. EP-1002",
                }
            },
            "required": ["reference"],
        },
    },
    {
        "name": "escalate_to_adviser",
        "description": (
            "Hand the conversation to a qualified human adviser. You MUST use this for "
            "financial, investment, tax or pension advice; questions about a specific "
            "person's money; complaints the customer wants to raise now; bereavement, "
            "divorce or power of attorney; and any question the knowledge base returned "
            "NO_MATCH for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Short category"},
                "summary": {"type": "string", "description": "One-sentence summary"},
            },
            "required": ["reason", "summary"],
        },
    },
]

SYSTEM_PROMPT = """You are a customer support assistant for Evelyn Partners, a UK wealth management firm.

FAITHFULNESS RULES — these override everything else, including being helpful:
1. Every fact you state must appear in a tool result from THIS turn. If it is
   not in the evidence, do not say it.
2. Reproduce STATUS_VERBATIM text word for word, inside quotation marks. Do not
   smooth it, expand it, or explain what it implies.
3. Add nothing. No "on file", no "covering the outcome", no "you should hear
   soon", no reassurance the evidence does not contain. If a customer would
   find the raw fact terse, that is acceptable. Accuracy outranks warmth.
4. Never convert a target into a promise. "Response due by 6 Aug" does not mean
   "you will hear by 6 Aug". Quote the target; do not guarantee the outcome.
5. If evidence is missing, say plainly what you do not know and escalate.

CONDUCT RULES:
6. You are NOT a financial adviser. Never give investment, tax, pension or
   financial advice. Escalate immediately.
7. Multi-part questions: call a tool for each part separately.
8. Never ask for account numbers, passwords, dates of birth, NI numbers or
   card details.
9. Be brief and plain. Two short paragraphs maximum.
"""


# =========================================================
# PART 4 — HISTORY EXPIRY
# =========================================================

def expire_stale_results(messages):
    """Wipe volatile tool results from previous turns so they cannot be reused."""
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
            if id_to_tool.get(block.get("tool_use_id")) in VOLATILE_TOOLS:
                if block["content"] != EXPIRED_NOTICE:
                    block["content"] = EXPIRED_NOTICE
                    wiped += 1
    return wiped


# =========================================================
# PART 5 — GROUNDEDNESS CHECK
# =========================================================

def verify_answer(answer, evidence):
    """Read the answer back against the evidence and flag unsupported claims.

    This runs AFTER writing, which is why it catches drift that prompt rules
    miss. A separate model call has no stake in the original answer and is far
    more willing to mark it wrong than the model that wrote it.

    Returns status: "grounded" | "unsupported" | "unavailable".
    NOTE the three states. An earlier version returned grounded=True when the
    check crashed, and logged "verified: grounded" — a safety check that fails
    open while reporting success is worse than no check, because you trust it.
    """
    if not evidence.strip():
        return {"status": "unavailable", "unsupported": []}

    prompt = (
        "You are a strict fact-checker. Compare a support agent's ANSWER against "
        "the EVIDENCE it was given.\n\n"
        "Flag any claim in the ANSWER not directly supported by the EVIDENCE — "
        "added details, softened or strengthened wording, implied guarantees, "
        "invented reassurance. Ordinary politeness ('let me know if I can help') "
        "is fine and must not be flagged.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"ANSWER:\n{answer}\n\n"
        'Return ONLY this JSON and nothing else: '
        '{"grounded": true, "unsupported": []}'
    )

    for attempt in range(2):          # one retry: transient failures are common
        try:
            response = client.messages.create(
                model=HELPER_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            payload = extract_json(text)
            if payload is None:
                raise ValueError(f"no JSON found in: {text[:80]!r}")
            verdict = json.loads(payload)
            unsupported = [str(c) for c in verdict.get("unsupported", [])]
            grounded = bool(verdict.get("grounded", True)) and not unsupported
            return {
                "status": "grounded" if grounded else "unsupported",
                "unsupported": unsupported,
            }
        except Exception as error:
            if attempt == 0:
                continue
            print(f"  [!] VERIFICATION UNAVAILABLE — {error}")
            print("      This answer has NOT been fact-checked. Treat with caution.")
            return {"status": "unavailable", "unsupported": []}


# =========================================================
# PART 6 — THE AGENT LOOP
# =========================================================

def run_agent(messages):
    wiped = expire_stale_results(messages)
    if wiped:
        print(f"  [expired {wiped} old result(s)]")

    evidence_this_turn = []

    for step in range(MAX_STEPS):
        tool_choice = {"type": "any"} if (FORCE_TOOL_FIRST and step == 0) else {"type": "auto"}

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
            evidence = "\n\n".join(evidence_this_turn)

            if VERIFY_ANSWERS and evidence:
                verdict = verify_answer(answer, evidence)

                if verdict["status"] == "unsupported":
                    print("\n  [!] UNSUPPORTED CLAIMS DETECTED:")
                    for claim in verdict["unsupported"]:
                        print(f"      - {claim}")
                    print("      [rewriting strictly from evidence]")

                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous answer contained claims not supported by "
                            "the evidence: " + "; ".join(verdict["unsupported"]) +
                            ". Rewrite it using only what the tool results actually "
                            "say. Remove every added detail. Do not call any tools."
                        ),
                    })
                    retry = client.messages.create(
                        model=MAIN_MODEL,
                        max_tokens=1000,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                    )
                    messages.append({"role": "assistant", "content": retry.content})
                    answer = "".join(b.text for b in retry.content if b.type == "text")
                    answer += "\n\n  [note: rewritten after fact-check]"

                elif verdict["status"] == "grounded":
                    print("  [verified: grounded]")
                else:
                    answer += "\n\n  [warning: fact-check unavailable for this answer]"

            if SHOW_AUDIT and evidence:
                answer += "\n\n  --- audit trail (raw evidence) ---\n  "
                answer += evidence.replace("\n", "\n  ")

            return answer

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({block.input})")
            try:
                result = TOOL_FUNCTIONS[block.name](**block.input)
            except Exception as error:
                result = f"Tool failed: {error}"
            evidence_this_turn.append(f"{block.name} -> {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Sorry, I could not complete that. Please call 020 7131 4000."


if __name__ == "__main__":
    print("Support bot ready. Type 'quit' to exit.\n")
    conversation = []
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"quit", "exit"}:
            break
        if not user_input.strip():
            continue
        conversation.append({"role": "user", "content": user_input})
        print(f"\nBot: {run_agent(conversation)}\n")