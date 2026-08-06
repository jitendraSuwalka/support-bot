"""
Step 7 — forcing fresh tool calls.

Three layers that stop the agent reusing stale answers:
  Layer 1 (prompt) : ask it nicely            — helps, not guaranteed
  Layer 2 (API)    : tool_choice="any"        — forces a tool on the first step
  Layer 3 (code)   : expire old tool results  — it CANNOT reuse what isn't there

Personal learning project. Not affiliated with Evelyn Partners.
"""

from dotenv import load_dotenv
from anthropic import Anthropic
from knowledge_base import KNOWLEDGE_BASE

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5"
MAX_STEPS = 8

# Tools whose answers go out of date. Results from these are wiped
# from the history once the turn is over, so the model must re-check.
VOLATILE_TOOLS = {"look_up_client_query"}

EXPIRED_NOTICE = (
    "[This earlier result has expired and may no longer be accurate. "
    "Call the tool again if you need this information.]"
)

# Force at least one tool call on the first step of every turn.
# Trade-off: even "hello" will trigger a search. See notes.
FORCE_TOOL_FIRST = True


# =========================================================
# TOOLS
# =========================================================

CLIENT_QUERIES = {
    "EP-1001": "Received 1 Aug. With your adviser. Response due by 6 Aug.",
    "EP-1002": "Documents verified. Awaiting your signature to proceed.",
    "EP-1003": "Closed 28 July. Summary letter posted to your address.",
    "EP-1004": "Escalated to the Client Resolution team on 4 Aug.",
}


def search_knowledge_base(query):
    stopwords = {"the", "a", "an", "is", "to", "of", "and", "i", "my",
                 "how", "do", "what", "can", "you", "for", "in", "it"}
    words = {w.strip("?.,!'").lower() for w in query.split()} - stopwords

    scored = []
    for entry in KNOWLEDGE_BASE:
        haystack = (entry["topic"] + " " + entry["content"]).lower()
        score = sum(1 for w in words if w and w in haystack)
        if score > 0:
            scored.append((score, entry))

    if not scored:
        return "No matching information found in the knowledge base."

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n\n".join(
        f"[{e['topic']}]\n{e['content']}\nSource: {e['source']}"
        for _, e in scored[:2]
    )


def look_up_client_query(reference):
    return CLIENT_QUERIES.get(
        reference.upper().strip(),
        "No query found with that reference. Please check the number."
    )


def escalate_to_adviser(reason, summary):
    print(f"\n  [!] ESCALATION LOGGED — reason: {reason}")
    print(f"      summary: {summary}")
    return (
        "Escalation logged. Tell the customer a qualified adviser will contact "
        "them, and give them the switchboard number 020 7131 4000."
    )


TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "look_up_client_query": look_up_client_query,
    "escalate_to_adviser": escalate_to_adviser,
}


# =========================================================
# LAYER 1 — wording
# =========================================================

tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the company knowledge base for facts about services, offices, "
            "phone numbers, the client portal, complaints, appointments and account "
            "admin.\n\n"
            "IMPORTANT: run one search per distinct topic the customer raises. If "
            "they ask about two different things, search twice — once for each. Do "
            "not answer a new topic from information retrieved earlier in the "
            "conversation, because that information was retrieved for a different "
            "question and may be incomplete for this one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Key terms for ONE topic only, e.g. 'complaint timescale'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "look_up_client_query",
        "description": (
            "Look up the live status of a client query by reference, format EP-XXXX.\n\n"
            "IMPORTANT: this data changes. Always call this tool fresh every time a "
            "customer asks about a reference, even if you looked up the same "
            "reference earlier in this conversation. Never quote a status from "
            "memory or from an earlier message."
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
            "Hand the conversation to a qualified human adviser. You MUST use this "
            "for financial, investment, tax or pension advice; questions about a "
            "specific person's money; complaints the customer wants to raise now; "
            "bereavement, divorce or power of attorney; or anything the knowledge "
            "base cannot answer."
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

STRICT RULES:
1. You are NOT a financial adviser. Never give investment, tax, pension or
   financial advice. Escalate immediately instead.
2. Never state a fact unless a tool returned it IN THIS TURN. Never guess
   phone numbers, addresses, fees, timescales or policies.
3. MULTI-PART QUESTIONS: if the customer asks about more than one thing, deal
   with each part separately and call a tool for each part. Two questions means
   at least two tool calls. Do not skip a search because you think you already
   know the answer.
4. Query statuses change. Always look up a reference fresh. Never reuse a status
   from earlier in the conversation.
5. Never ask for account numbers, passwords, dates of birth, NI numbers or
   card details.
6. Be warm and brief. Two short paragraphs maximum.
"""


# =========================================================
# LAYER 3 — expire stale results from history
# =========================================================

def expire_stale_results(messages):
    """Blank out results from volatile tools in previous turns.

    The model can only reuse what it can see. Wipe the data and reuse
    becomes impossible rather than merely discouraged.
    """
    # Which tool produced which result? Map id -> tool name.
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
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_name = id_to_tool.get(block.get("tool_use_id"))
            if tool_name in VOLATILE_TOOLS and block["content"] != EXPIRED_NOTICE:
                block["content"] = EXPIRED_NOTICE
                wiped += 1
    return wiped


# =========================================================
# THE AGENT LOOP
# =========================================================

def run_agent(messages):
    # Everything from previous turns is now potentially stale.
    wiped = expire_stale_results(messages)
    if wiped:
        print(f"  [expired {wiped} old result(s) — bot must re-check]")

    for step in range(MAX_STEPS):
        # LAYER 2: force a tool on the first step only.
        # Leaving it on for later steps would loop forever, because the model
        # would never be allowed to stop and answer.
        if FORCE_TOOL_FIRST and step == 0:
            tool_choice = {"type": "any"}
        else:
            tool_choice = {"type": "auto"}

        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({block.input})")
            try:
                result = TOOL_FUNCTIONS[block.name](**block.input)
            except Exception as error:
                result = f"Tool failed: {error}"
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