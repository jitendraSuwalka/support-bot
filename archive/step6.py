"""
Step 6 — the agent loop.

A support bot with three tools. The loop keeps running while the model
asks for tools, and stops when it produces a final answer.

Personal learning project. Not affiliated with Evelyn Partners.
"""

from dotenv import load_dotenv
from anthropic import Anthropic
from knowledge_base import KNOWLEDGE_BASE

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5"
MAX_STEPS = 6          # safety net: never loop forever


# =========================================================
# 1. THE TOOLS  (ordinary Python functions)
# =========================================================

# Fake client queries. Invented data — no real client information.
CLIENT_QUERIES = {
    "EP-1001": "Received 1 Aug. With your adviser. Response due by 6 Aug.",
    "EP-1002": "Documents verified. Awaiting your signature to proceed.",
    "EP-1003": "Closed 28 July. Summary letter posted to your address.",
    "EP-1004": "Escalated to the Client Resolution team on 4 Aug.",
}


def search_knowledge_base(query):
    """Find the most relevant FAQ entries by keyword overlap."""
    stopwords = {"the", "a", "an", "is", "to", "of", "and", "i", "my",
                 "how", "do", "what", "can", "you", "for", "in", "it"}
    words = {w.strip("?.,!'").lower() for w in query.split()}
    words -= stopwords

    scored = []
    for entry in KNOWLEDGE_BASE:
        haystack = (entry["topic"] + " " + entry["content"]).lower()
        score = sum(1 for w in words if w and w in haystack)
        if score > 0:
            scored.append((score, entry))

    if not scored:
        return "No matching information found in the knowledge base."

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:2]

    return "\n\n".join(
        f"[{e['topic']}]\n{e['content']}\nSource: {e['source']}"
        for _, e in top
    )


def look_up_client_query(reference):
    """Check the status of an existing query by its reference number."""
    return CLIENT_QUERIES.get(
        reference.upper().strip(),
        "No query found with that reference. Please check the number."
    )


def escalate_to_adviser(reason, summary):
    """Hand off to a human. In a real system this would create a ticket."""
    print(f"\n  [!] ESCALATION LOGGED — reason: {reason}")
    print(f"      summary: {summary}")
    return (
        "Escalation logged. Tell the customer a qualified adviser will contact "
        "them, and give them the switchboard number 020 7131 4000."
    )


# Maps tool names to the actual functions
TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "look_up_client_query": look_up_client_query,
    "escalate_to_adviser": escalate_to_adviser,
}


# =========================================================
# 2. DESCRIBING THE TOOLS TO THE MODEL
# =========================================================

tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the company knowledge base for factual information about "
            "services, offices, phone numbers, the client portal, complaints, "
            "appointments and account admin. Use this for ANY question about how "
            "the company works. Never answer such questions from your own memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Key terms to search for, e.g. 'complaint timescale'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "look_up_client_query",
        "description": (
            "Look up the status of an existing client query using its reference "
            "number, format EP-XXXX. Use whenever a customer asks about progress "
            "on a query they have already raised."
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
            "for anything involving financial, investment, tax or pension advice; "
            "any request about a specific person's money or portfolio; complaints "
            "the customer wants to raise now; bereavement, divorce or power of "
            "attorney; or anything the knowledge base cannot answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short category, e.g. 'investment advice request'",
                },
                "summary": {
                    "type": "string",
                    "description": "One sentence summarising what the customer needs",
                },
            },
            "required": ["reason", "summary"],
        },
    },
]


# =========================================================
# 3. THE GUARDRAILS
# =========================================================

SYSTEM_PROMPT = """You are a customer support assistant for Evelyn Partners, a UK wealth management firm.

STRICT RULES:
1. You are NOT a financial adviser. Never give investment, tax, pension or
   financial advice of any kind, not even general guidance. Any such question
   goes to escalate_to_adviser immediately.
2. Never state a fact about the company unless a tool returned it. If the
   knowledge base has no answer, say so and escalate. Never guess or invent
   phone numbers, addresses, fees, timescales or policies.
3. Never ask for or repeat sensitive details: account numbers, passwords,
   dates of birth, National Insurance numbers or card details.
4. Be warm, brief and clear. Two short paragraphs maximum.
5. If a customer seems distressed, or mentions bereavement, divorce or illness,
   respond with care and escalate to a human.
"""


# =========================================================
# 4. THE AGENT LOOP  — this is the whole idea
# =========================================================

def run_agent(messages):
    """Keep calling tools until the model gives a final answer."""
    for step in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # No tool wanted? Then this is the final answer.
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Otherwise: run every tool it asked for.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"  [tool] {block.name}({block.input})")
            function = TOOL_FUNCTIONS[block.name]

            try:
                result = function(**block.input)
            except Exception as error:
                result = f"Tool failed: {error}"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Sorry, I could not complete that. Please call 020 7131 4000."


# =========================================================
# 5. CHAT
# =========================================================

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
        answer = run_agent(conversation)
        print(f"\nBot: {answer}\n")