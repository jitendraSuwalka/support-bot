from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# --- our fake database -------------------------------------
CLIENT_QUERIES = {
    "EP-1001": "Received — with your adviser, response due 6 Aug.",
    "EP-1002": "Documents verified. Awaiting client signature.",
    "EP-1003": "Closed on 28 July. Summary letter posted.",
}

def look_up_client_query(reference):
    return CLIENT_QUERIES.get(reference, "No query found with that reference.")

# --- describing the tool TO the model ----------------------
tools = [
    {
        "name": "look_up_client_query",
        "description": "Look up the status of a client service query using its reference number, e.g. EP-1002. Use whenever a customer asks about the progress of an existing query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "The query reference, format EP-XXXX",
                }
            },
            "required": ["reference"],
        },
    }
]

# --- one turn ----------------------------------------------
messages = [
    {"role": "user", "content": "Hi, can you tell me what's happening with EP-1002?"}
]

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=500,
    tools=tools,
    messages=messages,
)

print("Stop reason:", response.stop_reason)
for block in response.content:
    print("Block type:", block.type)
    if block.type == "tool_use":
        print("  Wants tool:", block.name)
        print("  With input:", block.input)


# --- handle the tool request -------------------------------
messages.append({"role": "assistant", "content": response.content})

tool_results = []
for block in response.content:
    if block.type == "tool_use":
        result = look_up_client_query(**block.input)   # actually run it
        print("\n>>> Ran the function, got:", result)

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,       # links back to the request
            "content": result,
        })

messages.append({"role": "user", "content": tool_results})

# --- ask again, now that it has the answer -----------------
final = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=500,
    tools=tools,
    messages=messages,
)

print("\nBot:", "".join(b.text for b in final.content if b.type == "text"))