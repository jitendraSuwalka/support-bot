from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

messages = []                      # the conversation lives here

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    if not user_input.strip():        # ← add this
        continue                      # ← and this

    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        messages=messages,         # send the WHOLE history every time
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    reply = "\n".join(text_blocks)
    print(f"\nBot: {reply}")

    messages.append({"role": "assistant", "content": response.content})