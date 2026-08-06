import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "Explain what an AI agent is in two sentences."}
    ],
)

print(response.content[0].text)