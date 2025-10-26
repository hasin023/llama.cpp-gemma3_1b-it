from openai import OpenAI

# Configure client to use your local llama.cpp endpoint
client = OpenAI(
    base_url="http://localhost:5050/v1",
    api_key="not-needed"  # llama.cpp doesn't require a real API key
)

response = client.chat.completions.create(
    model="gemma-3-1b-it",
    messages=[
        {"role": "user", "content": "What is a butterfly?"}
    ],
)

print(response.choices[0].message.content)
