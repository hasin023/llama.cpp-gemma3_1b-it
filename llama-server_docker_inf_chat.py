import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Config
ENDPOINTS = ["http://localhost:8080/v1"]
API_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-required")

def run_chat_inference(endpoint, context, questions, conversation_history):
    """
    conversation_history: list of dicts with 'role' and 'content'
                          (only user/model turns, not system)
    """
    client = OpenAI(base_url=endpoint, api_key=API_KEY)

    # System message with full instructions and context
    system_message = {
        "role": "system",
        "content": (
            "You are a polite Bangla phone survey agent. "
            "ALWAYS respond in Bangla. "
            "ONLY ask the questions from the Question List one by one, exactly as written, in order. "
            "If the user asks about anything else, respond using information from the Survey Context, "
            "then continue with the next question.\n\n"
            "Survey Context:\n"
            f"{context}\n\n"
            "Question List:\n"
            + "\n".join([f"• {q}" for q in questions])
        )
    }

    # Build full messages list: system + conversation history
    messages = [system_message] + conversation_history

    start_time = time.time()

    response = client.chat.completions.create(
        model="gemma3_1b_400S_p77s16v20-Q8_0",
        messages=messages,
        max_tokens=512,
        temperature=0.4,            # Lowered for stricter, less creative responses (as discussed)
        stream=False                # Set to True if you want streaming later
    )

    end_time = time.time()
    inference_time = end_time - start_time
    generated_text = response.choices[0].message.content.strip()

    print(f"Endpoint: {endpoint}")
    print(f"Generated Response: {generated_text}")
    print(f"Inference Time: {inference_time:.2f} seconds")

    return generated_text, inference_time


if __name__ == "__main__":
    survey_context = "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, Unimart, Shawpno এর মতো অনলাইন শপ ও সুপারশপ থেকে।"

    question_list = [
        "আপনি কি FrozenBerry ব্যবহার করেছেন?",
        "স্বাদ কেমন লাগলো?",
        "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
        "সংরক্ষণ করা সহজ কি?"
    ]

    # Conversation history (only user and model turns)
    conversation_history = [
        {"role": "model", "content": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?"},
        {"role": "user",  "content": "জি বলছি"},
        {"role": "model", "content": "আপনি কি FrozenBerry ব্যবহার করেছেন?"},
        {"role": "user",  "content": "FrozenBerry কোথা থেকে কিনতে পারবো?"}
    ]

    # Test with first endpoint
    run_chat_inference(ENDPOINTS[0], survey_context, question_list, conversation_history)