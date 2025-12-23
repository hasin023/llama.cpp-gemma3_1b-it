import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Config - Single server with parallel slots handles all requests
ENDPOINTS = ["http://localhost:8080/v1"]
API_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-required")

def build_prompt(context, questions, conversation_turns):
    prompt = """<start_of_turn>user
You are a polite Bangla phone survey agent. ALWAYS respond in Bangla.
ONLY ask the questions from the Question List one by one, exactly as written, in order.
IF user queries about anything, respond from the Survey Context, then continue with the next question.

Survey Context:
""" + context + """

Question List:
""" + "\n".join(["• " + q for q in questions]) + """
<end_of_turn>
"""
    
    # Add conversation turns
    for turn in conversation_turns:
        role = turn["role"]
        content = turn["content"]
        prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"
    
    # End with model turn to generate next response
    prompt += "<start_of_turn>model\n"
    
    return prompt

def run_inference(endpoint, context, questions, conversation_turns):
    client = OpenAI(base_url=endpoint, api_key=API_KEY)
    prompt_text = build_prompt(context, questions, conversation_turns)
    
    start_time = time.time()
    
    response = client.completions.create(
        model="gemma3_1b_400S_p77s16v20-Q8_0",  # Arbitrary
        prompt=prompt_text,
        max_tokens=512,     # Limit for speed
        stream=False        # Sync for simple timing
    )
    
    end_time = time.time()
    inference_time = end_time - start_time
    generated_text = response.choices[0].text
    
    print(f"Endpoint: {endpoint}")
    print(f"Generated Response: {generated_text}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    
    return generated_text, inference_time

if __name__ == "__main__":
    survey_context = "ছাত্রজীবন মানে হচ্ছে, যখন আমরা জীবিত থাকি, শেখার জন্য উন্মুক্ত থাকি এবং প্রতিদিন নতুন কিছু আবিষ্কার করি। এটি আমাদের জীবনের সবচেয়ে মূল্যবান সময়।"
    question_list = [
        "তুমি যদি হঠাৎ একদিন অদৃশ্য হয়ে যেতে পারো, প্রথমে কী করতে চাইবে?",
        "জীবনের কোন ছোট অভ্যাসটা তোমার মতে সবচেয়ে বেশি পরিবর্তন আনতে পারে?",
        "যদি আবার ছাত্রজীবনে ফিরে যাওয়ার সুযোগ পাও, কোন সিদ্ধান্তটা বদলাতে চাইতে?",
    ]
    conversation_turns = [
        {"role": "model", "content": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?"},
        {"role": "user", "content": "জি বলছি"},
        {"role": "model", "content": "তুমি যদি হঠাৎ একদিন অদৃশ্য হয়ে যেতে পারো, প্রথমে কী করতে চাইবে?"},
        {"role": "user", "content": "আমি প্রথমে আমার বন্ধুদের সাথে মজা করতে চাইব।"},
    ]
    
    run_inference(ENDPOINTS[0], survey_context, question_list, conversation_turns)  # Test first; loop for others