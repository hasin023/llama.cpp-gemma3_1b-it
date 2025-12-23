from locust import HttpUser, task, between

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

SURVEY_CONTEXT = "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, Unimart, Shawpno এর মতো অনলাইন শপ ও সুপারশপ থেকে।"

QUESTION_LIST = [
    "আপনি কি FrozenBerry ব্যবহার করেছেন?",
    "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
    "সংরক্ষণ করা সহজ কি?"
]

CONVERSATION_TURNS = [
    {"role": "model", "content": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?"},
    {"role": "user", "content": "জি বলছি"},
    {"role": "model", "content": "আপনি কি FrozenBerry ব্যবহার করেছেন?"},
    {"role": "user", "content": "হ্যাঁ, ব্যবহার করেছি।"},
]

PROMPT = build_prompt(
    SURVEY_CONTEXT,
    QUESTION_LIST,
    CONVERSATION_TURNS
)

PAYLOAD = {
    "model": "gemma3_1b_400S_p77s16v20-Q8_0",
    "prompt": PROMPT,
    "max_tokens": 512,
    "stream": False
}

class LLMUser(HttpUser):
    wait_time = between(0, 0)  # fire immediately
    host = "http://localhost:8080"  # overridden per task

    @task
    def hit_8080(self):
        self.client.post(
            "/v1/completions",
            json=PAYLOAD,
            headers={"Authorization": "Bearer sk-no-key-required"},
            name="endpoint_8080"
        )

    @task
    def hit_8081(self):
        self.client.post(
            "http://localhost:8081/v1/completions",
            json=PAYLOAD,
            headers={"Authorization": "Bearer sk-no-key-required"},
            name="endpoint_8081"
        )
