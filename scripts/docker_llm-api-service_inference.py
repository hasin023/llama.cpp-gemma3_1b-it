import time
import requests

BASE_URL = "http://localhost:8000"


def run_sequence():
    """
    Minimal client example for the LLM Service:
    - Uses requests.Session() so conversation_id is tracked automatically via cookies.
    - Prints the raw LLM responses that should be passed to TTS.
    - All detailed logging is handled inside the llm_service itself.
    """
    s = requests.Session()

    # Health check
    h = s.get(f"{BASE_URL}/health")
    h.raise_for_status()
    print("Service healthy:", h.json())

    survey_context = (
        "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, "
        "Unimart, Shawpno এর মতো অনলাইন শপ ও সুপারশপ থেকে।"
    )

    questions = [
        "আপনি কি FrozenBerry ব্যবহার করেছেন?",
        "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
        "সংরক্ষণ করা সহজ কি?",
    ]

    # ---- Start total inference timer ----
    total_start_time = time.perf_counter()

    # ---- First request ----
    initial_model_message = "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?"
    # This simulates the STT text we get from the STT service
    user_query_1 = "জি বলছি।"

    r1 = s.post(
        f"{BASE_URL}/query",
        json={
            "initial_model_message": initial_model_message,
            "survey_context": survey_context,
            "questions": questions,
            "user_query": user_query_1,
            "continue_in_same_conversation": True,
        },
        timeout=120,
    )
    r1.raise_for_status()
    data1 = r1.json()

    # This is the raw text from the LLM Service to send to TTS
    generated_text_1 = data1["generated_text"]
    print("Turn #1 - LLM text response:", generated_text_1)
    print("Turn #1 - conversation_id:", data1.get("conversation_id"))

    time.sleep(0.5)

    # ---- Second request ----
    # Session automatically sends back the conversation_id cookie; no manual ID needed.
    user_query_2 = "জি ব্যবহার করেছি।"

    r2 = s.post(
        f"{BASE_URL}/query",
        json={
            "survey_context": survey_context,
            "questions": questions,
            "user_query": user_query_2,
            "continue_in_same_conversation": True,
        },
        timeout=120,
    )
    r2.raise_for_status()
    data2 = r2.json()

    generated_text_2 = data2["generated_text"]
    print("Turn #2 - LLM text response:", generated_text_2)
    print("Turn #2 - conversation_id:", data2.get("conversation_id"))

    # ---- End total inference timer ----
    total_inference_time = time.perf_counter() - total_start_time
    print(f"Total inference time (end-to-end): {total_inference_time:.2f}s")


if __name__ == "__main__":
    run_sequence()
