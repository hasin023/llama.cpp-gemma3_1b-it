import time
import requests
import logging
from pathlib import Path

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "llm_service_api.log"

# Configure logging with both file and console handlers
logger = logging.getLogger("llm-client")
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers.clear()

# File handler - write to logs/llm-service-api.log
file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a",  # append mode
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler - also output to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


BASE_URL = "http://localhost:8000"

def log_conversation(conversation, title="Conversation"):
    logger.info("%s (%d turns)", title, len(conversation))
    for i, turn in enumerate(conversation, 1):
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        role_label = "M" if role == "model" else "U"
        logger.info("%02d. [%s] %s", i, role_label, content)


def log_response(label, data):
    logger.info("%s", label)
    logger.info("  model: %s", data.get("generated_text", "").strip())
    logger.info("  time : %.2fs", data.get("inference_time", 0))


def run_sequence():
    s = requests.Session()

    h = s.get(f"{BASE_URL}/health")
    assert h.status_code == 200
    logger.info("Service healthy")

    survey_context = (
        "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, "
        "Unimart, Shawpno এর মতো অনলাইন শপ ও সুপারশপ থেকে।"
    )

    questions = [
        "আপনি কি FrozenBerry ব্যবহার করেছেন?",
        "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
        "সংরক্ষণ করা সহজ কি?"
    ]

    final_conversation = []

    # ---- Start total inference timer ----
    total_start_time = time.perf_counter()

    # ---- First request ----
    initial_model_message = "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?"
    user_query_1 = "জি বলছি।"

    logger.info("Init model: %s", initial_model_message)
    logger.info("User input: %s", user_query_1)

    r1 = s.post(
        f"{BASE_URL}/query",
        json={
            "initial_model_message": initial_model_message,
            "survey_context": survey_context,
            "questions": questions,
            "user_query": user_query_1,
            "continue_in_same_conversation": True
        },
        timeout=120
    )
    assert r1.status_code == 200
    data1 = r1.json()

    log_response("Response #1", data1)
    final_conversation = data1.get("conversation", [])

    time.sleep(0.5)

    # ---- Second request ----
    user_query_2 = "জি ব্যবহার করেছি।"
    logger.info("User input: %s", user_query_2)

    r2 = s.post(
        f"{BASE_URL}/query",
        json={
            "survey_context": survey_context,
            "questions": questions,
            "user_query": user_query_2,
            "continue_in_same_conversation": True
        },
        timeout=120
    )
    assert r2.status_code == 200
    data2 = r2.json()

    log_response("Response #2", data2)
    final_conversation = data2.get("conversation", [])

    # ---- End total inference timer ----
    total_inference_time = time.perf_counter() - total_start_time

    # ---- Final output ----
    log_conversation(final_conversation, "Final Conversation")
    logger.info("Total inference time (end-to-end): %.2fs", total_inference_time)


if __name__ == "__main__":
    run_sequence()
