"""
LLM Service
--------------------------

This FastAPI service wraps a running llama.cpp HTTP server and exposes a minimal
OpenAI-style `/query` endpoint tailored for a *voice-to-voice* survey agent:

STT (speech-to-text) → LLM Service (this app) → TTS (text-to-speech)

Responsibilities:
- Build the Gemma-3 survey prompt exactly as used during fine-tuning.
- Maintain turn-by-turn conversation history (user/model turns).
- Support long-running conversations via `conversation_id` or HTTP cookies.
"""

import os
import time
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (parent of llm_service)
BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load .env (for OPENAI_API_KEY, LLAMA_CPP_ENDPOINT, etc.)
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI()

# -----------------------------------------------------------------------------
# Logging configuration (service-side)
# -----------------------------------------------------------------------------
LOG_PATH = Path(os.getenv("LLM_SERVICE_LOG_FILE", "/logs/llm_service_api.log"))
try:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    # In case /logs isn't writable, fall back to stdout-only logging
    pass

logger = logging.getLogger("llm-service")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # File handler (if path is writable)
    try:
        file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If file handler fails, we still want console logs
        pass

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# In-memory store of conversations:
#   { conversation_id: [ { "role": "user"|"model", "content": "..." }, ... ] }
CONVERSATIONS: Dict[str, List[Dict[str, Any]]] = {}


def build_prompt(context: str, questions: List[str], conversation_turns: List[Dict[str, str]]) -> str:
    """
    Build the Gemma-3 prompt with:
    - A fixed system-style user block describing the survey agent behavior.
    - Survey Context and Question List (exactly as provided).
    - All prior conversation turns in order, using <start_of_turn>/<end_of_turn> tags.
    """
    prompt = (
        "<start_of_turn>user\n"
        "You are a polite Bangla phone survey agent. ALWAYS respond in Bangla.\n"
        "ONLY ask the questions from the Question List one by one, exactly as written, in order.\n"
        "IF user queries about anything, respond from the Survey Context, then continue with the next question.\n\n"
        "Survey Context:\n"
        + context
        + "\n\nQuestion List:\n"
        + "\n".join(["• " + q for q in questions])
        + "\n<end_of_turn>\n"
    )

    for turn in conversation_turns:
        role = turn["role"]
        content = turn["content"]
        prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"

    # The model should generate the next `model` turn
    prompt += "<start_of_turn>model\n"
    return prompt


class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    Typical usage in a voice agent loop:
    - STT sends latest `user_query` along with the same `survey_context` and `questions`.
    - The client keeps passing back `conversation_id` (or relies on cookies) while
      `continue_in_same_conversation` stays True.
    """

    user_query: Optional[str] = None
    initial_model_message: Optional[str] = None
    survey_context: str
    questions: List[str]
    continue_in_same_conversation: bool = True
    conversation_id: Optional[str] = None
    model: str = "gemma3_1b_400S_p77s16v20"
    max_tokens: int = 512
    stream: bool = False


class QueryResponse(BaseModel):
    """
    Response body for POST /query.

    `conversation` is the full list of user/model turns for this conversation_id.
    Voice clients typically care about:
    - `generated_text`: send to TTS
    - `conversation_id`: store and reuse on the next turn
    """

    conversation_id: str
    generated_text: str
    inference_time: float
    conversation: List[Dict[str, str]]
    # Add model name for tracking what model is actually running
    model_name: Optional[str] = None
    usage: Optional[Dict[str, int]] = None



@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health check used by Docker and test clients."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request, response: Response) -> QueryResponse:
    """
    Main inference endpoint.

    Conversation continuity rules:
    - If `continue_in_same_conversation` is False → always start a new conversation.
    - Otherwise:
      - Prefer explicit `req.conversation_id` if provided.
      - Else fall back to `conversation_id` cookie if present.
      - If neither exist → start a new conversation.

    In all cases, the effective `conversation_id` is returned in the response and
    also written to an HTTP cookie for browser-based clients.
    """
    endpoint = os.getenv("LLAMA_CPP_ENDPOINT", "http://localhost:8080/v1")
    api_key = os.getenv("OPENAI_API_KEY", "sk-no-key-required")
    client = OpenAI(base_url=endpoint, api_key=api_key)

    context = req.survey_context
    questions = req.questions

    if not context or not isinstance(questions, list) or len(questions) == 0:
        raise HTTPException(
            status_code=400,
            detail="survey_context and questions are required and must be non-empty",
        )

    logger.info(
        "Incoming /query | continue=%s | explicit_cid=%s | cookie_cid=%s",
        req.continue_in_same_conversation,
        req.conversation_id,
        request.cookies.get("conversation_id"),
    )

    cid_cookie = request.cookies.get("conversation_id")
    cid_body = req.conversation_id

    # Decide which conversation_id to use
    cid: str
    is_new_conversation = False

    if not req.continue_in_same_conversation:
        # Always start a new conversation
        cid = str(uuid.uuid4())
        is_new_conversation = True
    else:
        # Prefer explicit body value, then cookie, else new
        if cid_body:
            cid = cid_body
        elif cid_cookie:
            cid = cid_cookie
        else:
            cid = str(uuid.uuid4())
            is_new_conversation = True

    if cid not in CONVERSATIONS:
        CONVERSATIONS[cid] = []
        # Treat unknown id as a new conversation for initial prompts
        if not is_new_conversation:
            is_new_conversation = True

    # Always (re)set the cookie so browser clients don't need to manage IDs manually
    response.set_cookie(
        key="conversation_id",
        value=cid,
        httponly=False,
        samesite="lax",
        path="/",
    )

    # Seed the conversation with an initial model greeting *once*, at the start
    if req.initial_model_message and is_new_conversation:
        CONVERSATIONS[cid].append({"role": "model", "content": req.initial_model_message})

    # Append the latest user turn, if any (voice agent's STT output)
    if req.user_query:
        CONVERSATIONS[cid].append({"role": "user", "content": req.user_query})

    logger.info(
        "Conversation %s | new=%s | turns_before_inference=%d",
        cid,
        is_new_conversation,
        len(CONVERSATIONS[cid]),
    )

    prompt_text = build_prompt(context, questions, CONVERSATIONS[cid])

    start_time = time.time()
    completion = client.completions.create(
        model=req.model,
        prompt=prompt_text,
        max_tokens=req.max_tokens,
        stream=False,
    )
    end_time = time.time()

    generated_text = completion.choices[0].text
    CONVERSATIONS[cid].append({"role": "model", "content": generated_text})

    usage = completion.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    logger.info(
        "Conversation %s | prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | inference_time=%.2fs",
        cid,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        end_time - start_time,
    )

    # Log the full conversation in a compact format for debugging
    logger.info("Conversation %s | total_turns=%d", cid, len(CONVERSATIONS[cid]))
    for idx, turn in enumerate(CONVERSATIONS[cid], start=1):
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        role_label = "M" if role == "model" else "U"
        logger.info("  %02d. [%s] %s", idx, role_label, content)

    return QueryResponse(
        conversation_id=cid,
        generated_text=generated_text,
        inference_time=end_time - start_time,
        conversation=CONVERSATIONS[cid],
        model_name=completion.model,  # Capture actual model used
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    )
