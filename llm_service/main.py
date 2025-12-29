import os
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

CONVERSATIONS: Dict[str, List[Dict[str, Any]]] = {}

def build_prompt(context: str, questions: List[str], conversation_turns: List[Dict[str, str]]) -> str:
    prompt = "<start_of_turn>user\nYou are a polite Bangla phone survey agent. ALWAYS respond in Bangla.\nONLY ask the questions from the Question List one by one, exactly as written, in order.\nIF user queries about anything, respond from the Survey Context, then continue with the next question.\n\nSurvey Context:\n" + context + "\n\nQuestion List:\n" + "\n".join(["• " + q for q in questions]) + "\n<end_of_turn>\n"
    for turn in conversation_turns:
        role = turn["role"]
        content = turn["content"]
        prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

class QueryRequest(BaseModel):
    user_query: Optional[str] = None
    initial_model_message: Optional[str] = None
    survey_context: str
    questions: List[str]
    continue_in_same_conversation: bool = True
    model: str = "gemma3_1b_400S_p77s16v20"
    max_tokens: int = 512
    stream: bool = False

class QueryResponse(BaseModel):
    conversation_id: str
    generated_text: str
    inference_time: float
    conversation: List[Dict[str, str]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request, response: Response):
    endpoint = os.getenv("LLAMA_CPP_ENDPOINT", "http://localhost:8080/v1")
    api_key = os.getenv("OPENAI_API_KEY", "sk-no-key-required")
    client = OpenAI(base_url=endpoint, api_key=api_key)

    context = req.survey_context
    questions = req.questions

    if not context or not isinstance(questions, list) or len(questions) == 0:
        raise HTTPException(status_code=400, detail="survey_context and questions are required and must be non-empty")

    cid_cookie = request.cookies.get("conversation_id")
    if req.continue_in_same_conversation and cid_cookie:
        cid = cid_cookie
        if cid not in CONVERSATIONS:
            CONVERSATIONS[cid] = []
    else:
        cid = str(uuid.uuid4())
        CONVERSATIONS[cid] = []
        response.set_cookie(key="conversation_id", value=cid, httponly=False, samesite="lax", path="/")

    if req.initial_model_message and (not req.continue_in_same_conversation or not cid_cookie):
        CONVERSATIONS[cid].append({"role": "model", "content": req.initial_model_message})

    if req.user_query:
        CONVERSATIONS[cid].append({"role": "user", "content": req.user_query})

    prompt_text = build_prompt(context, questions, CONVERSATIONS[cid])

    start_time = time.time()
    response = client.completions.create(
        model=req.model,
        prompt=prompt_text,
        max_tokens=req.max_tokens,
        stream=False
    )
    end_time = time.time()

    generated_text = response.choices[0].text
    CONVERSATIONS[cid].append({"role": "model", "content": generated_text})

    return QueryResponse(
        conversation_id=cid,
        generated_text=generated_text,
        inference_time=end_time - start_time,
        conversation=CONVERSATIONS[cid],
    )

