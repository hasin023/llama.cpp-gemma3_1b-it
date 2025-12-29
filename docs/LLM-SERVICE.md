# LLM Service (Survey Agent) — Setup & Voice Integration

This service wraps your running `llama.cpp` server and exposes a minimal HTTP API to:
- Build the Gemma-3 survey prompt (kept exactly as in your fine-tuning).
- Manage conversation turns.
- Run inference via the OpenAI-compatible endpoint of `llama.cpp`.

It is designed to plug into a voice pipeline: STT → LLM → TTS.

## Prerequisites
- `docker` and `docker compose`
- `OPENAI_API_KEY` (string; for llama.cpp you may use any non-empty value)
- `LLAMA_ARG_*` configured in `compose.yaml` for the llama.cpp server
- The fine-tuned Gemma-3 1B GGUF model available via HF or local

## Start the System
1. Configure model source in `compose.yaml` for `llm-server` (HF repo/file or local `/models`).
2. Start services:
   - `docker compose up -d`
3. Verify:
   - LLM llama.cpp server health: `curl http://localhost:8080/health`
   - LLM service health: `curl http://localhost:8000/health`

## API (LLM Service)
Base URL: `http://localhost:8000`

### POST /query
Runs inference with the Gemma survey prompt and conversation control.

Request (JSON):
```
{
  "survey_context": "string, required",
  "questions": ["string", "string", "..."],  // required, 1+ items
  "user_query": "string, optional",
  "conversation_id": "string, optional",
  "continue_in_same_conversation": true,     // default true
  "model": "gemma3_1b_400S_p77s16v20",
  "max_tokens": 512,
  "stream": false
}
```

Notes:
- `survey_context` and `questions` are mandatory. The service will return 400 if missing/empty.
- `conversation_id` controls continuity. If omitted:
  - `continue_in_same_conversation=true` starts a new conversation and returns its ID.
  - `continue_in_same_conversation=false` always starts a new conversation.
- `user_query` may be omitted to let the model produce the next survey question from the prompt.

Response (JSON):
```
{
  "conversation_id": "string",
  "generated_text": "string",
  "inference_time": 1.23
}
```

### Prompt Format
The service uses the same Gemma-3 turn-based prompt you fine-tuned on:
- Preserves `<start_of_turn>user ... <end_of_turn>` and `<start_of_turn>model`.
- Appends conversation turns in order.
- Does not alter the system prompt instructions.

## Voice Agent Integration
Pipeline: STT → LLM Service → TTS

1. STT captures user audio and produces `user_query` text.
2. Call `POST /query` with:
   - `survey_context` (Bengali or any language per your use-case),
   - `questions` (ordered list used verbatim),
   - `user_query` (latest transcribed text).
3. Receive `generated_text` and pass to TTS for playback.
4. Repeat per turn with the returned `conversation_id`.

Example turn:
```
POST http://localhost:8000/query
Content-Type: application/json

{
  "survey_context": "বাংলাদেশে FrozenBerry ...",
  "questions": [
    "আপনি কি FrozenBerry ব্যবহার করেছেন?",
    "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
    "সংরক্ষণ করা সহজ কি?"
  ],
  "continue_in_same_conversation": true
}
```
Response:
```
{
  "conversation_id": "c6a8f4c1-...",
  "generated_text": "আমি একটি জরিপ কোম্পানি থেকে বলছি...",
  "inference_time": 0.87
}
```
Next user turn:
```
POST http://localhost:8000/query
Content-Type: application/json

{
  "survey_context": "বাংলাদেশে FrozenBerry ...",
  "questions": [
    "আপনি কি FrozenBerry ব্যবহার করেছেন?",
    "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
    "সংরক্ষণ করা সহজ কি?"
  ],
  "user_query": "জি বলছি। আমি FrozenBerry ব্যবহার করেছি।",
  "continue_in_same_conversation": true
}
```
Notes on session handling:
- The service manages `conversation_id` via an HTTP cookie. Clients do not need to send it explicitly.
- Set `continue_in_same_conversation=false` to start a fresh conversation; the cookie will be updated automatically.

## Error Cases
- 400: Missing `survey_context` or `questions`, or empty `questions` list.
- 502/504: LLM server unavailable or slow (check `llm-server` container).

## Performance & Scaling
- Tune `LLAMA_ARG_THREADS`, `LLAMA_ARG_THREADS_BATCH`, `LLAMA_ARG_N_PARALLEL`, and batching in `llm-server` to match hardware.
- Use your existing Locust script to stress the `llm-server` route `/v1/completions`. The llm-service forwards one request per call.

## File References
- Service code: [`llm_service/main.py`](../llm_service/main.py)
- Container: [`Dockerfile.llm_service`](../Dockerfile.llm_service)
- Dependencies: [`llm_service/requirements.txt`](../llm_service/requirements.txt)
- Stack: [`compose.yaml`](../compose.yaml)
- Test script: [`scripts/docker_llm-api-service_inference.py`](../scripts/docker_llm-api-service_inference.py)
