## Model Download

Download the model manually to a location locally and save it there -

```bash
C:\llama\models
```

Download the models from here for 1b quantized to 4-bit [https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF)

Choose a quantized model file (recommended for balance of speed/quality):

- `gemma-3-1b-it-Q4_K_M.gguf` → good for most systems (4-bit, ~0.6 GB)

Save it to:

```bash
C:\llama\models\gemma-3-1b-it-Q4_K_M.gguf
```

---

## Run the Docker Command (Using `-m`)

Open **PowerShell** (as a regular user — no admin needed) and run:

```powershell
docker run -v C:/llama/models:/models -p 5050:8000 ghcr.io/ggml-org/llama.cpp:server -m /models/gemma-3-1b-it-Q4_K_M.gguf --port 8000 --host 0.0.0.0 -n 512
```

Basic web UI can be accessed via browser: [http://localhost:5050](localhost:5050)
Chat completion endpoint: [http://localhost:5050/v1/chat/completions](localhost:5050/v1/chat/completions)

---

## Test the Server

Run the `base.py`

---

llm-server params ->

- `--list-devices` print list of available devices and exit
- `-m, --model FNAME` model path to load
- `-mu, --model-url MODEL_URL` model download url (default: unused)
- `-dr, --docker-repo [<repo>/]<model>[:quant]` Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest. example: gemma3 (default: unused)
- `-hf, -hfr, --hf-repo <user>/<model>[:quant]` Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist. mmproj is also downloaded automatically if available. to disable, add --no-mmproj, example: unsloth/phi-4-GGUF:q4_k_m (default: unused)
- `--offline` Offline mode: forces use of cache, prevents network access
- `--temp N` temperature (default: 0.8)
- `--top-k N` top-k sampling (default: 40, 0 = disabled), (env: LLAMA_ARG_TOP_K)
- `--top-p N` top-p sampling (default: 0.9, 1.0 = disabled)
- `--warmup`, `--no-warmup` whether to perform warmup with an empty run (default: enabled)
- `-np, --parallel N` number of server slots (default: -1, -1 = auto)
- `-a, --alias STRING` set alias for model name (to be used by REST API)
- `--host HOST` ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: 127.0.0.1)
- `--port PORT` port to listen (default: 8080)
- `--path PATH` path to serve static files from (default: )
- `--api-prefix PREFIX` prefix path the server serves from, without the trailing slash (default: )
- `--webui`, `--no-webui` whether to enable the Web UI (default: enabled)
- `--chat-template-kwargs STRING` sets additional params for the json template parser
- `-to, --timeout N` server read/write timeout in seconds (default: 600)
- `--threads-http N` number of threads used to process HTTP requests (default: -1)
- `--metrics` enable prometheus compatible metrics endpoint (default: disabled)
- `--models-max N` for router server, maximum number of models to load simultaneously (default: 4, 0 = unlimited)
- `--jinja, --no-jinja` whether to use jinja template engine for chat (default: enabled)
- `--chat-template-file JINJA_TEMPLATE_FILE` set custom jinja chat template file (default: template taken from model's metadata) if suffix/prefix are specified, template will be disabled only commonly used templates are accepted (unless --jinja is set before this flag):
  list of built-in templates:
  bailing, bailing-think, bailing2, chatglm3, chatglm4, chatml, command-r, deepseek, deepseek2, deepseek3, exaone3, exaone4, falcon3, gemma, gigachat, glmedge, gpt-oss, granite, grok-2, hunyuan-dense, hunyuan-moe, kimi-k2, llama2, llama2-sys, llama2-sys-bos, llama2-sys-strip, llama3, llama4, megrez, minicpm, mistral-v1, mistral-v3, mistral-v3-tekken, mistral-v7, mistral-v7-tekken, monarch, openchat, orion, pangu-embedded, phi3, phi4, rwkv-world, seed_oss, smolvlm, vicuna, vicuna-orca, yandex, zephyr

- `--sleep-idle-seconds SECONDS` number of seconds of idleness after which the server will sleep (default: -1; -1 = disabled)
- `-t, --threads N` number of CPU threads to use during generation (default: -1) (env: LLAMA_ARG_THREADS)
- `-td, --threads-draft N` number of threads to use during generation (default: same as --threads)
- `-tbd, --threads-batch-draft N` number of threads to use during batch and prompt processing (default: same as --threads-draft)

- `-np, --parallel N` number of server slots (default: -1, -1 = auto) (env: LLAMA_ARG_N_PARALLEL)
- `-cb, --cont-batching, -nocb, --no-cont-batching` whether to enable continuous batching (a.k.a dynamic batching) (default: enabled) (env: LLAMA_ARG_CONT_BATCHING)

---

```bash
cpus: N # Docker's hard limit on CPU usage
LLAMA_ARG_THREADS=N # Threads for token generation, During inference (outputting tokens)
LLAMA_ARG_THREADS_BATCH=N # Threads for prompt processing, When processing the input prompt
LLAMA_ARG_N_PARALLEL=N # How many slots the server has for concurrent requests
```

# On the Orange Pi

```bash
nproc --all
```

### Recommended Configuration

| Orange Pi Cores | cpus        | LLAMA_ARG_THREADS | LLAMA_ARG_N_PARALLEL |
| :-------------- | :---------- | :---------------- | :------------------- |
| 2 cores         | 2 (or omit) | 2                 | 2-4                  |
| 4 cores         | 4 (or omit) | 4                 | 4-6                  |
| 8 cores         | 6-8         | 6-8               | 4-8                  |

For a typical **4-core Orange Pi**, I recommend:

```yaml
services:
  llm-server:
    # ... other config ...
    environment:
      - LLAMA_ARG_THREADS=4 # Use all cores for inference
      - LLAMA_ARG_THREADS_BATCH=4 # Use all cores for prompt processing
      - LLAMA_ARG_N_PARALLEL=4 # 4 concurrent slots
    deploy:
      resources:
        limits:
          cpus: "4" # Allow container to use all 4 cores
```

For a **2-core device**:

```yaml
environment:
  - LLAMA_ARG_THREADS=2
  - LLAMA_ARG_THREADS_BATCH=2
  - LLAMA_ARG_N_PARALLEL=2 # Reduce slots to match
deploy:
  resources:
    limits:
      cpus: "2"
```

- Find Orange PI IP - `hostname -I`

```python
# Change this on your laptop/client
ENDPOINTS = ["http://192.168.1.50:8080/v1"]
```
