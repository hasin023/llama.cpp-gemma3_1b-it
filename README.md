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

