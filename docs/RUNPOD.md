# Load Testing with Locust on RunPod (Direct Execution)

This guide outlines the steps to deploy the LLM service on a RunPod GPU pod and run load tests using Locust **without Docker Compose**.

> [!IMPORTANT]
> RunPod Pods are containers, not VMs. Docker-in-Docker is not supported. This guide runs services directly inside the pod.

## 1. Setup RunPod Pod

- **Template**: Select `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPU Selection**: Select a GPU based on your requirements (e.g., `RTX 3090`, `RTX 4090`, or `A40`).
- **Pod Name**: Example `llama_service_loadtest`
- **Disk Space**:
  - **Container Disk**: `20GB`
  - **Volume Disk**: `30GB` (to store models)
- **Expose Ports**:
  - **Port 22** (SSH)
  - **Port 8000** (llm_service API - **required for load testing**)
- **Environment Variables** (set in RunPod Pod Configuration):
  - `HF_TOKEN` - Your Hugging Face access token
  - `HF_REPO` - Your model repository (e.g., `username/model-repo`)
  - `HF_FILE` - Model file name (e.g., `gemma3-1b-ft-Q4_K_M.gguf`)
  - `LLAMA_CPP_ENDPOINT` - `http://localhost:8080/v1` (default)
  - `OPENAI_API_KEY` - `sk-no-key-required` (default)

## 2. Connect to the Pod

RunPod provides an SSH command string. It usually looks like this:

```bash
ssh -p <port> root@<pod-ip> -i ~/.ssh/id_rsa
```

> [!TIP]
> Use `tmux` or `screen` once connected to keep your session alive if the connection drops.

---

## 3. Git Configuration (Private Repo Access)

Inside the pod, generate an SSH key:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Add the private key to the agent:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

View the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output and add it to your GitHub repository's **Deploy keys** (Read-only access).

Test the connection:

```bash
ssh -T git@github.com
```

---

## 4. Build and Run llama.cpp Server

### 4.1 Install Build Dependencies

```bash
apt update && apt install -y cmake build-essential libcurl4-openssl-dev
```

### 4.2 Clone and Build llama.cpp with CUDA

```bash
cd ~
mkdir -p workspace && cd workspace

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

> [!NOTE]
> The build may take 5-10 minutes. The `-j $(nproc)` flag uses all available CPU cores.

### 4.3 Start the Server

```bash
cd ~/workspace/llama.cpp/build/bin

./llama-server \
  --host 0.0.0.0 \
  --port 8080 \
  --hf-repo "$HF_REPO" \
  --hf-file "$HF_FILE" \
  --hf-token "$HF_TOKEN" \
  --n-gpu-layers -1 \
  --ctx-size 8192 \
  --threads 4 \
  --threads-batch 4 \
  --batch-size 512 \
  --ubatch-size 256 \
  --n-parallel 1 \
  --chat-template gemma \
  --metrics \
  --slots \
  --warmup \
  --no-webui
```

> [!TIP]
> Run this inside a `tmux` session so it persists after you disconnect:
>
> ```bash
> tmux new -s llama
> # Run the command above
> # Detach with Ctrl+B, then D
> ```

### 4.4 Verify Server is Running

In another terminal (or tmux pane):

```bash
curl http://<pod-ip>:8080/health
```

Expected response: `{"status":"ok"}`

---

## 5. Run LLM Service (FastAPI Wrapper)

### 5.1 Clone the Repository

```bash
cd ~/workspace
git clone git@github.com:<username>/<repo-name>.git llm-app
cd llm-app
```

### 5.2 Install Dependencies

```bash
cd llm_service
pip install -r requirements.txt
```

### 5.3 Start the Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

> [!TIP]
> Run this in a separate `tmux` session:
>
> ```bash
> tmux new -s llm-service
> uvicorn main:app --host 0.0.0.0 --port 8000
> # Detach with Ctrl+B, then D
> ```

### 5.4 Verify Service is Running

```bash
curl http://<pod-ip>:8000/health
```

Expected response: `{"status":"ok"}`

---

## 6. Load Testing (From Local Machine)

Run Locust from your **local machine**, targeting the RunPod pod's public IP and port 8000.

### 6.1 Find Your Pod's Public IP

In the RunPod web console, find the pod's **Public IP** (or use the SSH connection hostname).

### 6.2 Run Locust (Headless Mode)

```bash
python -m locust -f .\scripts\LOCUST_llm-service-api.py --headless -u 1 -r 1 -t 1800s --host http://<pod-ip>:8000 --system-config "RunPod A40"
```

### 6.3 Run Locust (GUI Mode)

```bash
python -m locust -f .\scripts\LOCUST_llm-service-api.py
```

Then open `http://localhost:8089` in your browser and set:

- **Host**: `http://<pod-ip>:8000`
- **Number of users**: 1 (or more)
- **Spawn rate**: 1

---

## 7. Cleanup

To avoid unnecessary costs:

1. **Terminate** the pod from the RunPod console (deletes the pod and container disk).
2. Alternatively, **Stop** the pod to keep the Volume disk (you will still be charged for storage).
