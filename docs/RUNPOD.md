# Load Testing with Locust on RunPod

This guide outlines the steps to deploy the LLM service on a RunPod GPU instance and run load tests using Locust.

## 1. Setup RunPod Instance

- **Template**: Select `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` which is Docker-ready template.
- **GPU Selection**: Select a GPU based on your requirements (e.g., `RTX 3090`, `RTX 4090`, or `A40`).
- **POD Name**: Example `llama-service-load-test`
- **Disk Space**:
  - **Container Disk**: `20GB`
  - **Volume Disk**: `30GB` (to store models and logs).
- **Expose Ports**:
  - Ensure **Port 22** is exposed (for SSH).
  - Ensure **Port 80** is mapped and exposed (for the application).
  - Ensure **Port 8089** is mapped and exposed (if running Locust GUI from the pod) (optional).

## 2. Connect to the Pod

RunPod provides an SSH command string. It usually looks like this:

```bash
ssh -p <port> root@<pod-ip> -i ~/.ssh/id_rsa
```

### Tips for Connection:

- **User**: Always `root` for standard RunPod templates.
- **Port**: Note the specific port assigned by RunPod (e.g., `12345`).
- **SSH Key**: Ensure your public key is added to your RunPod account settings before creating the pod.

> [!TIP]
> Use a terminal multiplexer like `tmux` or `screen` once connected to keep your session alive if the connection drops.

## 3. Environment Setup

### Git Configuration (Private Repo Access)

Inside the pod, generate an SSH key if needed:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Add the public key to your GitHub repository's **Deploy keys**.

### Verify Docker Installation

The recommended RunPod template (`runpod/pytorch`) comes with Docker pre-installed. You can simply verify it:

```bash
docker --version
docker compose version
```

## 4. Application Deployment

Clone the repository:

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone <your-repository-url>
cd <repo-directory>
```

### Enable GPU Mode

Before starting, modify `compose.yaml` to enable GPU acceleration if using a CUDA-enabled instance.

1.  Open `compose.yaml`.
2.  Uncomment the GPU-specific lines:
    ```yaml
    # === MODE B: GPU ENABLED ===
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ```
3.  Uncomment `LLAMA_ARG_N_GPU_LAYERS=-1` in the environment section to offload all layers to the GPU.

### Start the Application

If you have set the necessary environment variables (like `HF_TOKEN`) in the RunPod template or configuration, they will be automatically picked up. Otherwise, create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your keys if not set in RunPod
# nano .env
```

Start the service:

```bash
docker compose up --build -d
```

Verify containers:

```bash
docker ps
```

## 5. Load Testing

### Run Locust (Headless Mode)

From your **local machine**, target the RunPod IP and mapped port:

```bash
python -m locust -f .\scripts\locust_llm-service-api.py --headless -u 1 -r 1 -t 1800s --host http://<pod-ip> --system-config "RunPod A40"
```

### Run Locust (GUI Mode)

If you have port 8089 mapped:

```bash
python -m locust -f .\scripts\locust_llm-service-api.py
```

- Access at `http://<pod-ip>:8089`

## 6. Cleanup

To avoid unnecessary costs:

1.  **Terminate** the pod from the RunPod console (this deletes the pod and container disk).
2.  Alternatively, **Stop** the pod if you want to keep the Volume disk (you will still be charged for storage).
