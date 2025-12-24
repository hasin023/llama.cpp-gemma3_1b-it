# LLaMA.cpp Server on Edge device (Docker)

This repository contains the configuration for a self-hosted, OpenAI-compatible Large Language Model (LLM) server optimized for the Edge device platform (ARM64). It leverages `llama.cpp` to provide high-performance inference on resource-constrained edge devices.

## Technical Overview and Advantages

### Why LLaMA.cpp for Edge Deployment?

For edge deployments on devices like the Edge device (typically 2-8 GB RAM, ARMv8 CPU), standard heavy-framework solutions (e.g., PyTorch/HuggingFace Transformers) are often infeasible due to memory overhead and lack of optimization. LLaMA.cpp offers critical advantages for this use case:

1.  **Quantization Efficiency**:
    Using GGUF quantization (e.g., Q4_K_M), we reduce memory bandwidth requirements significantly while retaining model accuracy. A 1B parameter model typically requires <1GB RAM, allowing it to reside entirely in physical memory without swapping.

2.  **Continuous Batching**:
    Unlike traditional sequential inference, our configuration uses continuous batching. This allows the server to process multiple requests concurrently by interleaving their token generation steps. This maximizes CPU/GPU utilization and prevents a long-running request from blocking shorter ones.

3.  **Slot-Based Parallelism**:
    We configure a fixed number of "slots" (concurrent contexts). Each slot maintains its own KV cache. When a request arrives, it is assigned to an idle slot. This architecture is essential for maintaining throughput in a multi-user environment.

4.  **Hardware Optimization**:
    The Docker image (`ghcr.io/ggml-org/llama.cpp:server`) includes ARM64-optimized kernels (NEON/Int8/Int4 dot products), ensuring that the CPU is utilized to its maximum theoretical throughput.

## System Architecture

- **Runtime**: Docker Container (Isolated, Portable)
- **Engine**: llama.cpp Server (HTTP/REST API)
- **Monitoring**: Python Sidecar Service (Prometheus Metrics Poll)
- **Hardware Profile**:
  - Target: Edge device (2-Core / 4-Core variants)
  - Optimization: Thread affinity matched to physical cores

## Configuration Reference

The system is tuned via environment variables in `compose.yaml`. The current configuration is optimized for a **2-Core Edge device**.

### Configuration Parameters Reference

| Parameter                      | Current Value | Description                                                  | Impact of Lowering Value                                                                         | Impact of Raising Value                                                                                                      |
| :----------------------------- | :------------ | :----------------------------------------------------------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| `LLAMA_ARG_THREADS`            | `2`           | Number of threads used for token generation (inference).     | **Slower Generation:** Underutilizes CPU cores, reducing tokens-per-second.                      | **Context Switching:** On a 2-core device, raising this causes CPU contention and degrades performance.                      |
| `LLAMA_ARG_THREADS_BATCH`      | `2`           | Number of threads used for prompt processing (batch).        | **Slower TTTF:** Increases Time To First Token as prompt evaluation takes longer.                | **Context Switching:** Similar to generation threads, excess threads cause contention on limited cores.                      |
| `LLAMA_ARG_N_PARALLEL`         | `2`           | Number of concurrent requests (slots) the server can handle. | **Queueing:** Reduces concurrency; 3rd user must wait for a slot to free up.                     | **OOM Risk:** Each slot consumes significant RAM for KV Cache. Raising this may crash the container on low-RAM devices.      |
| `LLAMA_ARG_BATCH`              | `256`         | Logical batch size for prompt processing.                    | **None/Minor:** May slightly reduce memory usage but increases prompt processing time.           | **Higher Memory:** Increases RAM usage significantly. Benefit diminishes if CPU cannot compute distinct batches fast enough. |
| `LLAMA_ARG_UBATCH`             | `128`         | Physical batch size (micro-batch) for execution.             | **Latency:** May improve responsiveness/latency slightly but reduces total throughput.           | **Bus Saturation:** Larger ubatches choke the memory bandwidth of edge devices, increasing latency.                          |
| `LLAMA_ARG_CTX_SIZE`           | `8192`        | Context window size (in tokens).                             | **Limited Context:** Model 'forgets' earlier conversation parts sooner. Saves RAM.               | **OOM Risk:** Memory usage scales linearly. Raising this often leads to Out Of Memory crashes on 2GB/4GB boards.             |
| `LLAMA_ARG_CONT_BATCHING`      | `true`        | Enables continuous batching (dynamic scheduling).            | **Blocking:** Disabling this forces sequential processing, destroying multi-user performance.    | **N/A:** Boolean flag.                                                                                                       |
| `LLAMA_ARG_ENDPOINT_METRICS`   | `1`           | Enables Prometheus metrics endpoint.                         | **Blindness:** Disables monitoring; cannot track cache usage or system load.                     | **N/A:** Boolean flag.                                                                                                       |
| `LLAMA_ARG_SLEEP_IDLE_SECONDS` | `-1`          | Time until server sleeps when idle (-1 = disabled).          | **Power Saving:** Setting a positive value saves power but causes startup delay on next request. | **N/A:** Setting to -1 keeps server always ready.                                                                            |
| `LLAMA_ARG_TEMP`               | `0.4`         | Temperature (randomness) of response.                        | **Rigid:** Responses become deterministic and repetitive. Good for factual queries.              | **Hallucination:** Responses become creative but prone to errors and incoherence.                                            |
| `LLAMA_ARG_TOP_K`              | `20`          | Limits vocabulary to top K probable tokens.                  | **Focus:** Responses are very "safe" and predictable.                                            | **Diversity:** Responses effectively consider more vocabulary, increasing diversity but risk of nonsense.                    |

_Note: For 4-core devices, increase `LLAMA_ARG_THREADS`, `LLAMA_ARG_THREADS_BATCH`, and `LLAMA_ARG_N_PARALLEL` to 4, provided sufficient RAM is available._

### Critical Warning: CPU Oversubscription

**Scenario:** Running multiple containers on a single edge device (e.g., 2 containers, each set to use all 4 cores).
**Outcome:** Severe performance degradation.

If you run two containers efficiently configured for 4 cores on a single 4-core device, you are oversubscribing resources by 200%.

1.  **Context Switching:** The OS spends significant cycles pausing/resuming containers, reducing actual inference time.
2.  **Cache Trashing:** CPU caches are constantly flushed between containers, causing high latency memory fetches.
3.  **Memory Bandwidth:** Two active models will saturate the shared system RAM bandwidth.

**Recommendation:**

- **Single Container (Best Performance):** Run 1 container with `LLAMA_ARG_THREADS=4` and `LLAMA_ARG_N_PARALLEL=4`.
- **Multi-Container (Isolation):** If you MUST run two containers, limit each to 2 cores (`cpus: "2"` in Docker) and set `LLAMA_ARG_THREADS=2`.

### Performance Troubleshooting: Memory Constraints (Q4 vs Q8)

If you observe **low CPU usage (40-70%)** and **extremely low throughput (< 0.1 req/s)** when using higher precision models (e.g., Q8), you are likely hitting a **Memory Bottleneck**.

**The Issue:**
The `compose.yaml` sets a hard memory limit for the container (default `1G`).

- **Q4 Models (~700MB):** Fit comfortably within RAM along with the KV cache (Context). Performance is high because the model stays in physical memory.
- **Q8 Models (~1.2GB):** Exceed the 1GB limit. The OS is forced to continuously swap memory pages to disk ("thrashing"). The CPU sits idle waiting for slow Disk I/O, resulting in poor performance.

**Solution:**

1. **Increase Memory Limit:** Update `compose.yaml` to allow more RAM if your device has it (uncomment `memory: 2G` for Q8 models, requires >1.5GB System RAM).
2. **Use Lower Quantization:** Stick to Q4_K_M or Q5_K_M models for devices with < 1.5GB RAM.
3. **Optimize Cache:** If stuck with 1GB, reduce context size (`LLAMA_ARG_CTX_SIZE`) or quantize the KV cache (`LLAMA_ARG_CACHE_TYPE_K=q8_0`).

### Alternative Scenario: Split-Core Deployment (2 Containers, 2 Cores Each)

**Scenario:** Running 2 independent containers, each strictly limited to 2 physical cores.
**Total Load:** 4 cores fully utilized (2+2).

**Pros:**

- **High Availability (Isolation):** If Container A crashes (e.g., OOM or segfault), Container B continues processing requests. This is the primary benefit.
- **Resource Fairness:** Docker guarantees each container gets 50% CPU time, preventing one "bully" request from starving others.

**Cons:**

- **Double Memory Usage:** You must load the model weights into RAM _twice_. For a 1B model, this wastes ~1GB of RAM that could have been used for a larger context window or batch size.
- **Memory Bandwidth Contention:** Even though cores are split, they share the same system memory bus. Two active models fighting for memory access will lower the tokens-per-second of _both_ containers.
- **Higher Latency:** A 2-core inference is inherently slower than a 4-core inference. Individual user request latency will increase (slower generation speed), even if total throughput (requests/min) remains similar.
- **Complexity:** Requires an external load balancer (NGINX/HAProxy) to route traffic between ports 8080 and 8081.

**Verdict:** For a single edge device, **one large container** is almost always better than **two small containers** due to memory efficiency and lower latency. Use the split-core approach only if **fault tolerance** is strictly more important than raw performance.

## Quick Start Guide

### 1. Model Preparation

The configuration in `compose.yaml` is set to automatically download the optimized `gemma3-1b-ft` model from Hugging Face.

**Option A: Automatic Download (Recommended)**
Ensure your `HF_TOKEN` is set in your environment (if accessing private repos) or simply run the stack. The server will fetch the model defined in `LLAMA_ARG_HF_REPO` on first launch.

**Option B: Manual Placement**
If you prefer to manually manage models or fully offline operation:

1.  Download your `.gguf` file.
2.  Place it in the `./models/` directory.
3.  Update `compose.yaml` to point `LLAMA_ARG_MODEL` to your local file (e.g., `/models/my-model.gguf`) and comment out the `HF_REPO` lines.

```bash
mkdir -p models logs
```

### 2. Service Initialization

Start the containerized stack.

```bash
docker compose up -d
```

### 3. Verification

Verify the service health and network accessibility.

```bash
# Query health endpoint (replace IP accordingly)
hostname -I
curl http://<ORANGE_PI_IP>:8080/health
```

### 4. Hardware Acceleration (Optional)

#### Enabling GPU Support

If your host machine possesses an NVIDIA GPU, enabling hardware acceleration can drastically improve inference throughput (tokens/sec).

**Prerequisites:**

- NVIDIA GPU with valid drivers installed.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.

**Instructions:**

1. Open `compose.yaml`.
2. **Comment out** the entire **MODE A: CPU ONLY** block (add `#` to valid lines).
3. **Uncomment** the entire **MODE B: GPU ENABLED** block (remove `#`).
4. Scroll down to the `environment:` section and **uncomment** `- LLAMA_ARG_N_GPU_LAYERS=-1`.

**Example of GPU Mode:**

```yaml
  llm-server:
    ports:
      - "8080:8080"

    # === MODE A: CPU ONLY (DEFAULT) ===
    # image: ghcr.io/ggml-org/llama.cpp:server        <-- COMMENTED
    # deploy:                                         <-- COMMENTED
    #   resources:                                    <-- COMMENTED
    #     limits:                                     <-- COMMENTED
    #       cpus: "2"                                 <-- COMMENTED
    #       memory: 1G                                <-- COMMENTED
    # ==================================

    # === MODE B: GPU ENABLED (Uncomment to enable) ===
    image: ghcr.io/ggml-org/llama.cpp:server-cuda     <-- UNCOMMENTED
    deploy:                                           <-- UNCOMMENTED
      resources:                                      <-- UNCOMMENTED
        reservations:                                 <-- UNCOMMENTED
          devices:                                    <-- UNCOMMENTED
            - driver: nvidia                          <-- UNCOMMENTED
              count: all                              <-- UNCOMMENTED
              capabilities: [gpu]                     <-- UNCOMMENTED
    # =================================================

    environment:
      # ...
      # GPU Offloading (Uncomment for GPU Mode)
      - LLAMA_ARG_N_GPU_LAYERS=-1                     <-- UNCOMMENTED
```

5. Restart the stack: `docker compose up -d --force-recreate`

## Logging and Observability

The system implements a dual-layer logging strategy for audit and performance tracking.

### Application Logs

Located at: `./logs/server.log`

- Captures full verbosity input prompts and generated responses.
- Includes token counts (prompt tokens vs. generated tokens) for billing/usage analysis.
- Provides millisecond-level timing for "prompt eval" and "generation" stages.

### Performance Metrics

Located at: `./logs/monitor_metrics.log`

- Generated by the sidecar `monitor` service.
- Structured JSON logs suitable for ingestion by log aggregation systems (e.g., ELK, Splunk).
- Key Metrics:
  - `kv_cache_usage`: Ratio of context window utilized.
  - `slots_busy`: Live count of active requests.
  - `requests_queued`: Backpressure indicator.

## Deployment Strategy (Cluster/Fleet)

To deploy this scalable architecture across a fleet of Edge devices, follow this standardized procedure.

### 1. Base Image Preparation

Ensure all nodes have the following prerequisites:

- Docker Engine & Docker Compose plugin installed.
- User permissions set for the `docker` group.
- Static IP address assignment (recommended for API stability).

### 2. Standardization

Maintain a consistent directory structure across the cluster:
`/opt/llm-server/` containing:

- `compose.yaml`
- `monitor.py`
- `models/` (Populated ideally via rsync or shared network storage mount to ensure model version consistency)

### 3. Ansible / SSH Deployment

For multi-device management, use a simple loop or Ansible playbook to update configuration.

**Update Procedure:**

1.  `docker compose down`
2.  `git pull` (or rsync updated config)
3.  `docker compose up -d`

### 4. Load Balancing Strategy (4 Boards, 4 Containers)

To distribute traffic across 4 independent Orange Pi boards (e.g., IPs `.50` to `.53`), use **NGINX** as a Layer 7 load balancer.

**Recommended Tool:** NGINX (Standard, high performance, easy SSE support)
**Algorithm:** `least_conn` (Least Connections)

- **Why?** LLM inference times vary wildly (short vs. long prompts). Round-robin might send a new request to a node that is still busy processing a massive context. `least_conn` ensures new work goes to the idlest node.

**Configuration (`nginx.conf`):**

```nginx
http {
    upstream llm_cluster {
        # 'least_conn' is CRITICAL for LLM workloads to prevent queue pile-ups
        least_conn;

        server 192.168.1.50:8080 max_fails=3 fail_timeout=30s;
        server 192.168.1.51:8080 max_fails=3 fail_timeout=30s;
        server 192.168.1.52:8080 max_fails=3 fail_timeout=30s;
        server 192.168.1.53:8080 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://llm_cluster;

            # === SSE / Streaming Support (MANDATORY for LLMs) ===
            # Disable buffering so tokens flow to the client immediately
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            chunked_transfer_encoding on;

            # === Timeouts ===
            # LLMs can take minutes to generate long responses
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
        }
    }
}
```

**Implementation Note:** Run this NGINX instance on a separate lightweight device, or on one of the 4 nodes (though this consumes some CPU). Ideally, run it on your router or a dedicated entry-point Raspberry Pi/VM.

## Testing and Validation

Use the included Locust script for load verification.

```bash
# Stress test with 50 concurrent users
locust -f LOCUST_llama-server_docker_inf.py --headless -u 50 -r 17 -t 120s --host http://localhost:8080
```

## Interactive Survey CLI

To interact with the LLM survey agent using a turn-by-turn CLI tool, use `survey_cli.py`. This tool manages conversation history and provides pre-defined survey templates.

### Prerequisites

Ensure you have installed the dependencies:

```bash
pip install -r requirements.txt
```

### Usage

**1. List Available Surveys**

View all pre-defined survey templates (e.g., FrozenBerry, Healthcare, Banking).

```bash
python survey_cli.py list
```

**2. Start a Survey Chat**

Start an interactive session with a specific survey template.

```bash
# Chat with the FrozenBerry product survey
python survey_cli.py chat frozen-berry

# Chat with the Student Life reflection survey
python survey_cli.py chat student-life
```

**3. Custom Survey**

Create your own survey on the fly by providing a context and a list of questions.

```bash
python survey_cli.py custom
```

### Features

- **Automatic History:** Manages the conversation turns ("user", "model") automatically.
- **Rich Output:** Uses colored text and spinners for a better user experience.
- **Bengali Support:** Configured to display Bengali script correctly in Windows terminals.
