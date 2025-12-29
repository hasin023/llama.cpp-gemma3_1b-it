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

The system is tuned via environment variables in `compose.yaml`. The current configuration is optimized for CPU-based inference on edge devices.

### Environment Variables Explained

This section provides detailed explanations of all environment variables used in `compose.yaml`, their purpose, and CPU-based recommendations.

#### Model Loading & Source Configuration

| Variable            | Current Value          | Description                                                                                                                                                                      | CPU Recommendation                                                                                        |
| ------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `HF_TOKEN`          | `${HF_TOKEN}`          | Hugging Face access token for downloading private models. Required only if accessing private repositories.                                                                       | Set this if you need to download private models. For public models, this can be omitted.                  |
| `LLAMA_ARG_HF_REPO` | `${LLAMA_ARG_HF_REPO}` | Hugging Face repository path in format `<user>/<model>[:quant]`. The server automatically downloads the model on first launch. Quantization defaults to Q4_K_M if not specified. | Use this for automatic model management. Example: `hasin023/gemma3-1b-ft:Q4_K_M`                          |
| `LLAMA_ARG_HF_FILE` | `${LLAMA_ARG_HF_FILE}` | Specific GGUF file name to download from the Hugging Face repo. Overrides the quant specified in `HF_REPO`.                                                                      | Use when you need a specific quantization (e.g., `gemma3-1b-ft-Q4_K_M.gguf`). Leave empty to use default. |
| `LLAMA_CACHE`       | `/models`              | Directory path where downloaded models are cached. Models are stored here for reuse across container restarts.                                                                   | Set to a persistent volume mount (`./models:/models`) to avoid re-downloading models.                     |

#### Network & Server Configuration

| Variable                 | Current Value       | Description                                                                                                     | CPU Recommendation                                                                                                                                              |
| ------------------------ | ------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLAMA_ARG_HOST`         | `0.0.0.0`           | IP address the server binds to. `0.0.0.0` allows external connections; `127.0.0.1` restricts to localhost only. | Use `0.0.0.0` for edge deployments where clients connect over the network. Use `127.0.0.1` only for local testing.                                              |
| `LLAMA_ARG_PORT`         | `${LLAMA_ARG_PORT}` | TCP port number for the HTTP server. Default is `8080` if not set.                                              | Standard port `8080` works well. Change if you have port conflicts. Ensure firewall rules allow this port.                                                      |
| `LLAMA_ARG_THREADS_HTTP` | `-1`                | Number of threads dedicated to processing HTTP requests. `-1` means auto-detect based on CPU cores.             | **CPU Recommendation:** Keep at `-1` for automatic detection. For CPU-only deployments, this typically uses 1-2 threads, which is sufficient for HTTP handling. |

#### CPU Threading Configuration

| Variable                  | Current Value | Description                                                                                                                        | CPU Recommendation                                                                                                                                                                                               |
| ------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLAMA_ARG_THREADS`       | `4`           | Number of CPU threads used for **token generation** (inference phase). Each thread processes tokens in parallel during generation. | **CPU Recommendation:** Set to match your physical CPU cores. For 2-core devices: `2`, for 4-core: `4`. Oversubscribing (e.g., 8 threads on 4 cores) causes context switching overhead and degrades performance. |
| `LLAMA_ARG_THREADS_BATCH` | `4`           | Number of CPU threads used for **prompt processing** (batch evaluation phase). This affects Time-To-First-Token (TTFT).            | **CPU Recommendation:** Match `LLAMA_ARG_THREADS`. For CPU-only inference, prompt processing benefits from parallelization. Use same value as `THREADS` for balanced performance.                                |

**Threading Best Practices for CPU:**

- **2-Core Device:** `THREADS=2`, `THREADS_BATCH=2`
- **4-Core Device:** `THREADS=4`, `THREADS_BATCH=4`
- **8-Core Device:** `THREADS=8`, `THREADS_BATCH=8`
- **Avoid:** Setting threads > physical cores (causes contention)

#### Parallel Request Handling (Slots)

| Variable               | Current Value | Description                                                                                                             | CPU Recommendation      |
| ---------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `LLAMA_ARG_N_PARALLEL` | `1`           | Number of concurrent request **slots**. Each slot maintains its own KV cache and can process one request independently. | **CPU Recommendation:** |

- **2-Core:** Start with `1-2` slots. Each slot consumes ~200-500MB RAM for KV cache.
- **4-Core:** Can handle `2-4` slots depending on RAM availability.
- **Memory Constraint:** Each slot = `CTX_SIZE × ~64 bytes` of RAM. With `CTX_SIZE=8192`, each slot needs ~512KB-1MB for KV cache.
- **Trade-off:** More slots = better concurrency but higher memory usage. For CPU-only, `1-2` slots is often optimal to avoid memory pressure. |

#### Batch Processing Configuration

| Variable          | Current Value | Description                                                                                                                                                 | CPU Recommendation      |
| ----------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `LLAMA_ARG_BATCH` | `512`         | **Logical batch size** - maximum number of tokens processed together during prompt evaluation. Larger batches improve throughput but increase memory usage. | **CPU Recommendation:** |

- **2-Core/Low RAM:** `256-512` (current setting is good)
- **4-Core/Medium RAM:** `512-1024`
- **8-Core/High RAM:** `1024-2048`
- **Memory Impact:** Each batch token consumes memory. Lower values reduce RAM usage but may slightly increase prompt processing time. |
  | `LLAMA_ARG_UBATCH` | `256` | **Physical batch size** (micro-batch) - actual tokens processed per GPU/CPU operation. Smaller values improve latency but reduce throughput. | **CPU Recommendation:**
- **Edge Devices:** `128-256` (current `256` is optimal)
- **Desktop CPUs:** `256-512`
- **Why Smaller:** CPU memory bandwidth is limited. Large ubatches saturate the memory bus, causing latency spikes. Smaller ubatches keep the pipeline flowing smoothly. |

#### Context Window Configuration

| Variable             | Current Value | Description                                                                                                  | CPU Recommendation      |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------- |
| `LLAMA_ARG_CTX_SIZE` | `8192`        | Maximum context window size in tokens. This determines how much conversation history the model can remember. | **CPU Recommendation:** |

- **Low RAM (2GB):** `4096-6144` tokens
- **Medium RAM (4GB):** `8192` tokens (current setting)
- **High RAM (8GB+):** `16384` tokens
- **Memory Formula:** Context memory ≈ `CTX_SIZE × 64 bytes × N_PARALLEL`. With `8192` and `N_PARALLEL=1`, expect ~512KB-1MB per slot.
- **Trade-off:** Larger context = more memory but better conversation continuity. |

#### Continuous Batching

| Variable                  | Current Value | Description                                                                                                                                              | CPU Recommendation                                                                                                                                                                                  |
| ------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLAMA_ARG_CONT_BATCHING` | `true`        | Enables **continuous batching** (dynamic batching). Allows the server to interleave token generation from multiple requests, maximizing CPU utilization. | **CPU Recommendation:** **Always enable (`true`)** for CPU-only deployments. This is critical for multi-user scenarios. Disabling forces sequential processing, which severely degrades throughput. |

#### Logging & Monitoring

| Variable                     | Current Value      | Description                                                                                                                 | CPU Recommendation                                                                                                                                                 |
| ---------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LLAMA_LOG_TIMESTAMPS`       | `1`                | Adds timestamps to log messages. Useful for debugging and performance analysis.                                             | Enable (`1`) for production deployments to track request timing and debug issues.                                                                                  |
| `LLAMA_ARG_VERBOSE`          | `0`                | Enables verbose logging that prints prompts and responses. `0` = disabled, `1` = enabled.                                   | **CPU Recommendation:** Keep at `0` for production (reduces I/O overhead). Set to `1` only for debugging. Verbose logging adds significant I/O load.               |
| `LLAMA_LOG_VERBOSITY`        | `20`               | Log verbosity threshold. Values: `0`=generic, `1`=error, `2`=warning, `3`=info, `4`=debug. Higher values show more detail.  | **CPU Recommendation:** Use `3` (info) for production, `4` (debug) for troubleshooting. Value `20` seems incorrect - should be `0-4`.                              |
| `LLAMA_LOG_FILE`             | `/logs/server.log` | File path where application logs are written. Must be a mounted volume for persistence.                                     | Ensure the `/logs` directory is mounted as a volume (`./logs:/logs`) to persist logs across container restarts.                                                    |
| `LLAMA_ARG_ENDPOINT_METRICS` | `1`                | Enables Prometheus-compatible metrics endpoint at `/metrics`. Provides performance metrics (tokens/sec, cache usage, etc.). | **CPU Recommendation:** Enable (`1`) for monitoring. The metrics endpoint has minimal overhead and provides valuable insights into CPU utilization and throughput. |
| `LLAMA_ARG_ENDPOINT_SLOTS`   | `1`                | Enables the `/slots` endpoint for monitoring active request slots and their status.                                         | Enable (`1`) to monitor slot utilization and debug request queuing issues.                                                                                         |

#### Model Behavior (Sampling Parameters)

| Variable                  | Current Value | Description                                                                                        | CPU Recommendation                                                                                           |
| ------------------------- | ------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `LLAMA_ARG_CHAT_TEMPLATE` | `gemma`       | Chat template format. Must match your model's expected format (e.g., `gemma`, `llama3`, `chatml`). | Match this to your model. For Gemma models, use `gemma`. Check model documentation for the correct template. |
| `LLAMA_ARG_N_PREDICT`     | `-1`          | Maximum tokens to generate. `-1` means unlimited (generates until EOS token or context limit).     | **CPU Recommendation:**                                                                                      |

- **Chat Applications:** `-1` (unlimited) allows natural conversation flow
- **API/Production:** Set a limit (e.g., `512`, `1024`) to prevent runaway generation and control costs
- **Edge Devices:** Consider `256-512` to limit response time and memory usage |
  | `LLAMA_ARG_TEMP` | `0.8` | **Temperature** - controls randomness. Lower = more deterministic, Higher = more creative. Range: `0.0` to `2.0`. | **CPU Recommendation:**
- **Factual/Code:** `0.1-0.3` (deterministic)
- **Creative Writing:** `0.7-0.9` (current `0.8` is good)
- **Balanced:** `0.5-0.7`
- **Note:** Lower temperature doesn't reduce CPU load, only affects output diversity. |
  | `LLAMA_ARG_TOP_K` | `10` | **Top-K sampling** - limits vocabulary to the K most probable tokens. `0` = disabled (considers all tokens). | **CPU Recommendation:**
- **Focused Responses:** `10-20` (current `10` is good for focused output)
- **Balanced:** `40` (default)
- **More Diverse:** `50-100`
- **CPU Impact:** Lower values slightly reduce computation but have minimal performance impact. |
  | `LLAMA_ARG_TOP_P` | `0.8` | **Nucleus sampling** - considers tokens with cumulative probability up to P. `1.0` = disabled. Works together with Top-K. | **CPU Recommendation:**
- **Focused:** `0.7-0.8` (current `0.8` is good)
- **Balanced:** `0.9` (default)
- **More Diverse:** `0.95-1.0`
- **Combined with Top-K:** Lower values create more focused, predictable responses. |
  | `LLAMA_ARG_MIN_P` | `0.05` | **Min-P sampling** - minimum probability threshold relative to the most likely token. Filters out very low-probability tokens. | **CPU Recommendation:**
- **Standard:** `0.05` (current setting, good default)
- **Stricter:** `0.1` (more focused)
- **Disabled:** `0.0`
- **Purpose:** Prevents the model from considering extremely unlikely tokens, improving quality. |
  | `LLAMA_ARG_REPEAT_PENALTY` | `1.1` | **Repetition penalty** - penalizes repeated token sequences. `1.0` = no penalty, `>1.0` = penalize repetition. | **CPU Recommendation:**
- **Standard:** `1.1` (current setting, good default)
- **More Creative:** `1.0` (no penalty)
- **Less Repetitive:** `1.15-1.2`
- **Note:** Higher values don't increase CPU load, only affect output quality. |

#### Performance Optimizations

| Variable         | Current Value | Description                                                                                                                                                | CPU Recommendation      |
| ---------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `LLAMA_ARG_NUMA` | `distribute`  | NUMA (Non-Uniform Memory Access) optimization. Options: `distribute` (spread across nodes), `isolate` (local node only), `numactl` (use numactl settings). | **CPU Recommendation:** |

- **Multi-Socket Systems:** Use `distribute` (current setting) to utilize all CPU sockets
- **Single-Socket/Edge:** Can be omitted or set to `distribute` (harmless on single-socket systems)
- **Note:** Most edge devices are single-socket, so this has minimal impact but doesn't hurt |
  | `LLAMA_ARG_WARMUP` | `true` | Performs an empty inference run on startup to "warm up" the model and initialize caches. Reduces latency for the first request. | **CPU Recommendation:** **Always enable (`true`)**. The warmup cost is minimal (one empty inference) and significantly improves first-request latency. |
  | `LLAMA_ARG_NO_WEBUI` | `true` | Disables the built-in web UI. Reduces memory footprint and startup time. | **CPU Recommendation:** Disable (`true`) for production/API deployments. Enable only if you need the web interface for testing. Saves ~10-20MB RAM. |
  | `LLAMA_ARG_SLEEP_IDLE_SECONDS` | `-1` | Automatic sleep mode after idle period. `-1` = disabled (always ready). Positive value = seconds of idleness before sleeping. | **CPU Recommendation:**
- **Production/Always-On:** `-1` (current setting, always ready)
- **Power-Saving:** `300-600` (sleep after 5-10 minutes idle)
- **Trade-off:** Sleep mode saves power but causes delay on next request (model reload time). For edge devices that need instant responses, keep at `-1`. |

### CPU-Based Configuration Recommendations Summary

#### For 2-Core Edge Devices (2GB-4GB RAM)

LLAMA_ARG_THREADS=2
LLAMA_ARG_THREADS_BATCH=2
LLAMA_ARG_N_PARALLEL=1
LLAMA_ARG_BATCH=512
LLAMA_ARG_UBATCH=256
LLAMA_ARG_CTX_SIZE=4096-6144
LLAMA_ARG_CONT_BATCHING=true#### For 4-Core Edge Devices (4GB-8GB RAM)
LLAMA_ARG_THREADS=4
LLAMA_ARG_THREADS_BATCH=4
LLAMA_ARG_N_PARALLEL=2
LLAMA_ARG_BATCH=512-1024
LLAMA_ARG_UBATCH=256
LLAMA_ARG_CTX_SIZE=8192
LLAMA_ARG_CONT_BATCHING=true#### For 8-Core Desktop CPUs (8GB+ RAM)
LLAMA_ARG_THREADS=8
LLAMA_ARG_THREADS_BATCH=8
LLAMA_ARG_N_PARALLEL=4
LLAMA_ARG_BATCH=1024-2048
LLAMA_ARG_UBATCH=512
LLAMA_ARG_CTX_SIZE=16384
LLAMA_ARG_CONT_BATCHING=true### Critical Warning: CPU Oversubscription

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

First of all, validate the docker compose file syntax.

```bash
docker compose config
```

If the syntax is correct, start the containerized stack.

```bash
docker compose up -d
```

**What happens during startup:**

1. **llama-cpp-server** starts and loads the model (this can take 30-60 seconds)
2. **llm-service-api** waits for llama-cpp-server to be healthy, then starts
3. **warmup** service automatically runs once:
   - Waits for both services to be healthy
   - Sends a minimal warmup request to `/query` endpoint
   - Ensures the model is fully initialized and ready for real traffic
   - Exits cleanly (one-shot container)

You can monitor the warmup progress:

```bash
# Watch warmup logs
docker compose logs -f warmup

# Check all service status
docker compose ps
```

The warmup service ensures your first real request won't experience cold-start latency. If warmup fails, the stack will still work, but the first request may be slower.

**Note:** The warmup service is a one-shot container that runs once and exits. To disable it, comment out the `warmup:` service block in `compose.yaml`.

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

Located at: `./logs/llama_cpp_server.log`

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

### Test Scripts

The repository includes several test scripts in the `scripts/` directory:

- **`docker_llm-api-service_inference.py`**: Tests the LLM Service API wrapper (conversation management)
- **`llama-server_docker_inf.py`**: Tests direct llama.cpp server using completions API
- **`llama-server_docker_inf_chat.py`**: Tests direct llama.cpp server using chat completions API

Example usage:

```bash
# Test the LLM Service API
python scripts/docker_llm-api-service_inference.py

# Test direct llama.cpp server
python scripts/llama-server_docker_inf.py
```

## Voice Agent Orchestrator (Pseudocode)

End-to-end voice pipeline:

- **STT (Speech-to-Text)**: converts user audio → `user_query` text
- **LLM Service (this project)**: builds Gemma survey prompt, maintains conversation turns, returns `generated_text`
- **TTS (Text-to-Speech)**: converts `generated_text` → audio reply

### Automatic Conversation Tracking (Recommended)

The LLM Service automatically handles conversation continuity via **HTTP cookies**. Use `requests.Session()` to automatically manage cookies:

```python
import requests

s = requests.Session()  # Session automatically handles cookies

while True:
    # 1. Get latest user utterance from STT
    user_query = stt_transcribe_audio()  # returns text or None
    if user_query is None:
        break

    # 2. Call LLM Service /query
    # No need to manually track conversation_id - cookies handle it automatically!
    payload = {
        "survey_context": SURVEY_CONTEXT,
        "questions": QUESTION_LIST,
        "user_query": user_query,
        "continue_in_same_conversation": True,
        # conversation_id NOT needed - cookie from previous response is sent automatically
    }

    resp = s.post("http://localhost:8000/query", json=payload, timeout=30)
    data = resp.json()

    # 3. Get model reply (conversation_id is stored in cookie automatically)
    model_reply = data["generated_text"]
    conversation_id = data["conversation_id"]  # Optional: for logging/debugging only

    # 4. Send model reply to TTS for playback
    tts_play_audio(model_reply)
```

**How it works:**

- First request: Service creates a new `conversation_id` and sets it as an HTTP cookie
- Subsequent requests: `requests.Session()` automatically sends the cookie back
- No manual tracking needed: The service maintains conversation history server-side

### Manual Conversation Tracking (Alternative)

If you cannot use cookies (e.g., stateless HTTP client, microservice architecture), you can explicitly pass `conversation_id`:

```python
conversation_id = None

while True:
    user_query = stt_transcribe_audio()
    if user_query is None:
        break

    payload = {
        "survey_context": SURVEY_CONTEXT,
        "questions": QUESTION_LIST,
        "user_query": user_query,
        "continue_in_same_conversation": True,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id  # Explicit ID for stateless clients

    resp = requests.post("http://localhost:8000/query", json=payload, timeout=30)
    data = resp.json()

    conversation_id = data["conversation_id"]  # Must track this manually
    model_reply = data["generated_text"]
    tts_play_audio(model_reply)
```

**When to use manual tracking:**

- Stateless HTTP clients that don't support cookies
- Microservice architectures where cookies aren't shared
- Custom HTTP clients that don't handle cookies

**Recommendation:** Use automatic cookie-based tracking (Session approach) unless you have a specific reason not to. It's simpler and less error-prone.

---

In production, the STT and TTS components run in their own services; the LLM Service maintains conversation history in-memory using `conversation_id` (via cookies or explicit ID) to keep turns together.

## Documentation

- **[README.md](README.md)**: Main documentation (this file)
- **[LLM-SERVICE.md](LLM-SERVICE.md)**: LLM Service API documentation and voice integration guide
- **[Benchmark.md](Benchmark.md)**: Performance benchmarks and resource analysis
- **[docs/LlaMacppServer.md](docs/LlaMacppServer.md)**: Complete llama.cpp server reference (auto-generated)

## Project Structure

```
.
├── llm_service/              # LLM Service API wrapper (FastAPI)
│   ├── main.py              # Service implementation
│   └── requirements.txt     # Service dependencies
├── scripts/                 # Test and utility scripts
│   ├── docker_llm-api-service_inference.py
│   ├── llama-server_docker_inf.py
│   ├── llama-server_docker_inf_chat.py # Unused script, doesn't work for Gemma Chat Template
│   └── warmup.py            # Automated warmup script (runs via Docker Compose)
├── docs/                    # Documentation
│   └── LlaMacppServer.md    # llama.cpp server reference
├── compose.yaml             # Docker Compose configuration
├── Dockerfile.llm_service   # LLM Service container
├── Dockerfile.server        # Custom llama.cpp server (optional)
├── monitor.py               # Metrics monitoring helper (optional)
├── LOCUST_llama-server_docker_inf.py  # Load testing script
├── Benchmark.md             # Performance benchmarks
├── LLM-SERVICE.md           # Service API documentation
└── README.md                # This file
```
