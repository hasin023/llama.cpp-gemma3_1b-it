# LLM Inference Cost Analysis

This document outlines the strategy for measuring and calculating the _true_ cost of self-hosted LLM inference on RunPod (GPU) and AWS (CPU).

## 1. Core Metrics & Cost Driver

Unlike managed APIs (paid per token), self-hosting is paid per **Time** (Compute + Disk).

| Metric             | Definition                                  | Impact                                                   |
| :----------------- | :------------------------------------------ | :------------------------------------------------------- |
| **Inference Time** | Duration the request occupies the hardware. | **Primary Cost Driver** (Time = Money).                  |
| **Tokens**         | `prompt_tokens` + `completion_tokens`.      | Usage metric. Used to calculate "Effective Cost per 1M". |
| **Throughput**     | Tokens Per Second (TPS).                    | Hardware efficiency metric.                              |

## 2. Cost Formulas

To compare against APIs like OpenAI, we convert "Time Cost" into "Token Cost".

**A. Total Hourly Rate**
$$ \text{Total Rate} = \text{Compute}_{\text{/hr}} + \text{Disk Storage}_{\text{/hr}} $$

> **Note**: We include **$0.006/hr** (RunPod) and **~$0.0022/hr** (AWS) for mandatory disk storage.

**B. Cost Per Request**
$$ \text{Cost} = \frac{\text{Inference Time (s)}}{3600} \times \text{Total Rate} $$

**C. Effective Cost Per 1M Tokens**
$$ \text{Cost per 1M} = \frac{\text{Total Rate}}{\text{Tokens Processed per Hour}} \times 1,000,000 $$

## 3. Benchmarking Strategy (GPU vs CPU)

Run two separate load tests to find the best Price/Performance ratio.

### Setup 1: RunPod (GPU Mode)

- **Target**: High Throughput, Low Latency.
- **Hardware**: NVIDIA 3090 / 4090 / A100.
- **Config**: In `compose.yaml`, **Enable MODE B** (GPU), disable Mode A.

### Setup 2: AWS (CPU Mode)

- **Target**: Cost Efficiency for low-traffic background tasks.
- **Hardware**: Graviton `c8g` series.
- **Config**: In `compose.yaml`, **Enable MODE A** (CPU), disable Mode B.

## 4. Execution Guide

We use a local `locust` script to drive traffic to the remote VM and aggregate logs locally.

### Step 1: Deploy

SSH into your remote VM (RunPod or AWS) and start the service:

```bash
docker compose up -d
```

### Step 2: Run Load Test (From Local Machine)

Execute the script from your development machine, pointing to the remote IP.

```bash
# RunPod Example (GPU)
locust -f scripts/LOCUST_llm-service-api.py --headless -u 10 -r 2 -t 120s --host http://<RUNPOD_IP>:8000

# AWS Example (CPU)
locust -f scripts/LOCUST_llm-service-api.py --headless -u 10 -r 2 -t 120s --host http://<AWS_IP>:8000
```

### Step 3: Analyze

The script generates a report in your terminal and saves a detailed CSV locally:

```bash
logs/cost_analysis_YYYYMMDD_HHMMSS.csv
```

**Look for:** "Effective Cost per 1M Tokens" in the terminal output to compare directly vs OpenAI/Anthropic.
