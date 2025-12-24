# Llama.cpp Server Performance Benchmark & Resource Analysis

**Date:** 2025-12-24
**Subject:** Comparative Analysis of 1B Model on different Hardware

## 1. Executive Summary

This document benchmarks the `gemma3-1b-ft` model running on the `llama.cpp` server (Docker) across two distinct hardware profiles: a high-performance workstation (Ryzen 5) and a constrained edge-proxy device (Intel i5-8365U).

**Key Findings:**

- **Parallelism:** The server successfully handles concurrent requests (2 slots) with 100% success rate on both platforms when resources are sufficient.
- **Quantization Impact:** Shifting from Q4 (4-bit) to Q8 (8-bit) quantization increases memory pressure significantly.
- **Memory Wall:** On constrained memory setups (1GB limit), the Q8 model causes severe **memory thrashing** (swapping), reducing throughput by **~94%** (1.10 req/s $\to$ 0.07 req/s).
- **Resolution:** Increasing the memory allocation to 2GB eliminates the bottleneck, restoring throughput to acceptable levels (0.71 req/s) for the Q8 model.

---

## 2. Test Environment Specifications

### 2.1 Software Stack

- **Engine:** `llama.cpp` server (Docker: `ghcr.io/ggml-org/llama.cpp:server`)
- **Load Testing Tool:** Locust (Headless)
  - _Users:_ 50 concurrent
  - _Spawn Rate:_ 17 users/sec
  - _Duration:_ 120 seconds
- **Endpoint:** `http://localhost:8080` (2 parallel slots)

### 2.2 Hardware Profiles

| Feature           | System A                    | System B                      |
| :---------------- | :-------------------------- | :---------------------------- |
| **CPU**           | AMD Ryzen 5 3600 (6 Cores)  | Intel Core i5-8365U (4 Cores) |
| **Clock Speed**   | 4.2 - 4.4 GHz (Overclocked) | 1.60 GHz (Base)               |
| **OS**            | Windows 10                  | Windows 11                    |
| **Power Profile** | High Performance            | Balanced / Mobile             |

---

## 3. Benchmark Results

### 3.1 Scenario A: Baseline Performance (Q4 Model)

_Model:_ `gemma3-1b-ft-Q4_K_M.gguf` (~700MB)
_Constraint:_ Default (1GB RAM Limit)

| Metric              | System A (Ryzen 5)     | System B (Intel i5)    | Difference |
| :------------------ | :--------------------- | :--------------------- | :--------- |
| **Total Requests**  | 311                    | 116                    | -62.7%     |
| **Throughput**      | **2.61 req/s**         | **1.10 req/s**         | -57.8%     |
| **Avg Latency**     | 4.11 s                 | 7.03 s                 | +71.0%     |
| **Min Latency**     | 0.673 s                | 2.673 s                | \_         |
| **Max Latency**     | 1.910 s                | 12.910 s               | \_         |
| **CPU Utilization** | 200% (Full Saturation) | 200% (Full Saturation) | Equal      |

**Analysis:**
The Q4 model fits comfortably within the 1GB memory limit. Both systems achieve optimal CPU saturation (200% for 2 slots). The throughput difference highlights the raw compute gap between the overclocked desktop CPU and the low-power mobile CPU.

### 3.2 Scenario B: The Memory Bottleneck (Q8 Model)

_Model:_ `gemma3-1b-ft-Q8_0.gguf` (~1.2GB)
_Hardware:_ System B (Intel i5)

| Metric              | Test 1: 1GB Limit (Def) | Test 2: 2GB Limit (Fix) | Impact of Fix |
| :------------------ | :---------------------- | :---------------------- | :------------ |
| **Total Requests**  | 8                       | 83                      | **+937%**     |
| **Throughput**      | **0.07 req/s**          | **0.71 req/s**          | **+914%**     |
| **Avg Latency**     | 71.15 s                 | 9.56 s                  | -86.5%        |
| **CPU Utilization** | 40-70% (I/O Wait)       | 200% (Full Saturation)  | Optimization  |

**Deep Dive Analysis:**

- **Test 1 (Failure State):** With a 1GB limit, the 1.2GB Q8 model forces the OS to use disk swap. The CPU spends 30-60% of its time waiting for memory pages to be fetched from disk (I/O Wait), resulting in catastrophic performance degradation.
- **Test 2 (Success State):** Relaxing the limit to 2GB allows the entire model + KV cache to reside in RAM. The CPU bottleneck returns (200% usage), and throughput represents the true compute capability of the i5 for 8-bit inference.

### 3.3 Scenario C: The Middle Ground (Q5 Model)

_Model:_ `gemma3-1b-ft-Q5_K_M.gguf`
_Hardware:_ System B (Intel i5)
_Constraint:_ Default (1GB RAM Limit)

| Metric              | System B (Intel i5)    |
| :------------------ | :--------------------- |
| **Total Requests**  | 120                    |
| **Throughput**      | **1.01 req/s**         |
| **Avg Latency**     | 6.73 s                 |
| **Min Latency**     | 0.873 s                |
| **Max Latency**     | 2.910 s                |
| **CPU Utilization** | 200% (Full Saturation) |

**Analysis:**
The Q5 model fits within the 1GB limit, showing performance very close to the Q4 model (1.01 req/s vs 1.10 req/s). This confirms that Q5 is a viable option for 1GB devices if slightly higher precision is needed without hitting the memory wall.

---

### 3.4 Scenario D: The Optimized (Q4 Model)

_Model:_ `gemma3-1b-ft-Q4_K_M.gguf`
_Hardware:_ System B (Intel i5)
_Constraint:_ No RAM Limit

| Metric              | System B (Intel i5)                |
| :------------------ | :--------------------------------- |
| **Total Requests**  | 55 (in 60 seconds)                 |
| **Throughput**      | **1.17 req/s**                     |
| **Avg Latency**     | 1.017 s                            |
| **Min Latency**     | 0.673 s                            |
| **Max Latency**     | 1.910 s                            |
| **CPU Utilization** | 400% (Full Saturation) (4 Threads) |

---

## 4. Technical Conclusion

1.  **Memory Hard Limits are Critical:** For 1B parameter models, a **1GB RAM** allocation is sufficient _only_ for **Q4 (4-bit)** quantization.
2.  **Q8 Requires >1GB:** Running **Q8 (8-bit)** models requires at least **1.5GB - 2GB** of dedicated RAM to avoid thrashing.
3.  **Throughput vs. Precision:**
    - Q4 Model: 1.10 req/s (Fastest)
    - Q8 Model: 0.71 req/s (~35% slower due to compute intensity)
    - _Note: This 35% drop is expected due to the higher computational cost of processing 8-bit weights vs 4-bit weights._

### 4.1 Analysis: 2GB Memory for Q4/Q5?

**Question:** _If we allocate 2GB memory to Q4 or Q5 scenarios, will performance improve?_

**Answer: No/Negligible.**

- **Reasoning:** The performance cliff seen in Q8 is due to **swapping** (data moving between RAM and Disk).
- **Q4 & Q5 status:** Both models (~700MB - 900MB) already fit entirely within the 1GB allocation.
- **Impact:** Increasing memory to 2GB provides **zero additional throughput** because the model is already running at memory-speed (RAM), not disk-speed. You cannot go faster than "Pure RAM."
- **Use Case:** The only benefit of 2GB for Q4/Q5 would be if you wanted to drastically increase the **Context Window** (e.g., from 8192 to 16384 tokens) or **Batch Size**, which consumes extra VRAM/RAM. for the _same_ workload, it yields no gain.

## 5. Recommendations

- **For 1GB RAM Devices (e.g., Orange Pi Zero 2):** STRICTLY use **Q4_K_M** or **Q5_K_M** quantization. Do not attempt Q8.
- **For 2GB+ RAM Devices:** Q8 is viable but offers ~35% lower throughput than Q4. Use only if higher precision is strictly necessary for the application logic.
- **Production Config:** Ensure `compose.yaml` memory limits always exceed `Model Size + (Context Size * 2 * Layers)` to prevent silent performance death due to swapping.
