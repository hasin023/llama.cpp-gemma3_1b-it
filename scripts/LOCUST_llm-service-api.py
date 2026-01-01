"""
LLM Service API Cost Analysis Load Test
=========================================
This load test is designed to produce detailed logs for cost analysis on
GPU (e.g., RunPod L40S) and CPU (e.g., AWS c8g) deployments.

Key Metrics Collected:
1. Tokens processed (prompt, completion, total) - for cost calculation
2. Inference time per request - for time-based cost calculation
3. Throughput (requests/sec, tokens/sec) - for capacity planning
4. Parallelism verification - for efficiency analysis

Usage (LOCAL testing against local Docker stack):
    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 1 -r 1 -t 30s --host http://localhost --system-config "i5-8365U CPU"
    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 1 -r 1 -t 30s --host http://localhost --system-config "Ryzen-5_3600_6-Core"

Usage (REMOTE testing against RunPod/AWS VM):
    # Replace <VM_IP> with your RunPod/AWS public IP address
    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://<VM_IP>

    # Examples:
    # RunPod:  python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://85.123.45.67
    # AWS:    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://ec2-12-34-56-78.compute-1.amazonaws.com

Note: All results (CSV, report) are saved to YOUR LOCAL machine in ./logs/
"""

from locust import HttpUser, task, between, events
import time
import json
import hashlib
import statistics
import csv
from threading import Lock
from datetime import datetime
from pathlib import Path


# ============================================================================
# COST ANALYSIS METRICS TRACKER
# ============================================================================
class CostMetrics:
    """Thread-safe metrics collector for cost analysis"""

    def __init__(self):
        self.lock = Lock()
        self.requests = []  # All successful request data
        self.failed_requests = []
        self.concurrent_count = 0
        self.peak_concurrent = 0
        
        # Token aggregates
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_all_tokens = 0
        
    def start_request(self):
        with self.lock:
            self.concurrent_count += 1
            current = self.concurrent_count
            self.peak_concurrent = max(self.peak_concurrent, current)
            return current

    def end_request(self):
        with self.lock:
            self.concurrent_count -= 1

    def record_success(self, data):
        with self.lock:
            self.requests.append(data)
            # Aggregate tokens
            usage = data.get("usage", {})
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)
            self.total_all_tokens += usage.get("total_tokens", 0)

    def record_failure(self, data):
        with self.lock:
            self.failed_requests.append(data)


metrics = CostMetrics()


# ============================================================================
# TEST DATA (Varied length prompts for realistic testing)
# ============================================================================
SURVEY_CONTEXT = (
    "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, Unimart, Shawpno "
    "এর মতো অনলাইন শপ ও সুপারশপ থেকে। এটি ১০০% প্রাকৃতিক উপাদান দিয়ে তৈরি।"
)

QUESTION_LIST = [
    "আপনি কি FrozenBerry ব্যবহার করেছেন?",
    "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
    "সংরক্ষণ করা সহজ কি?",
]

# Different conversation states to simulate realistic load
CONVERSATION_SCENARIOS = [
    # Scenario 1: Short conversation (2 turns)
    {
        "initial_model_message": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?",
        "user_query": "জি বলছি।"
    },
    # Scenario 2: Medium conversation (4+ turns worth of context)
    {
        "initial_model_message": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?",
        "user_query": "জি, আমি FrozenBerry ব্যবহার করেছি। স্বাদ খুব ভালো ছিল।"
    },
    # Scenario 3: Longer user input
    {
        "initial_model_message": "শুভ সন্ধ্যা! আমি FrozenBerry এর পক্ষ থেকে কিছু প্রশ্ন করতে চাইছিলাম।",
        "user_query": "আমি গত মাসে FrozenBerry কিনেছিলাম Daraz থেকে। পণ্যের মান বেশ ভালো ছিল, তবে ডেলিভারি একটু দেরি হয়েছিল।"
    },
]


# Multi-turn conversation user queries (simulates realistic voice agent flow)
MULTI_TURN_QUERIES = [
    "জি বলছি।",
    "হ্যাঁ, ব্যবহার করেছি।",
    "স্বাদ অনেক ভালো ছিল।",
    "হ্যাঁ, সংরক্ষণ করা সহজ।",
    "ধন্যবাদ।",
]


# ============================================================================
# LOCUST USER
# ============================================================================
class LLMServiceUser(HttpUser):
    """
    Load test user for the llm-service-api (FastAPI wrapper).
    Uses the /query endpoint with conversation management.
    
    Supports two test modes:
    1. Single-shot requests (for isolated cost measurement)
    2. Multi-turn conversations (for realistic voice agent simulation)
    """
    host = "http://localhost"
    wait_time = between(0.1, 0.5)  # Simulate realistic user delays
    
    def on_start(self):
        """Initialize user state for multi-turn conversations"""
        self.conversation_id = None  # Will be set after first request
        self.turn_count = 0
    
    @task(3)  # Weight 3: Run single-shot more often for cost measurement
    def single_shot_query(self):
        """
        Single independent request (no conversation continuation).
        Best for isolated cost-per-request measurement.
        """
        concurrent_at_start = metrics.start_request()
        
        request_id = hashlib.md5(f"{time.time()}-{id(self)}-single".encode()).hexdigest()[:8]
        start_time = time.time()
        
        # Rotate through scenarios for varied prompt lengths
        scenario_idx = hash(request_id) % len(CONVERSATION_SCENARIOS)
        scenario = CONVERSATION_SCENARIOS[scenario_idx]
        
        payload = {
            "survey_context": SURVEY_CONTEXT,
            "questions": QUESTION_LIST,
            "initial_model_message": scenario["initial_model_message"],
            "user_query": scenario["user_query"],
            "continue_in_same_conversation": False,  # Independent request
            "max_tokens": 256,
        }

        self._execute_request(request_id, start_time, concurrent_at_start, payload, "single_shot", scenario_idx)
    
    @task(1)  # Weight 1: Run multi-turn less often
    def multi_turn_query(self):
        """
        Multi-turn conversation simulation (continues conversation).
        Simulates real voice agent workflow with growing context.
        """
        concurrent_at_start = metrics.start_request()
        
        request_id = hashlib.md5(f"{time.time()}-{id(self)}-multi".encode()).hexdigest()[:8]
        start_time = time.time()
        
        # First turn: start new conversation
        if self.turn_count == 0 or self.conversation_id is None:
            user_query = MULTI_TURN_QUERIES[0]
            payload = {
                "survey_context": SURVEY_CONTEXT,
                "questions": QUESTION_LIST,
                "initial_model_message": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?",
                "user_query": user_query,
                "continue_in_same_conversation": False,  # Start new
                "max_tokens": 256,
            }
        else:
            # Subsequent turns: continue conversation
            query_idx = min(self.turn_count, len(MULTI_TURN_QUERIES) - 1)
            user_query = MULTI_TURN_QUERIES[query_idx]
            payload = {
                "survey_context": SURVEY_CONTEXT,
                "questions": QUESTION_LIST,
                "user_query": user_query,
                "continue_in_same_conversation": True,
                "conversation_id": self.conversation_id,  # Explicitly pass ID
                "max_tokens": 256,
            }
        
        # Execute and track conversation_id for next turn
        new_cid = self._execute_request(request_id, start_time, concurrent_at_start, payload, "multi_turn", self.turn_count)
        
        if new_cid:
            self.conversation_id = new_cid
            self.turn_count += 1
            
            # Reset after completing a full conversation (5 turns)
            if self.turn_count >= len(MULTI_TURN_QUERIES):
                self.conversation_id = None
                self.turn_count = 0
    
    def _execute_request(self, request_id, start_time, concurrent_at_start, payload, request_type, scenario_idx):
        """
        Common request execution logic.
        Returns conversation_id if successful, None otherwise.
        """
        inference_time_reported = 0
        conversation_id = None
        
        try:
            response = self.client.post(
                "/query",
                json=payload,
                name=f"llm_service_{request_type}",
                timeout=120
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            success = False
            generated_text = ""
            usage = {}
            
            if response.status_code == 200:
                try:
                    resp_json = response.json()
                    generated_text = resp_json.get("generated_text", "")
                    usage = resp_json.get("usage", {})
                    inference_time_reported = resp_json.get("inference_time", 0)
                    conversation_id = resp_json.get("conversation_id", None)
                    model_name = resp_json.get("model_name", "unknown")
                    
                    if generated_text and len(generated_text.strip()) > 5:
                        success = True
                        
                except Exception as e:
                    print(f"Response parse error: {e}")
            
            request_data = {
                "request_id": request_id,
                "request_type": request_type,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_total": elapsed,
                "inference_time_reported": inference_time_reported if success else 0,
                "concurrent_at_start": concurrent_at_start,
                "success": success,
                "generated_text_preview": generated_text[:100] if generated_text else "",
                "text_length": len(generated_text) if generated_text else 0,
                "usage": usage,
                "status_code": response.status_code,
                "scenario_idx": scenario_idx,
                "conversation_id": conversation_id,
                "model_name": model_name,
            }
            
            if success:
                metrics.record_success(request_data)
            else:
                metrics.record_failure(request_data)
                
            return conversation_id if success else None

        except Exception as e:
            end_time = time.time()
            metrics.record_failure({
                "request_id": request_id,
                "request_type": request_type,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_total": end_time - start_time,
                "concurrent_at_start": concurrent_at_start,
                "success": False,
                "error": str(e),
                "status_code": 0
            })
            print(f"Request {request_id} failed: {e}")
            return None
        
        finally:
            metrics.end_request()


# ============================================================================
# COST CALCULATION FUNCTIONS
# ============================================================================
def get_blended_api_price(model_name, total_prompts, total_gen, total):
    """Calculate blended API price based on input/output ratio"""
    prices = GEMINI_MODELS.get(model_name)
    if not prices or total == 0:
        return 0.0
    return ((total_prompts * prices["input"]) + (total_gen * prices["output"])) / total

def calculate_effective_cost_per_million_tokens(total_tokens, duration_seconds, hourly_rate):
    """
    Calculate the effective cost per 1M tokens for self-hosted hardware.
    
    Args:
        total_tokens: Total tokens processed during the test
        duration_seconds: Total wall-clock duration of the test in seconds
        hourly_rate: Cost of the hardware per hour (e.g., $0.79 for RunPod L40S)
        
    Returns:
        Cost per 1 million tokens
    """
    if total_tokens == 0:
        return 0.0
    
    # Time cost for processing these tokens (based on wall-clock time rented)
    time_cost = (duration_seconds / 3600) * hourly_rate
    
    # Scale to 1 million tokens
    cost_per_million = (time_cost / total_tokens) * 1_000_000
    
    return cost_per_million


def calculate_tokens_per_second(total_tokens, total_inference_time):
    """Calculate tokens per second throughput"""
    if total_inference_time == 0:
        return 0.0
    return total_tokens / total_inference_time


# ============================================================================
# CUSTOM ARGUMENTS
# ============================================================================
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument(
        "--system-config", 
        type=str, 
        env_var="LOCUST_SYSTEM_CONFIG", 
        default="unknown_system", 
        help="Description of system under test (e.g. 'RunPod L40S', 'AWS c8g.xlarge')"
    )

# ============================================================================
# CONFIGURATION & PRICING
# ============================================================================
# Cloud Compute Costs ($/hr)
PRICING_COMPUTE = {
    "RunPod 3090 (24GB VRAM)": 0.46,
    "RunPod 4090 (24GB VRAM)": 0.59,
    "RunPod 5090 (32GB VRAM)": 0.89,
    "RunPod A40 (48GB VRAM)": 0.40,
    "AWS t4g.xlarge (4vCPU)": 0.0899,
    "AWS c8g.xlarge (4vCPU)": 0.1052,
    "AWS c8g.2xlarge (8vCPU)": 0.2103,
    "AWS c8g.4xlarge (16vCPU)": 0.4206,
}

# Disk Storage Costs ($/hr)
# RunPod: $0.006/hr (20GB), AWS: ~$0.00333/hr (30GB gp3)
PRICING_STORAGE = {
    "RunPod 3090 (24GB VRAM)": 0.006,
    "RunPod 4090 (24GB VRAM)": 0.006,
    "RunPod 5090 (32GB VRAM)": 0.006,
    "RunPod A40 (48GB VRAM)": 0.006,
    "AWS t4g.xlarge (4vCPU)": 0.00333,
    "AWS c8g.xlarge (4vCPU)": 0.00333,
    "AWS c8g.2xlarge (8vCPU)": 0.00333,
    "AWS c8g.4xlarge (16vCPU)": 0.00333,
    "AWS c8g.8xlarge (32vCPU)": 0.00333,
}

# Gemini API Pricing (Reference) - $/1M tokens
GEMINI_MODELS = {
    "Gemini 2.5 Flash":      {"input": 0.30, "output": 2.50},
    "Gemini 3 Flash Prev":   {"input": 0.50, "output": 3.00},
    "Gemini 2.5 Pro":        {"input": 1.25, "output": 10.00},
    "Gemini 3 Pro Prev":     {"input": 2.00, "output": 12.00},
}


# ============================================================================
# DATA CALCULATION HELPER
# ============================================================================
def calculate_cost_data(metrics, duration, successful, system_config):
    """
    Calculate all cost metrics and return a structured dictionary.
    No printing happens here.
    """
    data = {
        "sufficient_data": False,
        "api_benchmarks": [],
        "actual_system": None,
        "hypothetical_comparisons": [],
        "best_hypothetical": None,
        "baseline_api": None,  # Will store Gemini 2.5 Flash data
        "metrics": {
            "total_requests": successful,
            "total_tokens": metrics.total_all_tokens,
            "duration": duration,
            "requests_per_m_tokens": 0
        }
    }

    if successful == 0 or metrics.total_all_tokens == 0:
        return data

    data["sufficient_data"] = True
    requests_per_m = (successful / metrics.total_all_tokens * 1_000_000)
    data["metrics"]["requests_per_m_tokens"] = requests_per_m

    # 1. API Benchmarks
    for model_name in GEMINI_MODELS:
        blended = get_blended_api_price(model_name, metrics.total_prompt_tokens, metrics.total_completion_tokens, metrics.total_all_tokens)
        per_req = blended / requests_per_m
        
        entry = {
            "name": model_name,
            "cost_per_1m": blended,
            "cost_per_req": per_req
        }
        data["api_benchmarks"].append(entry)
        
        if model_name == "Gemini 2.5 Flash":
            data["baseline_api"] = entry

    # 2. Actual System Cost
    matched_config = None
    for key in PRICING_COMPUTE.keys():
        if key.lower() in system_config.lower() or system_config.lower() in key.lower():
            matched_config = key
            break
            
    if matched_config:
        hourly_compute_rate = PRICING_COMPUTE[matched_config]
        disk_cost = PRICING_STORAGE.get(matched_config, 0.0)
        total_hourly = hourly_compute_rate + disk_cost
        
        actual_system_cost = calculate_effective_cost_per_million_tokens(
            metrics.total_all_tokens, duration, total_hourly
        )
        actual_cost_per_req = (duration / 3600) * total_hourly / successful
        
        data["actual_system"] = {
            "name": matched_config,
            "cost_per_1m": actual_system_cost,
            "cost_per_req": actual_cost_per_req,
            "hourly_rate": total_hourly
        }

    # 3. Hypothetical Comparison
    cheapest_cost = float('inf')
    
    for name, hourly_compute_rate in PRICING_COMPUTE.items():
        disk_cost = PRICING_STORAGE.get(name, 0.0)
        total_hourly = hourly_compute_rate + disk_cost
        
        cost_per_million = calculate_effective_cost_per_million_tokens(
            metrics.total_all_tokens, duration, total_hourly
        )
        cost_per_req = (duration / 3600) * total_hourly / successful
        
        entry = {
            "name": name,
            "cost_per_1m": cost_per_million,
            "cost_per_req": cost_per_req
        }
        data["hypothetical_comparisons"].append(entry)

        if cost_per_million < cheapest_cost:
            cheapest_cost = cost_per_million
            data["best_hypothetical"] = entry

    return data


# ============================================================================
# REPORTING HELPERS (Markdown Format)
# ============================================================================
def print_professional_report(printer, config, metrics, cost_data, timing_data):
    """
    Generates a professional executive-style report.
    Structure:
    1. Executive Summary (Verdict & Savings)
    2. Performance Snapshot (KPIs)
    3. Detailed Cost Analysis
    4. Technical Deep Dive (Latency, Tokens)
    5. Test Metadata
    """
    
    # -------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & VERDICT
    # -------------------------------------------------------------------------
    printer("# LLM Service Cost Analysis Report\n")
    printer(f"- **Test Date:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    printer(f"- **System:** {config.get('system_config', 'Unknown')}")
    printer(f"- **Model:** {config.get('model_name', 'Unknown')}\n")
    
    printer("## 1. Executive Summary\n")
    
    if not cost_data["sufficient_data"] or not cost_data["baseline_api"]:
        printer("> **Insufficient Data for Verdict**\n")
    else:
        baseline = cost_data["baseline_api"]
        baseline_cost = baseline["cost_per_1m"]
        
        # Determine comparison target (Actual System OR Best Hypothetical)
        target = cost_data["actual_system"] if cost_data["actual_system"] else cost_data["best_hypothetical"]
        target_name = target["name"]
        target_cost = target["cost_per_1m"]
        is_simulation = cost_data["actual_system"] is None

        # Calculate Savings/Cost
        if target_cost < baseline_cost:
            savings_pct = (1 - (target_cost / baseline_cost)) * 100
            diff_per_1m = baseline_cost - target_cost
            monthly_savings_proj = diff_per_1m * 5 # Projected for 5M tokens/month
            
            verdict_type = "SIMULATION VERDICT" if is_simulation else "VERDICT"
            
            printer(f"### {verdict_type}: SELF-HOSTING IS CHEAPER")
            printer(f"**Recommendation:** Run on **{target_name}**")
            printer(f"- **{savings_pct:.1f}% Savings** vs Gemini 2.5 Flash API")
            printer(f"- Saves **${diff_per_1m:.2f}** per 1 Million tokens")
            printer(f"- *Est. Monthly Savings (at 5M tokens/mo):* **${monthly_savings_proj:.2f}**\n")
        
        else:
            extra_cost_pct = ((target_cost / baseline_cost) - 1) * 100
            printer(f"### VERDICT: API IS CHEAPER")
            printer(f"**Recommendation:** Use **Gemini 2.5 Flash API**")
            printer(f"- Self-hosting on {target_name} is **{extra_cost_pct:.1f}% more expensive**")
            printer(f"- API Cost: ${baseline_cost:.4f} / 1M tokens vs Self-Host: ${target_cost:.4f}\n")

    # -------------------------------------------------------------------------
    # 2. PERFORMANCE SNAPSHOT
    # -------------------------------------------------------------------------
    printer("## 2. Performance Snapshot\n")
    
    total_reqs = cost_data["metrics"]["total_requests"]
    duration = cost_data["metrics"]["duration"]
    
    tps = cost_data["metrics"]["total_tokens"] / timing_data["total_inference_time"] if timing_data["total_inference_time"] > 0 else 0
    avg_latency = timing_data["avg_inference"]
    p50_latency = timing_data["p50_inference"]
    
    printer("| Key Metric | Value | Reference |")
    printer("|------------|-------|-----------|")
    printer(f"| **Throughput (TPS)** | **{tps:.2f}** tokens/s | Speed of generation |")
    printer(f"| **Avg Latency** | {avg_latency:.2f} s | Time to first response |")
    printer(f"| **Median Latency** | {p50_latency:.2f} s | Typical user wait time |")
    printer(f"| **Request Rate** | {total_reqs/duration:.2f} req/s | System capacity |")
    printer("")

    # -------------------------------------------------------------------------
    # 3. DETAILED COST ANALYSIS
    # -------------------------------------------------------------------------
    printer("## 3. Cost Analysis Comparison\n")
    
    if cost_data["sufficient_data"]:
        printer("| Provider / System | Cost per 1M Tokens | Cost per Request |")
        printer("|-------------------|--------------------|------------------|")
        
        # 1. API Line
        b = cost_data["baseline_api"]
        printer(f"| **Gemini 2.5 Flash** (Baseline) | **${b['cost_per_1m']:.4f}** | ${b['cost_per_req']:.6f} |")
        
        # 2. Actual System Line
        if cost_data["actual_system"]:
            a = cost_data["actual_system"]
            printer(f"| **{a['name']}** (Actual) | **${a['cost_per_1m']:.4f}** | ${a['cost_per_req']:.6f} |")
            
        # 3. Best Hypothetical Line
        actual_sys = cost_data.get("actual_system")
        actual_name = actual_sys["name"] if actual_sys else None
        
        if cost_data["best_hypothetical"] and cost_data["best_hypothetical"]["name"] != actual_name:
            h = cost_data["best_hypothetical"]
            printer(f"| **{h['name']}** (Best Option) | **${h['cost_per_1m']:.4f}** | ${h['cost_per_req']:.6f} |")
            
        printer("")
        
        # Break Even Analysis
        if cost_data["baseline_api"]:
            target = cost_data["actual_system"] if cost_data["actual_system"] else cost_data["best_hypothetical"]
            target_name = target["name"]
            
            # Recalculate fixed costs for break-even
            target_hourly_rate = 0
            if cost_data["actual_system"]:
                target_hourly_rate = cost_data["actual_system"]["hourly_rate"]
            else: 
                # Re-fetch for simulation
                base_rate = PRICING_COMPUTE.get(target_name, 0)
                disk = PRICING_STORAGE.get(target_name, 0)
                target_hourly_rate = base_rate + disk
                
            daily_fixed_cost = target_hourly_rate * 24
            api_cost_per_token = cost_data["baseline_api"]["cost_per_1m"] / 1_000_000
            
            if api_cost_per_token > 0:
                break_even_tokens = daily_fixed_cost / api_cost_per_token
                break_even_reqs = daily_fixed_cost / cost_data["baseline_api"]["cost_per_req"]
                
                # Calculate hours needed based on TPS
                tps = cost_data["metrics"]["total_tokens"] / timing_data["total_inference_time"] if timing_data["total_inference_time"] > 0 else 0
                daily_hours_needed = 0
                if tps > 0:
                    daily_hours_needed = break_even_tokens / (tps * 3600)

                printer(f"### Break-Even Point for {target_name}")
                printer("> Activity level required to justify monthly fixed costs over paying per-token API fees.\n")
                printer("| Metric | Daily Requirement | Monthly Requirement |")
                printer("|--------|-------------------|---------------------|")
                printer(f"| **Tokens Generated** | {break_even_tokens/1_000_000:.2f} M | {break_even_tokens*30/1_000_000:.2f} M |")
                printer(f"| **Total Requests** | {break_even_reqs:,.0f} | {break_even_reqs*30:,.0f} |")
                printer(f"| **Hours of Traffic** | {daily_hours_needed:.2f} hours | {daily_hours_needed*30:.2f} hours |")
                printer("")

    # -------------------------------------------------------------------------
    # 4. TECHNICAL DEEP DIVE
    # -------------------------------------------------------------------------
    printer("## 4. Technical Deep Dive\n")
    
    printer("### Total Token Volume")
    printer(f"- **Total Tokens:** {metrics.total_all_tokens:,}")
    printer(f"- **Prompt Tokens:** {metrics.total_prompt_tokens:,}")
    printer(f"- **Completion Tokens:** {metrics.total_completion_tokens:,}")
    
    total_reqs = cost_data["metrics"]["total_requests"]
    if total_reqs > 0:
        avg_prompt = metrics.total_prompt_tokens / total_reqs
        avg_compl = metrics.total_completion_tokens / total_reqs
        printer(f"- **Avg Context Window:** {avg_prompt:.0f} prompt + {avg_compl:.0f} completion = {avg_prompt+avg_compl:.0f} tokens/req\n")

    printer("### Latency Distribution")
    printer("| Metric | Time (s) |")
    printer("|--------|----------|")
    printer(f"| Min | {timing_data['min_inference']:.3f} |")
    printer(f"| Median | {timing_data['p50_inference']:.3f} |")
    printer(f"| Avg | {timing_data['avg_inference']:.3f} |")
    printer(f"| Max | {timing_data['max_inference']:.3f} |")
    printer("")

    # -------------------------------------------------------------------------
    # 5. TEST METADATA
    # -------------------------------------------------------------------------
    printer("## 5. Test Metadata\n")
    printer("| Parameter | Value |")
    printer("|-----------|-------|")
    printer(f"| Timestamp | {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')} |")
    printer(f"| Duration | {duration:.2f}s |")
    printer(f"| Users | {config.get('users')} |")
    printer(f"| Spawn Rate | {config.get('spawn_rate')} |")
    printer(f"| Host | {config.get('host')} |")


# ============================================================================
# CSV EXPORT HELPER
# ============================================================================
def export_csv_data(printer, successful_requests, system_config):    
    # Sanitize system config for filename (replace non-alphanumeric with underscore)
    safe_config = "".join(c if c.isalnum() else "_" for c in system_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = Path(f"./logs/request_logs_{safe_config}_{timestamp}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "request_type", "model_name", "start_time", "inference_time", "elapsed_total",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "concurrent_at_start", "success", "status_code", "system_config"
            ])
            for req in successful_requests:
                usage = req.get("usage", {})
                writer.writerow([
                    req["request_id"], req["request_type"], req.get("model_name", "unknown"), req["start_time"], req.get("inference_time_reported", 0),
                    req["elapsed_total"], usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0), usage.get("total_tokens", 0),
                    req["concurrent_at_start"], req["success"], req["status_code"], system_config
                ])
    except Exception as e:
        printer(f"   ✗ Failed to export CSV: {e}")


# ============================================================================
# MAIN REPORTING CONTROLLER
# ============================================================================
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate comprehensive cost analysis report when test stops"""
    
    system_config = environment.parsed_options.system_config
    all_requests = metrics.requests + metrics.failed_requests
    successful_requests = metrics.requests
    
    # 1. Gather Timing Data explicitly
    inference_times = [r.get("inference_time_reported", 0) for r in successful_requests if r.get("inference_time_reported", 0) > 0]
    total_inference_time = sum(inference_times)
    
    timing_data = {
        "total_inference_time": total_inference_time,
        "avg_inference": statistics.mean(inference_times) if inference_times else 0,
        "p50_inference": statistics.median(inference_times) if inference_times else 0,
        "min_inference": min(inference_times) if inference_times else 0,
        "max_inference": max(inference_times) if inference_times else 0
    }

    # Prepare log file
    safe_config = "".join(c if c.isalnum() else "_" for c in system_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"./logs/report__{safe_config}_{timestamp}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define a printer that writes to both stdout and the file
    def printer(msg):
        print(msg)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Gather extended config
    test_start = min((r["start_time"] for r in all_requests), default=time.time())
    test_end = max((r["end_time"] for r in all_requests), default=time.time())
    duration = test_end - test_start
    
    # Try to find model name from successful requests
    model_name = "Unknown"
    for req in successful_requests:
        if req.get("model_name") and req.get("model_name") != "unknown":
            model_name = req.get("model_name")
            break

    config = {
        "system_config": system_config,
        "model_name": model_name,
        "duration": duration,
        "users": getattr(environment.parsed_options, 'num_users', "N/A"),
        "spawn_rate": getattr(environment.parsed_options, 'spawn_rate', "N/A"),
        "host": environment.host
    }
    
    # 2. Calculate Costs
    cost_data = calculate_cost_data(metrics, duration, len(successful_requests), system_config)
    
    # 3. Print Professional Report
    print_professional_report(printer, config, metrics, cost_data, timing_data)
    
    # 4. Export CSV
    export_csv_data(printer, successful_requests, system_config)

    

# Optional: Real-time progress updates
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Real-time request logging"""
    if not exception:
        print(f"[{time.strftime('%H:%M:%S')}] ✓ {name}: {response_time:.2f}ms")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ {name} failed: {exception}")
