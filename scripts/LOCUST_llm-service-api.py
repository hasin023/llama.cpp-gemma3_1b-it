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
    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 1 -r 1 -t 120s --host http://localhost:8000 --system-config "i5-8365U CPU"
    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 1 -r 1 -t 1800s --host http://localhost:8000 --system-config "Ryzen-5_3600_6-Core"

Usage (REMOTE testing against RunPod/AWS VM):
    # Replace <VM_IP> with your RunPod/AWS public IP address
    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://<VM_IP>:8000

    # Examples:
    # RunPod:  python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://85.123.45.67:8000
    # AWS:    python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://ec2-12-34-56-78.compute-1.amazonaws.com:8000

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
    host = "http://localhost:8000"
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
def calculate_effective_cost_per_million_tokens(total_tokens, total_inference_time, hourly_rate):
    """
    Calculate the effective cost per 1M tokens for self-hosted hardware.
    
    Args:
        total_tokens: Total tokens processed during the test
        total_inference_time: Total inference time in seconds
        hourly_rate: Cost of the hardware per hour (e.g., $0.79 for RunPod L40S)
        
    Returns:
        Cost per 1 million tokens
    """
    if total_tokens == 0:
        return 0.0
    
    # Time cost for processing these tokens
    time_cost = (total_inference_time / 3600) * hourly_rate
    
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
    "RunPod A100 (80GB VRAM)": 1.39,
    "AWS c8g.xlarge (4vCPU)": 0.1052,
    "AWS c8g.2xlarge (8vCPU)": 0.2103,
    "AWS c8g.4xlarge (16vCPU)": 0.4206,
}

# Disk Storage Costs ($/hr)
# RunPod: $0.006/hr (20GB), AWS: ~$0.0022/hr (20GB gp3)
PRICING_STORAGE = {
    "RunPod 3090 (24GB VRAM)": 0.006,
    "RunPod 4090 (24GB VRAM)": 0.006,
    "RunPod 5090 (32GB VRAM)": 0.006,
    "RunPod A40 (48GB VRAM)": 0.006,
    "RunPod A100 (80GB VRAM)": 0.006,
    "AWS c8g.xlarge (4vCPU)": 0.0022,
    "AWS c8g.2xlarge (8vCPU)": 0.0022,
    "AWS c8g.4xlarge (16vCPU)": 0.0022,
    "AWS c8g.8xlarge (32vCPU)": 0.0022,
}

# Gemini API Pricing (Reference) - $/1M tokens
GEMINI_MODELS = {
    "Gemini 2.5 Flash":      {"input": 0.30, "output": 2.50},
    "Gemini 3 Flash Prev":   {"input": 0.50, "output": 3.00},
    "Gemini 2.5 Pro":        {"input": 1.25, "output": 10.00},
    "Gemini 3 Pro Prev":     {"input": 2.00, "output": 12.00},
}


# ============================================================================
# REPORTING HELPERS
# ============================================================================
# ============================================================================
# REPORTING HELPERS
# ============================================================================
def log_section(printer, title):
    printer("\n" + "="*80)
    printer(f" {title.center(78)} ")
    printer("="*80)

def log_header(printer, config):
    """
    Print the report header with all test configuration details.
    
    Args:
        printer: Function to write output
        config: Dictionary containing:
            - system_config: System description
            - model_name: Name of the model
            - duration: Test duration in seconds
            - users: Number of users
            - spawn_rate: Spawn rate
            - host: Target host
    """
    printer("\n" + "#"*80)
    printer(f"# {'LLM SERVICE LOAD TEST & COST ANALYSIS REPORT':^76} #")
    printer("#"*80)
    
    # 1. Test Configuration
    printer(f"\n{' TEST CONFIGURATION ':=^80}")
    printer(f"   Now:                        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    printer(f"   System Config:              {config.get('system_config', 'Unknown')}")
    printer(f"   Model Name:                 {config.get('model_name', 'Unknown')}")
    printer(f"   Target Host:                {config.get('host', 'Unknown')}")
    printer(f"   Test Duration:              {config.get('duration', 0):.2f} seconds")
    printer(f"   Concurrent Users:           {config.get('users', 'N/A')}")
    printer(f"   Spawn Rate:                 {config.get('spawn_rate', 'N/A')} users/sec")
    printer("-" * 80)

def log_request_summary(printer, total, successful, failed, peak_concurrent):
    printer(f"\n{' REQUEST METRICS ':=^80}")
    printer(f"   Total requests made:        {total}")
    printer(f"   ✓ Successful:               {successful}")
    printer(f"   ✗ Failed:                   {failed}")
    printer(f"   Peak concurrent requests:   {peak_concurrent}")
    rate = (successful/total*100) if total else 0
    printer(f"   Success rate:               {rate:.1f}%")

def log_token_analysis(printer, metrics, successful):
    printer(f"\n{' TOKEN ANALYSIS ':=^80}")
    printer(f"   Total Prompt Tokens:        {metrics.total_prompt_tokens:,}")
    printer(f"   Total Completion Tokens:    {metrics.total_completion_tokens:,}")
    printer(f"   Total All Tokens:           {metrics.total_all_tokens:,}")
    
    if successful > 0:
        printer(f"\n   Avg Prompt/Req:             {metrics.total_prompt_tokens / successful:.1f}")
        printer(f"   Avg Completion/Req:         {metrics.total_completion_tokens / successful:.1f}")
        printer(f"   Avg Total/Req:              {metrics.total_all_tokens / successful:.1f}")

def log_timing_analysis(printer, successful_requests):
    printer(f"\n{' LATENCY ANALYSIS ':=^80}")
    if not successful_requests:
        printer("   No successful requests to analyze.")
        return 0, 0

    inference_times = [r["inference_time_reported"] for r in successful_requests if r.get("inference_time_reported", 0) > 0]
    total_times = [r["elapsed_total"] for r in successful_requests]
    
    total_inference = sum(inference_times) if inference_times else 0
    
    if inference_times:
        printer(f"   Total Inference Time:       {total_inference:.2f}s")
        printer(f"   Average Inference Time:     {statistics.mean(inference_times):.3f}s")
        printer(f"   Median Inference Time:      {statistics.median(inference_times):.3f}s")
        printer(f"   Min Inference Time:         {min(inference_times):.3f}s")
        printer(f"   Max Inference Time:         {max(inference_times):.3f}s")
        
        # Total elapsed
        printer(f"\n   Avg Total Elapsed:          {statistics.mean(total_times):.3f}s")
        
    return total_inference, sum(total_times)

def get_blended_api_price(model_name, total_prompts, total_gen, total):
    """Calculate blended API price based on input/output ratio"""
    prices = GEMINI_MODELS.get(model_name)
    if not prices or total == 0:
        return 0.0
    return ((total_prompts * prices["input"]) + (total_gen * prices["output"])) / total

def log_cost_analysis(printer, metrics, total_inference_time, successful, system_config):
    printer(f"\n{' COST ANALYSIS ':=^80}")
    
    if successful == 0 or metrics.total_all_tokens == 0:
        printer("\n   Not enough data for cost calculation")
        return None

    # 1. API Benchmarks
    printer("\n   [API BENCHMARKS - Blended Cost based on I/O ratio]")
    printer("-" * 80)
    
    baseline_price = 0.0
    requests_per_m = (successful / metrics.total_all_tokens * 1_000_000)
    
    for model_name in GEMINI_MODELS:
        blended = get_blended_api_price(model_name, metrics.total_prompt_tokens, metrics.total_completion_tokens, metrics.total_all_tokens)
        per_req = blended / requests_per_m
        
        printer(f"   {model_name:<20}: ${blended:.4f}/1M tokens | ${per_req:.6f}/req")
        
        if model_name == "Gemini 2.5 Flash":
            baseline_price = blended

    printer("")

    # 2. Actual System Cost (If config matches known hardware)
    actual_system_cost = None
    actual_system_name = None
    matched_config = None
    
    # Simple substring match to find if system_config corresponds to a known pricing profile
    for key in PRICING_COMPUTE.keys():
        if key.lower() in system_config.lower() or system_config.lower() in key.lower():
            matched_config = key
            break
            
    if matched_config:
        printer(f"\n   [ACTUAL SYSTEM PERFORMANCE: {matched_config}]")
        printer("-" * 80)
        
        hourly_compute_rate = PRICING_COMPUTE[matched_config]
        disk_cost = PRICING_STORAGE.get(matched_config, 0.0)
        total_hourly = hourly_compute_rate + disk_cost
        
        actual_system_cost = calculate_effective_cost_per_million_tokens(
            metrics.total_all_tokens, total_inference_time, total_hourly
        )
        actual_cost_per_req = (total_inference_time / 3600) * total_hourly / successful
        
        printer(f"   Effective Cost:             ${actual_system_cost:.4f} / 1M tokens")
        printer(f"   Cost per Request:           ${actual_cost_per_req:.6f} / req")
        printer(f"   Hardware Rate Used:         ${total_hourly:.4f}/hr")
        actual_system_name = matched_config

    # 3. Hypothetical Comparison
    printer("\n   [HYPOTHETICAL COMPARISON (Projected costs for other hardware)]")
    printer("   (Assumes they would perform identically to the current test run)")
    printer("-" * 80)
    
    cheapest_cost = float('inf')
    cheapest_name = None

    for name, hourly_compute_rate in PRICING_COMPUTE.items():
        # Skip the actual system in the hypothetical list to avoid redundancy, 
        # unless we want to show it alongside others for comparison context.
        # We'll keep it for context.
        
        disk_cost = PRICING_STORAGE.get(name, 0.0)
        total_hourly = hourly_compute_rate + disk_cost
        
        cost_per_million = calculate_effective_cost_per_million_tokens(
            metrics.total_all_tokens, total_inference_time, total_hourly
        )
        cost_per_req = (total_inference_time / 3600) * total_hourly / successful
        
        printer(f"   {name:<25}: ${cost_per_million:.4f}/1M tokens")

        if cost_per_million < cheapest_cost:
            cheapest_cost = cost_per_million
            cheapest_name = name

    return cheapest_name, cheapest_cost, baseline_price, actual_system_name, actual_system_cost

def log_throughput_summary(printer, all_requests, successful, metrics):
    if not all_requests: 
        return

    printer(f"\n{' THROUGHPUT SUMMARY ':=^80}")
    
    test_start = min(r["start_time"] for r in all_requests)
    test_end = max(r["end_time"] for r in all_requests)
    duration = test_end - test_start
    
    printer(f"   Test Duration:              {duration:.2f}s")
    if duration > 0:
        printer(f"   Requests Per Second:        {successful / duration:.2f}")
        if metrics.total_all_tokens > 0:
            printer(f"   Tokens Per Wall-Clock Sec:  {metrics.total_all_tokens / duration:.2f}")

def export_csv_data(printer, successful_requests, system_config):
    printer(f"\n{' EXPORTING DATA ':=^80}")
    
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
        printer(f"   ✓ Detailed results exported to: {csv_path.absolute()}")
    except Exception as e:
        printer(f"   ✗ Failed to export CSV: {e}")

def log_final_summary(printer, successful, total, metrics, total_inference_time):
    printer(f"\n{' FINAL SUMMARY ':=^80}")
    
    if successful > 0 and metrics.total_all_tokens > 0:
        tps = metrics.total_all_tokens / total_inference_time if total_inference_time > 0 else 0
        printer(f"\n   Total Requests:        {successful} successful / {total} total")
        printer(f"   Total Tokens:          {metrics.total_all_tokens:,}")
        printer(f"   Total Inference Time:  {total_inference_time:.2f}s")
        printer(f"   Tokens/Second:         {tps:.2f}\n")

def log_final_verdict(printer, cheapest_name, cheapest_cost, baseline_price, avg_tps, actual_system_name, actual_system_cost):
    printer(f"\n{' VERDICT & RECOMMENDATION ':=^80}")
    
    if baseline_price == 0:
        printer("   Verdict unavailable (insufficient data).")
        return

    printer(f"   Scenario: Running at {avg_tps:.2f} Tokens/Sec (Avg)")
    
    printer(f"\n   1. API Baseline (Gemini 2.5 Flash):  ${baseline_price:.4f} / 1M tokens")
    
    # Verdict Branch A: We tested a known system
    if actual_system_name and actual_system_cost:
        printer(f"   2. ACTUAL SYSTEM ({actual_system_name}):  ${actual_system_cost:.4f} / 1M tokens\n")
        
        if actual_system_cost < baseline_price:
            savings = (1 - (actual_system_cost / baseline_price)) * 100
            printer(f"   >>> VERDICT: SELF-HOSTING ON {actual_system_name} IS CHEAPER <<<")
            printer(f"   Your setup is {savings:.1f}% CHEAPER than Gemini API.")
        else:
            extra_cost = ((actual_system_cost / baseline_price) - 1) * 100
            printer(f"   >>> VERDICT: GEMINI API IS CHEAPER <<<")
            printer(f"   Your setup ({actual_system_name}) is {extra_cost:.1f}% MORE EXPENSIVE.")
            
    # Verdict Branch B: Simulation / Unknown System
    else:
        printer(f"   2. Best Hypothetical Option:         ${cheapest_cost:.4f} / 1M tokens ({cheapest_name})\n")
        if cheapest_cost < baseline_price:
            printer(f"   >>> SIMULATION VERDICT: {cheapest_name} WOULD BE CHEAPER <<<")
            printer(f"   (Assuming it achieves the same {avg_tps:.2f} TPS performance)")
        else:
             printer(f"   >>> SIMULATION VERDICT: API IS CHEAPER <<<")


# ============================================================================
# MAIN REPORTING CONTROLLER
# ============================================================================
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate comprehensive cost analysis report when test stops"""
    
    system_config = environment.parsed_options.system_config
    all_requests = metrics.requests + metrics.failed_requests
    successful_requests = metrics.requests

    # Prepare log file
    safe_config = "".join(c if c.isalnum() else "_" for c in system_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"./logs/report_summary_{safe_config}_{timestamp}.txt")
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
    
    # 1. Header
    log_header(printer, config)
    
    # 2. Request Summary
    log_request_summary(printer, len(all_requests), len(successful_requests), len(metrics.failed_requests), metrics.peak_concurrent)
    
    # 3. Token Analysis
    log_token_analysis(printer, metrics, len(successful_requests))
    
    # 4. Timing Analysis
    total_inference_time, _ = log_timing_analysis(printer, successful_requests)
    
    # 5. Cost Analysis
    cheapest_name, cheapest_cost, baseline_price = None, 0.0, 0.0
    actual_system_name, actual_system_cost = None, None
    
    if successful_requests:
         result = log_cost_analysis(printer, metrics, total_inference_time, len(successful_requests), system_config)
         if result:
             cheapest_name, cheapest_cost, baseline_price, actual_system_name, actual_system_cost = result

    # 6. Throughput Summary
    log_throughput_summary(printer, all_requests, len(successful_requests), metrics)
    
    # 7. CSV Export
    export_csv_data(printer, successful_requests, system_config)
    
    # 8. Final Summary
    log_final_summary(printer, len(successful_requests), len(all_requests), metrics, total_inference_time)
    
    # 9. Final Verdict
    if baseline_price > 0:
        tps = metrics.total_all_tokens / total_inference_time if total_inference_time else 0
        log_final_verdict(printer, cheapest_name, cheapest_cost, baseline_price, tps, actual_system_name, actual_system_cost)
    
    printer(f"\n[INFO] Full report saved to: {report_path.absolute()}")


# Optional: Real-time progress updates
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Real-time request logging"""
    if not exception:
        print(f"[{time.strftime('%H:%M:%S')}] ✓ {name}: {response_time:.2f}ms")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ {name} failed: {exception}")
