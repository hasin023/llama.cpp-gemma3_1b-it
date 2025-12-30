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
    locust -f  .\scripts\LOCUST_llm-service-api.py --headless -u 1 -r 1 -t 120s --host http://localhost:8000 --system-config "i5-8365U CPU"

Usage (REMOTE testing against RunPod/AWS VM):
    # Replace <VM_IP> with your RunPod/AWS public IP address
    locust -f  .\scripts\LOCUST_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://<VM_IP>:8000

    # Examples:
    # RunPod:  locust -f  .\scripts\LOCUST_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://85.123.45.67:8000
    # AWS:    locust -f  .\scripts\LOCUST_llm-service-api.py --headless -u 10 -r 5 -t 120s --host http://ec2-12-34-56-78.compute-1.amazonaws.com:8000

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
    "AWS c8g.xlarge (4vCPU)": 0.16,
    "AWS c8g.2xlarge (8vCPU)": 0.32,
    "AWS c8g.4xlarge (16vCPU)": 0.64,
    "AWS c8g.8xlarge (32vCPU)": 1.28,
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
    printer("\n" + "="*100)
    printer(" " * 25 + title)
    printer("="*100)

def log_header(printer, system_config):
    printer("\n" + "="*100)
    printer(" " * 25 + "LLM SERVICE COST ANALYSIS REPORT")
    printer(f" " * 25 + f"System: {system_config}")
    printer("="*100)

def log_request_summary(printer, total, successful, failed, peak_concurrent):
    printer("\nREQUEST SUMMARY")
    printer("-" * 100)
    printer(f"\n   Total requests made:        {total}")
    printer(f"   ✓ Successful:               {successful}")
    printer(f"   ✗ Failed:                   {failed}")
    printer(f"   Peak concurrent requests:   {peak_concurrent}")
    printer(f"\n   Success rate:               {(successful/total*100) if total else 0:.1f}%")

def log_token_analysis(printer, metrics, successful):
    printer("\nTOKEN ANALYSIS")
    printer("-" * 100)
    printer(f"\n   Total Prompt Tokens:        {metrics.total_prompt_tokens:,}")
    printer(f"   Total Completion Tokens:    {metrics.total_completion_tokens:,}")
    printer(f"   Total All Tokens:           {metrics.total_all_tokens:,}")
    
    if successful > 0:
        printer(f"\n   Avg Prompt/Req:             {metrics.total_prompt_tokens / successful:.1f}")
        printer(f"   Avg Completion/Req:         {metrics.total_completion_tokens / successful:.1f}")
        printer(f"   Avg Total/Req:              {metrics.total_all_tokens / successful:.1f}")

def log_timing_analysis(printer, successful_requests):
    printer("\nTIMING ANALYSIS")
    printer("-" * 100)
    if not successful_requests:
        printer("   No successful requests to analyze.")
        return 0, 0

    inference_times = [r["inference_time_reported"] for r in successful_requests if r.get("inference_time_reported", 0) > 0]
    total_times = [r["elapsed_total"] for r in successful_requests]
    
    total_inference = sum(inference_times) if inference_times else 0
    
    if inference_times:
        printer(f"\n   Total Inference Time:       {total_inference:.2f}s")
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

def log_cost_analysis(printer, metrics, total_inference_time, successful):
    printer("\nCOST ANALYSIS")
    printer("-" * 100)
    
    if successful == 0 or metrics.total_all_tokens == 0:
        printer("\n   Not enough data for cost calculation")
        return None

    # 1. API Benchmarks
    printer("\n   [API BENCHMARKS - Blended Cost based on your Input/Output ratio]")
    
    baseline_price = 0.0
    requests_per_m = (successful / metrics.total_all_tokens * 1_000_000)
    
    for model_name in GEMINI_MODELS:
        blended = get_blended_api_price(model_name, metrics.total_prompt_tokens, metrics.total_completion_tokens, metrics.total_all_tokens)
        per_req = blended / requests_per_m
        
        printer(f"   {model_name:<20}: ${blended:.4f} per 1M tokens | ${per_req:.6f} per request")
        
        if model_name == "Gemini 2.5 Flash":
            baseline_price = blended

    printer("")

    # 2. Self-Hosted Hardware Costs
    printer("   Effective Cost per 1M Tokens (Compute + Disk):")
    
    cheapest_cost = float('inf')
    cheapest_name = None

    for name, hourly_compute_rate in PRICING_COMPUTE.items():
        disk_cost = PRICING_STORAGE.get(name, 0.0)
        total_hourly = hourly_compute_rate + disk_cost
        
        cost_per_million = calculate_effective_cost_per_million_tokens(
            metrics.total_all_tokens, total_inference_time, total_hourly
        )
        cost_per_req = (total_inference_time / 3600) * total_hourly / successful
        
        # ROI Comparison
        roi_text = ""
        if baseline_price > 0:
            roi_factor = cost_per_million / baseline_price
            roi_text = f"({roi_factor:.1f}x Flash 2.5)"
            
        printer(f"\n   {name}:")
        printer(f"      Compute: ${hourly_compute_rate:.2f}/hr + Disk: ${disk_cost:.4f}/hr")
        printer(f"      ${cost_per_million:.4f} per 1M tokens {roi_text}")
        printer(f"      ${cost_per_req:.6f} per request")

        if cost_per_million < cheapest_cost:
            cheapest_cost = cost_per_million
            cheapest_name = name

    return cheapest_name, cheapest_cost, baseline_price

def log_throughput_summary(printer, all_requests, successful, metrics):
    if not all_requests: 
        return

    printer("\nTHROUGHPUT SUMMARY")
    printer("-" * 100)
    
    test_start = min(r["start_time"] for r in all_requests)
    test_end = max(r["end_time"] for r in all_requests)
    duration = test_end - test_start
    
    printer(f"\n   Test Duration:              {duration:.2f}s")
    if duration > 0:
        printer(f"   Requests Per Second:        {successful / duration:.2f}")
        if metrics.total_all_tokens > 0:
            printer(f"   Tokens Per Wall-Clock Sec:  {metrics.total_all_tokens / duration:.2f}")

def export_csv_data(printer, successful_requests, system_config):
    printer("\nEXPORTING DATA")
    printer("-" * 100)
    
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
        printer(f"\n   ✓ Detailed results exported to: {csv_path.absolute()}")
    except Exception as e:
        printer(f"\n   ✗ Failed to export CSV: {e}")

def log_final_summary(printer, successful, total, metrics, total_inference_time):
    printer("\n\n" + "="*100)
    printer(" " * 35 + "SUMMARY")
    printer("="*100)
    
    if successful > 0 and metrics.total_all_tokens > 0:
        tps = metrics.total_all_tokens / total_inference_time if total_inference_time > 0 else 0
        printer(f"""
   Total Requests:        {successful} successful / {total} total
   Total Tokens:          {metrics.total_all_tokens:,}
   Total Inference Time:  {total_inference_time:.2f}s
   Tokens/Second:         {tps:.2f}
""")

def log_final_verdict(printer, cheapest_name, cheapest_cost, baseline_price, avg_tps):
    printer("-" * 100)
    printer("FINAL VERDICT & RECOMMENDATION")
    printer("-" * 100)
    
    if not cheapest_name or baseline_price == 0:
        printer("   Verdict unavailable (insufficient data).")
        return

    printer(f"Scenario: Running at {avg_tps:.2f} Tokens/Sec (Avg)")
    printer(f"          (Disclaimer: Comparative hardware costs assume this measured speed)\n")
    
    printer(f"1. API Baseline (Gemini 2.5 Flash):  ${baseline_price:.4f} / 1M tokens")
    printer(f"2. Best Self-Hosted Option:          ${cheapest_cost:.4f} / 1M tokens ({cheapest_name})\n")
    
    if cheapest_cost < baseline_price:
        savings = (1 - (cheapest_cost / baseline_price)) * 100
        printer(f"   VERDICT: SELF-HOSTING IS CHEAPER")
        printer(f"   The '{cheapest_name}' is {savings:.1f}% CHEAPER than Gemini API.")
        printer(f"   Condition: You must achieve > {avg_tps:.1f} TPS to maintain this efficiency.")
    else:
        extra_cost = ((cheapest_cost / baseline_price) - 1) * 100
        printer(f"   VERDICT: USE GEMINI API")
        printer(f"   Self-hosting is {extra_cost:.1f}% MORE EXPENSIVE than Gemini API at this speed.")
        printer(f"   To beat API pricing, you need faster inference time or cheaper hardware.")


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

    
    # 1. Header
    log_header(printer, system_config)
    
    # 2. Request Summary
    log_request_summary(printer, len(all_requests), len(successful_requests), len(metrics.failed_requests), metrics.peak_concurrent)
    
    # 3. Token Analysis
    log_token_analysis(printer, metrics, len(successful_requests))
    
    # 4. Timing Analysis
    total_inference_time, _ = log_timing_analysis(printer, successful_requests)
    
    # 5. Cost Analysis
    cheapest_name, cheapest_cost, baseline_price = None, 0.0, 0.0
    if successful_requests:
         result = log_cost_analysis(printer, metrics, total_inference_time, len(successful_requests))
         if result:
             cheapest_name, cheapest_cost, baseline_price = result

    # 6. Throughput Summary
    log_throughput_summary(printer, all_requests, len(successful_requests), metrics)
    
    # 7. CSV Export
    export_csv_data(printer, successful_requests, system_config)
    
    # 8. Final Summary
    log_final_summary(printer, len(successful_requests), len(all_requests), metrics, total_inference_time)
    
    # 9. Final Verdict
    if cheapest_name and baseline_price > 0:
        tps = metrics.total_all_tokens / total_inference_time if total_inference_time else 0
        log_final_verdict(printer, cheapest_name, cheapest_cost, baseline_price, tps)
    
    printer(f"\n[INFO] Full report saved to: {report_path.absolute()}")


# Optional: Real-time progress updates
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Real-time request logging"""
    if not exception:
        print(f"[{time.strftime('%H:%M:%S')}] ✓ {name}: {response_time:.2f}ms")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ {name} failed: {exception}")
