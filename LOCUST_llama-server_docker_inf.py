"""
LLaMA Server Parallelism Load Test
===================================
This test verifies TRUE parallelism - checking that ALL concurrent requests
receive valid LLM-generated responses simultaneously, not just that requests overlap.

Usage:
    locust -f LOCUST_llama-server_docker_inf.py --headless -u 50 -r 17 -t 120s --host http://localhost:8080


Key Metrics Tested:
1. All requests receive valid LLM-generated content
2. Requests are processed simultaneously (overlapping time windows)
3. Response quality is consistent across parallel requests
4. No request starvation (all requests complete successfully)
"""

from locust import HttpUser, task, between, events
import time
import os
from threading import Lock
import statistics
from collections import defaultdict
import json
import hashlib
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# METRICS TRACKING
# ============================================================================
class ParallelismMetrics:
    """Thread-safe metrics collector for parallelism testing"""
    
    def __init__(self):
        self.lock = Lock()
        self.requests = []  # All request data
        self.concurrent_count = 0  # Current concurrent requests
        self.peak_concurrent = 0  # Peak concurrent requests observed
        self.failed_requests = []  # Failed request details
        self.empty_responses = []  # Requests with empty/no LLM output
        
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
    
    def record_failure(self, data):
        with self.lock:
            self.failed_requests.append(data)
    
    def record_empty_response(self, data):
        with self.lock:
            self.empty_responses.append(data)


metrics = ParallelismMetrics()


# ============================================================================
# PROMPT BUILDER
# ============================================================================
def build_prompt(context, questions, conversation_turns):
    """Build Gemma-formatted prompt"""
    prompt = """<start_of_turn>user
You are a polite Bangla phone survey agent. ALWAYS respond in Bangla.
ONLY ask the questions from the Question List one by one, exactly as written, in order.
IF user queries about anything, respond from the Survey Context, then continue with the next question.

Survey Context:
""" + context + """

Question List:
""" + "\n".join(["• " + q for q in questions]) + """
<end_of_turn>
"""
    
    for turn in conversation_turns:
        role = turn["role"]
        content = turn["content"]
        prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"
    
    prompt += "<start_of_turn>model\n"
    
    return prompt


# ============================================================================
# TEST DATA
# ============================================================================
SURVEY_CONTEXT = "বাংলাদেশে FrozenBerry কিনতে পারবেন Foodpanda, Daraz, Unimart, Shawpno এর মতো অনলাইন শপ ও সুপারশপ থেকে।"

QUESTION_LIST = [
    "আপনি কি FrozenBerry ব্যবহার করেছেন?",
    "এটার স্বাদ নিয়ে কি কোনো feedback দিতে চান?",
    "সংরক্ষণ করা সহজ কি?"
]

CONVERSATION_TURNS = [
    {"role": "model", "content": "আমি একটি জরিপ কোম্পানি থেকে বলছি। আপনি কি অনিম বলছেন?"},
    {"role": "user", "content": "জি বলছি"},
    {"role": "model", "content": "আপনি কি FrozenBerry ব্যবহার করেছেন?"},
    {"role": "user", "content": "হ্যাঁ, ব্যবহার করেছি।"},
]

PROMPT = build_prompt(SURVEY_CONTEXT, QUESTION_LIST, CONVERSATION_TURNS)

PAYLOAD = {
    "model": "gemma3_1b_400S_p77s16v20",
    "prompt": PROMPT,
    "max_tokens": 256,
    "stream": False
}


# ============================================================================
# LOCUST USER
# ============================================================================
class LLMUser(HttpUser):
    """
    Load test user that verifies TRUE parallelism:
    - Each request must receive valid LLM-generated content
    - Requests should be processed simultaneously
    """
    host = "http://localhost:8080"
    wait_time = between(0, 0.1)  # Minimal wait to maximize concurrency
    api_key = os.getenv("OPENAI_API_KEY", "sk-no-key-required")

    @task
    def inference_request(self):
        """Send inference request and verify LLM generation"""
        
        # Track concurrency
        concurrent_at_start = metrics.start_request()
        
        request_id = hashlib.md5(f"{time.time()}-{id(self)}".encode()).hexdigest()[:8]
        start_time = time.time()
        
        try:
            response = self.client.post(
                "/v1/completions",
                json=PAYLOAD,
                headers={"Authorization": f"Bearer {self.api_key}"},
                name="llm_completion",
                timeout=120
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Parse and validate response
            success = False
            generated_text = ""
            token_count = 0
            slot_id = None
            
            if response.status_code == 200:
                try:
                    resp_json = response.json()
                    
                    # Extract generated text
                    if "choices" in resp_json and len(resp_json["choices"]) > 0:
                        choice = resp_json["choices"][0]
                        generated_text = choice.get("text", "")
                        
                        # Get token count if available
                        if "usage" in resp_json:
                            token_count = resp_json["usage"].get("completion_tokens", 0)
                    
                    # Check if we got actual LLM output
                    if generated_text and len(generated_text.strip()) > 5:
                        success = True
                    else:
                        # Empty or trivial response - LLM didn't actually generate
                        metrics.record_empty_response({
                            "request_id": request_id,
                            "timestamp": start_time,
                            "concurrent_at_start": concurrent_at_start,
                            "response_text": generated_text[:100] if generated_text else "(empty)",
                            "elapsed": elapsed
                        })
                        
                except Exception as e:
                    pass
            
            # Record metrics
            request_data = {
                "request_id": request_id,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed": elapsed,
                "concurrent_at_start": concurrent_at_start,
                "success": success,
                "generated_text": generated_text[:200] if generated_text else "",
                "text_length": len(generated_text) if generated_text else 0,
                "token_count": token_count,
                "status_code": response.status_code
            }
            
            if success:
                metrics.record_success(request_data)
            else:
                metrics.record_failure(request_data)
                
        except Exception as e:
            end_time = time.time()
            metrics.record_failure({
                "request_id": request_id,
                "start_time": start_time,
                "end_time": end_time,
                "elapsed": end_time - start_time,
                "concurrent_at_start": concurrent_at_start,
                "success": False,
                "error": str(e),
                "status_code": 0
            })
            print(f"Request {request_id} failed: {e}")
            
        finally:
            metrics.end_request()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def find_overlapping_requests(requests):
    """
    Find all requests that were genuinely running in parallel.
    Two requests overlap if request A started before request B ended AND
    request B started before request A ended.
    """
    overlapping_pairs = []
    
    for i, req1 in enumerate(requests):
        for j, req2 in enumerate(requests):
            if i >= j:
                continue
            
            # Check temporal overlap
            start1, end1 = req1["start_time"], req1["end_time"]
            start2, end2 = req2["start_time"], req2["end_time"]
            
            if start1 < end2 and start2 < end1:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_duration = overlap_end - overlap_start
                
                overlapping_pairs.append({
                    "req1_id": req1["request_id"],
                    "req2_id": req2["request_id"],
                    "overlap_duration": overlap_duration,
                    "both_successful": req1["success"] and req2["success"]
                })
    
    return overlapping_pairs


def calculate_parallelism_score(requests):
    """
    Calculate what percentage of the test time had multiple requests running.
    Higher score = more parallelism.
    """
    if len(requests) < 2:
        return 0.0
    
    # Create timeline of events
    events_list = []
    for req in requests:
        events_list.append((req["start_time"], "start"))
        events_list.append((req["end_time"], "end"))
    
    events_list.sort(key=lambda x: x[0])
    
    total_time = 0.0
    parallel_time = 0.0
    active_count = 0
    last_time = events_list[0][0]
    
    for event_time, event_type in events_list:
        duration = event_time - last_time
        total_time += duration
        
        if active_count >= 2:
            parallel_time += duration
        
        if event_type == "start":
            active_count += 1
        else:
            active_count -= 1
        
        last_time = event_time
    
    return (parallel_time / total_time * 100) if total_time > 0 else 0.0


def verify_all_requests_generated(requests):
    """
    The KEY test: Did ALL requests actually get LLM-generated responses?
    Returns (fully_successful, partial, failed) counts.
    """
    fully_successful = 0  # Got valid LLM output
    partial = 0  # Request completed but with issues
    failed = 0  # Request failed or got no output
    
    for req in requests:
        if req["success"] and req["text_length"] > 20:
            fully_successful += 1
        elif req.get("status_code") == 200:
            partial += 1
        else:
            failed += 1
    
    return fully_successful, partial, failed


# ============================================================================
# TEST COMPLETION REPORT
# ============================================================================
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Comprehensive parallelism analysis when test stops"""
    
    all_requests = metrics.requests + metrics.failed_requests
    successful_requests = metrics.requests
    
    print("\n" + "="*100)
    print(" " * 20 + "🔬 TRUE PARALLELISM VERIFICATION REPORT 🔬")
    print("="*100)
    
    # -------------------------------------------------------------------------
    # Section 1: Request Summary
    # -------------------------------------------------------------------------
    print("\n📊 REQUEST SUMMARY")
    print("-" * 100)
    
    total_requests = len(all_requests)
    successful = len([r for r in all_requests if r["success"]])
    failed = len([r for r in all_requests if not r["success"]])
    
    print(f"\n   Total requests made:         {total_requests}")
    print(f"   ✅ Successful (got LLM output): {successful}")
    print(f"   ❌ Failed/Empty response:       {failed}")
    print(f"   ⚠️  Empty response warnings:     {len(metrics.empty_responses)}")
    print(f"   Peak concurrent requests:    {metrics.peak_concurrent}")
    
    success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
    print(f"\n   Success rate: {success_rate:.1f}%")
    
    # -------------------------------------------------------------------------
    # Section 2: Response Time Analysis
    # -------------------------------------------------------------------------
    if successful_requests:
        print("\n\n⏱️  RESPONSE TIME ANALYSIS")
        print("-" * 100)
        
        response_times = [r["elapsed"] for r in successful_requests]
        
        print(f"\n   Average response time:    {statistics.mean(response_times):.3f}s")
        print(f"   Median response time:     {statistics.median(response_times):.3f}s")
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"   Min response time:        {min_time:.3f}s")
        print(f"   Max response time:        {max_time:.3f}s")

        # 90th Percentile Calculation
        sorted_times = sorted(response_times)
        if len(sorted_times) >= 10:
             # Use accurate index for 90th percentile
             p90_index = int(0.90 * len(sorted_times))
             if p90_index >= len(sorted_times): p90_index = len(sorted_times) - 1
             p90 = sorted_times[p90_index]
             print(f"   90th percentile time:     {p90:.3f}s")
        else:
             print(f"   90th percentile time:     (not enough samples)")

        # Calculate percentage of requests within 10% of minimum inference time
        threshold = min_time * 1.10
        fast_requests = len([t for t in response_times if t <= threshold])
        fast_percentage = (fast_requests / len(response_times)) * 100
        print(f"   Requests within 10% of min: {fast_percentage:.1f}% ({fast_requests}/{len(response_times)})")
        if len(response_times) > 1:
            print(f"   Std deviation:            {statistics.stdev(response_times):.3f}s")
        
        # Check for response time consistency (important for parallelism)
        if len(response_times) > 1:
            cv = statistics.stdev(response_times) / statistics.mean(response_times)
            consistency = "CONSISTENT" if cv < 0.3 else "VARIABLE" if cv < 0.6 else "HIGHLY VARIABLE"
            print(f"\n   Response time consistency: {consistency} (CV: {cv:.2f})")
    
    # -------------------------------------------------------------------------
    # Section 3: TRUE Parallelism Verification
    # -------------------------------------------------------------------------
    print("\n\n🔄 TRUE PARALLELISM VERIFICATION")
    print("-" * 100)
    
    if len(all_requests) < 2:
        print("\n   ⚠️  Not enough requests to verify parallelism")
    else:
        overlapping_pairs = find_overlapping_requests(all_requests)
        parallelism_score = calculate_parallelism_score(all_requests)
        
        # Count pairs where BOTH requests got valid LLM output
        both_successful_pairs = [p for p in overlapping_pairs if p["both_successful"]]
        
        print(f"\n   Overlapping request pairs found: {len(overlapping_pairs)}")
        print(f"   Pairs where BOTH got LLM output: {len(both_successful_pairs)}")
        print(f"   Parallelism score:               {parallelism_score:.1f}%")
        
        if both_successful_pairs:
            avg_overlap = statistics.mean([p["overlap_duration"] for p in both_successful_pairs])
            print(f"   Average overlap duration:        {avg_overlap:.3f}s")
    
    # -------------------------------------------------------------------------
    # Section 4: The KEY Test - All Requests Generated
    # -------------------------------------------------------------------------
    print("\n\n🎯 KEY TEST: DID ALL REQUESTS GET LLM GENERATION?")
    print("-" * 100)
    
    fully_gen, partial, failed_gen = verify_all_requests_generated(all_requests)
    
    print(f"\n   ✅ Fully generated (>20 chars):  {fully_gen}")
    print(f"   ⚠️  Partial/short response:       {partial}")
    print(f"   ❌ Failed/no generation:          {failed_gen}")
    
    if fully_gen == total_requests:
        print(f"\n   ✓ ALL {total_requests} REQUESTS GOT VALID LLM OUTPUT!")
    elif fully_gen > 0:
        print(f"\n   ⚠ Only {fully_gen}/{total_requests} requests got valid LLM output")
    else:
        print(f"\n   ✗ NO REQUESTS got valid LLM output!")
    
    # -------------------------------------------------------------------------
    # Section 5: Sample Outputs (proof of generation)
    # -------------------------------------------------------------------------
    print("\n\n📝 SAMPLE GENERATED OUTPUTS (Proof of LLM Generation)")
    print("-" * 100)
    
    sample_count = min(5, len(successful_requests))
    for i, req in enumerate(successful_requests[:sample_count]):
        print(f"\n   Request {req['request_id']}:")
        print(f"   Time: {req['elapsed']:.3f}s | Concurrent at start: {req['concurrent_at_start']}")
        print(f"   Output preview: {req['generated_text'][:80]}...")
    
    # -------------------------------------------------------------------------
    # Section 6: Throughput Analysis
    # -------------------------------------------------------------------------
    if all_requests:
        print("\n\n⚡ THROUGHPUT ANALYSIS")
        print("-" * 100)
        
        test_start = min(r["start_time"] for r in all_requests)
        test_end = max(r["end_time"] for r in all_requests)
        test_duration = test_end - test_start
        
        total_processing_time = sum(r["elapsed"] for r in successful_requests)
        
        print(f"\n   Test wall-clock duration:    {test_duration:.2f}s")
        print(f"   Total processing time:       {total_processing_time:.2f}s")
        print(f"   Requests completed:          {successful}")
        
        if test_duration > 0:
            throughput = successful / test_duration
            print(f"   Throughput:                  {throughput:.2f} req/s")
            
            # Calculate speedup vs sequential
            speedup = total_processing_time / test_duration if test_duration > 0 else 1
            print(f"\n   Parallelism speedup:         {speedup:.2f}x faster than sequential")
    
    # -------------------------------------------------------------------------
    # Section 7: Final Verdict
    # -------------------------------------------------------------------------
    print("\n\n" + "="*100)
    print(" " * 35 + "🏆 FINAL VERDICT 🏆")
    print("="*100)
    
    # Calculate overall parallelism quality
    issues = []
    
    if metrics.peak_concurrent < 2:
        issues.append("Peak concurrency never exceeded 1 (no parallel requests)")
    
    if len(both_successful_pairs) == 0 and len(all_requests) >= 2:
        issues.append("No overlapping requests both received valid LLM output")
    
    if success_rate < 90:
        issues.append(f"Low success rate: {success_rate:.1f}%")
    
    if len(metrics.empty_responses) > 0:
        issues.append(f"{len(metrics.empty_responses)} requests got empty/invalid responses")
    
    if not issues:
        parallelism_score = calculate_parallelism_score(all_requests) if len(all_requests) >= 2 else 0
        
        if parallelism_score >= 50 and fully_gen == total_requests:
            print("\n   ✅ TRUE PARALLEL PROCESSING CONFIRMED!")
            print("   All concurrent requests received valid LLM-generated responses.")
            print("   The server is correctly handling parallel inference.")
        elif parallelism_score >= 20:
            print("\n   ⚠️  PARTIAL PARALLELISM DETECTED")
            print("   Some parallel processing occurred, but not at full capacity.")
            print("   Consider sending more concurrent requests or checking --parallel setting.")
        else:
            print("\n   ⚠️  LIMITED PARALLELISM")
            print("   Requests were processed mostly sequentially.")
            print("   Check if LLAMA_ARG_N_PARALLEL is set correctly in your config.")
    else:
        print("\n   ❌ PARALLELISM ISSUES DETECTED:")
        for issue in issues:
            print(f"      • {issue}")
        print("\n   Recommendations:")
        print("      • Ensure LLAMA_ARG_N_PARALLEL=4 or higher in compose.yaml")
        print("      • Check server logs for errors")
        print("      • Increase concurrent users: locust -u 8 or higher")
    
    print("\n" + "="*100)
    print("\n💡 Run command: locust -f LOCUST_llama-server_docker_inf.py --headless -u 8 -r 2 -t 60s")
    print("="*100 + "\n")


# Optional: Real-time progress updates
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Real-time request logging"""
    if not exception:
        print(f"[{time.strftime('%H:%M:%S')}] Request {name}: {response_time:.2f}ms")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ Request {name} failed: {exception}")