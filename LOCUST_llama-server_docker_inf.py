from locust import HttpUser, task, between, events
import time
from threading import Lock
import requests
from collections import defaultdict
import statistics

# Metrics tracking
response_times = {"8080": [], "8081": []}
concurrent_requests = {"8080": 0, "8081": 0}
locks = {"8080": Lock(), "8081": Lock()}
request_timestamps = {"8080": [], "8081": []}

def build_prompt(context, questions, conversation_turns):
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

PROMPT = build_prompt(
    SURVEY_CONTEXT,
    QUESTION_LIST,
    CONVERSATION_TURNS
)

PAYLOAD = {
    "model": "gemma3_1b_400S_p77s16v20-Q8_0",
    "prompt": PROMPT,
    "max_tokens": 512,
    "stream": False
}

class LLMUser(HttpUser):
    host = "http://localhost:8080"
    wait_time = between(0, 0.1)

    @task(1)
    def hit_8080(self):
        endpoint = "8080"
        with locks[endpoint]:
            concurrent_requests[endpoint] += 1
            current_concurrent = concurrent_requests[endpoint]
        
        start_time = time.time()
        
        try:
            response = self.client.post(
                "/v1/completions",
                json=PAYLOAD,
                headers={"Authorization": "Bearer sk-no-key-required"},
                name="endpoint_8080",
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
            with locks[endpoint]:
                response_times[endpoint].append({
                    "time": elapsed,
                    "concurrent": current_concurrent,
                    "timestamp": start_time,
                    "end_time": time.time()
                })
                request_timestamps[endpoint].append((start_time, time.time()))
                
        except Exception as e:
            print(f"Error on 8080: {e}")
        finally:
            with locks[endpoint]:
                concurrent_requests[endpoint] -= 1

    @task(1)
    def hit_8081(self):
        endpoint = "8081"
        with locks[endpoint]:
            concurrent_requests[endpoint] += 1
            current_concurrent = concurrent_requests[endpoint]
        
        start_time = time.time()
        
        try:
            request_start = time.time()
            
            try:
                response = requests.post(
                    "http://localhost:8081/v1/completions",
                    json=PAYLOAD,
                    headers={"Authorization": "Bearer sk-no-key-required"},
                    timeout=60
                )
                response_time = (time.time() - request_start) * 1000
                
                events.request.fire(
                    request_type="POST",
                    name="endpoint_8081",
                    response_time=response_time,
                    response_length=len(response.content),
                    exception=None,
                    context={}
                )
                
            except Exception as e:
                response_time = (time.time() - request_start) * 1000
                events.request.fire(
                    request_type="POST",
                    name="endpoint_8081",
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={}
                )
                print(f"Error on 8081: {e}")
            
            elapsed = time.time() - start_time
            
            with locks[endpoint]:
                response_times[endpoint].append({
                    "time": elapsed,
                    "concurrent": current_concurrent,
                    "timestamp": start_time,
                    "end_time": time.time()
                })
                request_timestamps[endpoint].append((start_time, time.time()))
                
        finally:
            with locks[endpoint]:
                concurrent_requests[endpoint] -= 1


def calculate_overlap_percentage(timestamps_8080, timestamps_8081):
    """Calculate what percentage of time both endpoints were processing simultaneously"""
    if not timestamps_8080 or not timestamps_8081:
        return 0.0
    
    overlapping_time = 0.0
    
    for start1, end1 in timestamps_8080:
        for start2, end2 in timestamps_8081:
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlapping_time += (overlap_end - overlap_start)
    
    # Total time both endpoints were active
    total_8080 = sum(end - start for start, end in timestamps_8080)
    total_8081 = sum(end - start for start, end in timestamps_8081)
    
    # Average of both endpoints' total time
    avg_total_time = (total_8080 + total_8081) / 2
    
    if avg_total_time == 0:
        return 0.0
    
    return (overlapping_time / avg_total_time) * 100


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Comprehensive analysis when test stops"""
    print("\n" + "="*100)
    print(" " * 35 + "PARALLELISM TEST RESULTS")
    print("="*100)
    
    # Basic statistics
    print("\n📊 BASIC STATISTICS")
    print("-" * 100)
    
    for endpoint in ["8080", "8081"]:
        times = response_times[endpoint]
        if times:
            response_time_values = [t["time"] for t in times]
            avg_time = statistics.mean(response_time_values)
            median_time = statistics.median(response_time_values)
            max_concurrent = max(t["concurrent"] for t in times)
            
            print(f"\n🔹 Endpoint {endpoint}:")
            print(f"   Total requests:          {len(times)}")
            print(f"   Average response time:   {avg_time:.3f}s")
            print(f"   Median response time:    {median_time:.3f}s")
            print(f"   Min response time:       {min(response_time_values):.3f}s")
            print(f"   Max response time:       {max(response_time_values):.3f}s")
            print(f"   Std deviation:           {statistics.stdev(response_time_values) if len(response_time_values) > 1 else 0:.3f}s")
            print(f"   Max concurrent requests: {max_concurrent}")
    
    # Overlap analysis
    print("\n\n🔄 OVERLAP ANALYSIS")
    print("-" * 100)
    
    all_requests = []
    for endpoint in ["8080", "8081"]:
        for req in response_times[endpoint]:
            all_requests.append({
                "endpoint": endpoint,
                "start": req["timestamp"],
                "end": req["end_time"],
                "duration": req["time"]
            })
    
    all_requests.sort(key=lambda x: x["start"])
    
    overlaps = 0
    overlap_durations = []
    
    for i in range(len(all_requests)):
        for j in range(i + 1, len(all_requests)):
            req1, req2 = all_requests[i], all_requests[j]
            if req1["endpoint"] != req2["endpoint"]:
                # Check if they overlap
                overlap_start = max(req1["start"], req2["start"])
                overlap_end = min(req1["end"], req2["end"])
                
                if overlap_start < overlap_end:
                    overlaps += 1
                    overlap_duration = overlap_end - overlap_start
                    overlap_durations.append(overlap_duration)
    
    overlap_percentage = calculate_overlap_percentage(
        request_timestamps["8080"],
        request_timestamps["8081"]
    )
    
    print(f"\n   Cross-endpoint overlapping request pairs: {overlaps}")
    if overlap_durations:
        print(f"   Average overlap duration:                 {statistics.mean(overlap_durations):.3f}s")
        print(f"   Total overlapping time:                   {sum(overlap_durations):.3f}s")
        print(f"   Overlap percentage:                       {overlap_percentage:.1f}%")
    
    # Performance comparison
    print("\n\n⚡ PERFORMANCE COMPARISON")
    print("-" * 100)
    
    if response_times["8080"] and response_times["8081"]:
        avg_8080 = statistics.mean([t["time"] for t in response_times["8080"]])
        avg_8081 = statistics.mean([t["time"] for t in response_times["8081"]])
        
        diff_percentage = abs(avg_8080 - avg_8081) / min(avg_8080, avg_8081) * 100
        
        print(f"\n   Average response time difference:  {abs(avg_8080 - avg_8081):.3f}s ({diff_percentage:.1f}%)")
        
        if diff_percentage < 10:
            print(f"   ✓ Both endpoints have similar performance (< 10% difference)")
        else:
            slower = "8080" if avg_8080 > avg_8081 else "8081"
            print(f"   ⚠ Endpoint {slower} is significantly slower")
    
    # Final verdict
    print("\n\n🎯 FINAL VERDICT")
    print("-" * 100)
    
    print("\n   Parallelism Indicators:")
    
    # Indicator 1: Overlaps
    if overlaps > 0:
        print(f"   ✓ Overlapping requests detected: {overlaps} pairs")
        overlap_score = "HIGH" if overlaps > 10 else "MEDIUM" if overlaps > 3 else "LOW"
        print(f"     Overlap score: {overlap_score}")
    else:
        print(f"   ✗ No overlapping requests detected")
    
    # Indicator 2: Overlap percentage
    if overlap_percentage > 20:
        print(f"   ✓ High overlap percentage: {overlap_percentage:.1f}%")
    elif overlap_percentage > 5:
        print(f"   ~ Moderate overlap percentage: {overlap_percentage:.1f}%")
    else:
        print(f"   ✗ Low overlap percentage: {overlap_percentage:.1f}%")
    
    # Indicator 3: Performance similarity
    if response_times["8080"] and response_times["8081"]:
        avg_8080 = statistics.mean([t["time"] for t in response_times["8080"]])
        avg_8081 = statistics.mean([t["time"] for t in response_times["8081"]])
        diff_percentage = abs(avg_8080 - avg_8081) / min(avg_8080, avg_8081) * 100
        
        if diff_percentage < 15:
            print(f"   ✓ Similar performance between endpoints: {diff_percentage:.1f}% difference")
        else:
            print(f"   ⚠ Significant performance difference: {diff_percentage:.1f}%")
    
    # Overall conclusion
    print("\n   " + "="*96)
    
    if overlaps > 5 and overlap_percentage > 10:
        print("   ✅ CONCLUSION: Endpoints ARE running in TRUE PARALLEL")
        print("   Both endpoints process requests simultaneously without blocking each other.")
    elif overlaps > 0:
        print("   ⚠️  CONCLUSION: Endpoints have PARTIAL PARALLELISM")
        print("   Some parallel execution detected, but may have resource contention.")
    else:
        print("   ❌ CONCLUSION: Endpoints are NOT running in parallel")
        print("   Requests appear to be processed sequentially (one blocks the other).")
    
    print("   " + "="*96)
    print("\n" + "="*100 + "\n")


# Optional: Real-time monitoring during the test
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Optional: Print requests in real-time to see parallelism live"""
    # Uncomment below to see real-time request logs
    # print(f"[{time.strftime('%H:%M:%S')}] {name}: {response_time:.0f}ms")
    pass