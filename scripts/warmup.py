"""
Warmup Script for LLM Service
------------------------------
Waits for llama-cpp-server and llm-service-api to be healthy,
then sends a minimal warmup request to initialize the model.

This runs as a one-shot container in Docker Compose to ensure
the stack is ready before accepting real traffic.
"""

import time
import requests
import sys
from pathlib import Path

# Service endpoints
LLAMA_CPP_SERVER_URL = "http://llama-cpp-server:8080"
LLM_SERVICE_API_URL = "http://llm-service-api:8000"

# Wait configuration
MAX_WAIT_SECONDS = 300  # 5 minutes max wait
HEALTH_CHECK_INTERVAL = 2  # Check every 2 seconds
WARMUP_TIMEOUT = 60  # 60 seconds for warmup request


def wait_for_service(name: str, url: str, endpoint: str = "/health") -> bool:
    """Wait for a service to become healthy."""
    print(f"Waiting for {name} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < MAX_WAIT_SECONDS:
        try:
            response = requests.get(f"{url}{endpoint}", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"✓ {name} is healthy (took {elapsed:.1f}s)")
                return True
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass
        
        time.sleep(HEALTH_CHECK_INTERVAL)
    
    print(f"✗ {name} failed to become healthy within {MAX_WAIT_SECONDS}s")
    return False


def send_warmup_request() -> bool:
    """Send a minimal warmup request to the LLM Service."""
    print("Sending warmup request to LLM Service...")
    
    # Minimal warmup payload - just enough to trigger model initialization
    warmup_payload = {
        "survey_context": "Warmup test context.",
        "questions": ["Test question?"],
        "user_query": "test",
        "continue_in_same_conversation": False,  # Start fresh for warmup
        "max_tokens": 10,  # Very short response for speed
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{LLM_SERVICE_API_URL}/query",
            json=warmup_payload,
            timeout=WARMUP_TIMEOUT,
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Warmup successful (took {elapsed:.2f}s)")
            print(f"  Conversation ID: {data.get('conversation_id', 'N/A')}")
            print(f"  Inference time: {data.get('inference_time', 0):.2f}s")
            return True
        else:
            print(f"✗ Warmup failed: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Warmup request timed out after {WARMUP_TIMEOUT}s")
        return False
    except Exception as e:
        print(f"✗ Warmup request failed: {e}")
        return False


def main():
    """Main warmup sequence."""
    print("=" * 60)
    print("LLM Service Warmup")
    print("=" * 60)
    
    # Step 1: Wait for llama-cpp-server
    if not wait_for_service("llama-cpp-server", LLAMA_CPP_SERVER_URL):
        sys.exit(1)
    
    # Step 2: Wait for llm-service-api
    if not wait_for_service("llm-service-api", LLM_SERVICE_API_URL):
        sys.exit(1)
    
    # Step 3: Send warmup request
    if not send_warmup_request():
        print("\n⚠ Warning: Warmup request failed, but services are healthy.")
        print("  The stack may still work, but first request may be slower.")
        sys.exit(0)  # Don't fail the stack if warmup fails
    
    print("\n" + "=" * 60)
    print("✓ Warmup complete - Stack is ready!")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

