import time
import requests
import json
import logging
import os
from datetime import datetime

# Configure logging
LOG_FILE = "/logs/monitor_metrics.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SERVER_URL = "http://llm-server:8080"
METRICS_URL = f"{SERVER_URL}/metrics"
SLOTS_URL = f"{SERVER_URL}/slots"
HEALTH_URL = f"{SERVER_URL}/health"

def get_metrics():
    try:
        response = requests.get(METRICS_URL, timeout=30)
        if response.status_code == 200:
            return response.text
    except requests.exceptions.ReadTimeout:
        logging.warning("Server busy: Metrics request timed out (30s)")
    except Exception as e:
        logging.error(f"Failed to fetch metrics: {e}")
    return None

def get_slots():
    try:
        response = requests.get(SLOTS_URL, timeout=30)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ReadTimeout:
        logging.warning("Server busy: Slots request timed out (30s)")
    except Exception as e:
        logging.error(f"Failed to fetch slots: {e}")
    return None

def parse_prometheus_metrics(metrics_text):
    """
    Naive parser for Prometheus metrics format.
    Extracts key metrics like kv_cache_usage_ratio, requests_processing, etc.
    """
    parsed = {}
    if not metrics_text:
        return parsed

    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        
        parts = line.split(" ")
        if len(parts) >= 2:
            key = parts[0]
            val = parts[1]
            try:
                # Handle keys with labels if necessary, for now just taking the raw key
                # Example: llamacpp_kv_cache_usage_ratio 0.000000
                parsed[key] = float(val)
            except ValueError:
                pass
    return parsed

def log_system_status():
    """
    Polls server metrics and logs a structured JSON object to the log file.
    """
    metrics_text = get_metrics()
    slots_data = get_slots()
    
    parsed_metrics = parse_prometheus_metrics(metrics_text)
    
    # Calculate slot utilization
    total_slots = 0
    busy_slots = 0
    if slots_data:
        total_slots = len(slots_data)
        busy_slots = sum(1 for slot in slots_data if slot.get("state") != 0) # 0 is usually idle
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "kv_cache_usage": parsed_metrics.get("llamacpp_kv_cache_usage_ratio", 0),
        "requests_processing": parsed_metrics.get("llamacpp_requests_processing", 0),
        "requests_queued": parsed_metrics.get("llamacpp_requests_queued", 0),
        "tokens_second": parsed_metrics.get("llamacpp_tokens_predicted_total", 0), # This is a counter, would need delta manually if needed
        "slots_total": total_slots,
        "slots_busy": busy_slots,
        "slots_utilization_percent": (busy_slots / total_slots * 100) if total_slots > 0 else 0
    }
    
    logging.info(json.dumps(log_entry))

if __name__ == "__main__":
    logging.info("Starting LLaMA Server Monitor...")
    
    # Wait for server to be ready
    while True:
        try:
            r = requests.get(HEALTH_URL, timeout=10)
            if r.status_code == 200:
                logging.info("Server is healthy. Starting monitoring loop.")
                break
        except:
            pass
        time.sleep(5)

    # Monitoring loop
    while True:
        log_system_status()
        time.sleep(5) # Log every 5 seconds
