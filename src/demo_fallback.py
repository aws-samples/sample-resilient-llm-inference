#!/usr/bin/env python3
"""
Demo: Fallback Mechanism Demonstration
Shows automatic fallback from Claude models to Nova models when errors occur
"""

import openai
import time
import threading
import yaml
from datetime import datetime
from collections import defaultdict

# Load configuration with validation
try:
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        port = config.get('litellm', {}).get('port', 4000)
        # Validate port range
        if not isinstance(port, int) or port < 1024 or port > 65535:
            print("Warning: Invalid port in config, using default 4000")
            port = 4000
except Exception as e:
    print("Error loading config file, using defaults")
    config = {}
    port = 4000

client = openai.OpenAI(api_key="demo-key", base_url=f"http://0.0.0.0:{port}")

def print_router_settings():
    """Print router settings configuration as a formatted table"""
    print("ROUTER SETTINGS CONFIGURATION")
    print("=" * 80)
    
    # Extract router settings
    router_settings = config.get('router_settings', {})
    print(f"Routing Strategy:      {router_settings.get('routing_strategy', 'N/A')}")
    print()
    
    # Build model table - only show models used in fallback demo
    model_list = config.get('model_list', [])
    fallback_models = set()
    
    # Extract fallback models from router settings
    fallbacks = router_settings.get('fallbacks', [])
    for fallback_config in fallbacks:
        for primary, fallback_list in fallback_config.items():
            if primary == 'claude-sonnet-fallback-demo':
                fallback_models.update(fallback_list)
    
    # Process only models that participate in fallback demo
    relevant_models = []
    
    # Add primary models (claude-sonnet-fallback-demo)
    for model_config in model_list:
        model_name = model_config.get('model_name', 'unknown')
        if model_name == 'claude-sonnet-fallback-demo':
            rpm = model_config.get('rpm', 'N/A')
            litellm_params = model_config.get('litellm_params', {})
            actual_model = litellm_params.get('model', 'unknown')
            
            # Remove 'bedrock/' prefix if present
            if actual_model.startswith('bedrock/'):
                actual_model = actual_model[8:]
            
            relevant_models.append((actual_model, rpm, "Primary"))
    
    # Add fallback models
    for model_config in model_list:
        model_name = model_config.get('model_name', 'unknown')
        if model_name in fallback_models:
            rpm = model_config.get('rpm', 'N/A')
            litellm_params = model_config.get('litellm_params', {})
            actual_model = litellm_params.get('model', 'unknown')
            
            # Remove 'bedrock/' prefix if present
            if actual_model.startswith('bedrock/'):
                actual_model = actual_model[8:]
            
            relevant_models.append((actual_model, rpm, "Fallback"))
    
    # Print aligned table
    print("|-----------------------------------------------|-----------------------------|-----------| ")
    print("| Model                                         | Max Requests Per Min (RPM)  | Type      |")
    print("|-----------------------------------------------|-----------------------------|-----------| ")
    
    for model, rpm, model_type in relevant_models:
        print(f"| {model:<45} | {rpm:<27} | {model_type:<9} |")
    
    print()

def log_with_timestamp(message, color=""):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    colors = {
        "green": "\033[92m",
        "blue": "\033[94m", 
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "reset": "\033[0m"
    }
    color_code = colors.get(color, "")
    reset_code = colors["reset"] if color else ""
    print(f"{color_code}[{timestamp}] {message}{reset_code}")

def trigger_rate_limit_scenario():
    """Send rapid requests to trigger rate limiting and demonstrate fallback"""
    
    print("=" * 80)
    print("LITELLM FALLBACK DEMO")
    print("=" * 80)
    print("This demo triggers rate limits on Claude models to show fallback to Sonnet 3.5 models")
    print("Watch for model switches from Claude → Sonnet 3.5 when limits are hit")
    print()
    
    # Print router settings table
    print_router_settings()
    
    results = []
    model_usage = defaultdict(int)
    fallback_events = []
    
    log_with_timestamp("Sending 10 requests in parallel to trigger rate limits...", "blue")
    print()

    # Send 10 rapid requests to exceed the 5 RPM limit for Claude
    questions = [
        "What is AI?", "Define ML", "Explain NLP", "What is DL?", "Define CNN",
        "What is RNN?", "Explain GAN", "Define API", "What is REST?", "Explain GraphQL"
    ]
    
    # Send requests concurrently to better trigger rate limits
    threads = []
    results_lock = threading.Lock()
    
    def worker(req_id, question):
        try:
            start_time = time.time()
            
            log_with_timestamp(f"Request #{req_id:2d}: Asking '{question}'...", "cyan")
            
            response = client.chat.completions.create(
                model="claude-sonnet-fallback-demo",  # This should fallback to Sonnet 3.5 when rate limited
                messages=[{"role": "user", "content": question}],
                timeout=30
            )
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            model_used = getattr(response, 'model', 'unknown')
            
            # Check if this was a fallback (Sonnet 3.5 model used instead of primary Claude)
            is_fallback = 'sonnet-3-5' in model_used.lower() or '3-5-sonnet' in model_used.lower()
            
            if is_fallback:
                log_with_timestamp(
                    f"Request #{req_id:2d} → FALLBACK! Model: {model_used:<45} | Time: {response_time:5.2f}s", 
                    "yellow"
                )
            else:
                log_with_timestamp(
                    f"Request #{req_id:2d} → PRIMARY:  Model: {model_used:<45} | Time: {response_time:5.2f}s", 
                    "green"
                )
            
            # Thread-safe updates to shared data
            with results_lock:
                model_usage[model_used] += 1
                results.append({
                    "request_id": req_id,
                    "model": model_used,
                    "response_time": response_time,
                    "is_fallback": is_fallback,
                    "success": True
                })
                if is_fallback:
                    fallback_events.append({
                        "request_id": req_id,
                        "model": model_used,
                        "time": response_time
                    })
            
        except openai.RateLimitError as e:
            log_with_timestamp(f"Request #{req_id:2d} → RATE LIMIT: {str(e)}", "red")
            with results_lock:
                results.append({
                    "request_id": req_id,
                    "error": "RateLimitError",
                    "success": False
                })
                
        except openai.APIError as e:
            log_with_timestamp(f"Request #{req_id:2d} → API ERROR: {str(e)}", "red")
            with results_lock:
                results.append({
                    "request_id": req_id,
                    "error": "APIError", 
                    "success": False
                })
                
        except Exception as e:
            log_with_timestamp(f"Request #{req_id:2d} → ERROR: {str(e)}", "red")
            with results_lock:
                results.append({
                    "request_id": req_id,
                    "error": str(e),
                    "success": False
                })
    
    # Launch concurrent requests
    for i, question in enumerate(questions, 1):
        thread = threading.Thread(target=worker, args=(i, question))
        threads.append(thread)
        thread.start()
        time.sleep(0.05)  # Small delay to avoid overwhelming the API
    
    # Wait for all requests to complete
    for thread in threads:
        thread.join()
    
    print()
    print("=" * 80)
    print("FALLBACK DEMONSTRATION RESULTS")
    print("=" * 80)
    
    successful_requests = len([r for r in results if r.get("success", False)])
    failed_requests = len([r for r in results if not r.get("success", False)])
    fallback_count = len(fallback_events)
    primary_count = successful_requests - fallback_count
    
    print(f"Total Requests:        {len(results)}")
    print(f"Successful:            {successful_requests}")
    print(f"Failed:                {failed_requests}")
    print(f"Primary Model Used:    {primary_count}")
    print(f"Fallback Triggered:    {fallback_count}")
    print()
    
    if model_usage:
        print("Model Usage Distribution:")
        # Calculate the maximum width needed for proper alignment
        max_model_len = max(len(model) for model in model_usage.keys()) if model_usage else 0
        
        for model, count in sorted(model_usage.items()):
            model_type = "FALLBACK" if ('sonnet-3-5' in model.lower() or '3-5-sonnet' in model.lower()) else "PRIMARY "
            percentage = (count / successful_requests) * 100 if successful_requests > 0 else 0
            # Create properly aligned output with fixed-width type labels
            print(f"  {model_type} {model:<{max_model_len}} : {count:2d} requests ({percentage:5.1f}%)")
    
    print()
    
    if fallback_events:
        log_with_timestamp(f"FALLBACK WORKING: {fallback_count} requests successfully failed over to Sonnet 3.5 models!", "green")
        print()
        print("Fallback Events Detail:")
        # Calculate alignment width for request IDs and model names
        max_req_id_len = max(len(str(event['request_id'])) for event in fallback_events)
        max_model_len = max(len(event['model']) for event in fallback_events)
        
        for event in fallback_events:
            req_id = f"Request #{event['request_id']:>{max_req_id_len}}"
            print(f"  {req_id} → {event['model']:<{max_model_len}} (Response: {event['time']:5.2f}s)")
    else:
        log_with_timestamp("No fallbacks triggered (may need to increase request rate or check config)", "yellow")
    
    print("=" * 80)
    print()

if __name__ == "__main__":
    trigger_rate_limit_scenario()