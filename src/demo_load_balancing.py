#!/usr/bin/env python3
"""
Demo: Load Balancing Demonstration
Shows how requests are distributed across multiple model instances
with fallback to Claude Sonnet 3.5 when primary models are overloaded
"""

import json
import openai
import time
import threading
import argparse
import signal
import sys
import yaml
from datetime import datetime
from collections import defaultdict

# Load configuration
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    port = config.get('litellm', {}).get('port', 4000)

client = openai.OpenAI(api_key="demo-key", base_url=f"http://0.0.0.0:{port}")

def print_router_settings():
    """Print router settings configuration as a formatted table"""
    print("ROUTER SETTINGS CONFIGURATION")
    print("=" * 80)
    
    # Extract router settings
    router_settings = config.get('router_settings', {})
    print(f"Routing Strategy:      {router_settings.get('routing_strategy', 'N/A')}")
    print()
    
    # Build model table - only show models used in load balancing
    model_list = config.get('model_list', [])
    fallback_models = set()
    
    # Extract fallback models from router settings - only for claude-sonnet-loadbalance-demo
    fallbacks = router_settings.get('fallbacks', [])
    for fallback_config in fallbacks:
        for primary, fallback_list in fallback_config.items():
            if primary == 'claude-sonnet-loadbalance-demo':
                fallback_models.update(fallback_list)
    
    # Process only models that participate in load balancing (claude-sonnet and fallbacks)
    relevant_models = []
    
    # Add primary models (claude-sonnet-loadbalance-demo)
    for model_config in model_list:
        model_name = model_config.get('model_name', 'unknown')
        if model_name == 'claude-sonnet-loadbalance-demo':
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
        "reset": "\033[0m"
    }
    color_code = colors.get(color, "")
    reset_code = colors["reset"] if color else ""
    print(f"{color_code}[{timestamp}] {message}{reset_code}")

def send_request(request_id, question):
    """Send a single request and track which model responds"""
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="claude-sonnet-loadbalance-demo",  # Load balances across instances, falls back to Sonnet 3.5
            messages=[{"role": "user", "content": question}],
            timeout=30
        )
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        # Extract model info from response
        model_used = getattr(response, 'model', 'unknown')
        
        # Determine if this is a fallback model
        is_fallback = "claude-3-5-sonnet" in model_used
        type_label = "FALLBACK!" if is_fallback else "PRIMARY: "
        color = "yellow" if is_fallback else "green"
        
        log_with_timestamp(
            f"Request #{request_id:2d} → {type_label} Model: {model_used:50s} | Time: {response_time:5.2f}s", 
            color
        )
        
        return {
            "request_id": request_id,
            "model_used": model_used,
            "response_time": response_time,
            "success": True
        }
        
    except Exception as e:
        log_with_timestamp(f"Request #{request_id:2d} → ERROR: {str(e)}", "red")
        return {
            "request_id": request_id,
            "model_used": "error",
            "response_time": 0,
            "success": False,
            "error": str(e)
        }

def demo_load_balancing(run_number=None):
    """Demonstrate load balancing across multiple model instances"""
    
    header = "LITELLM LOAD BALANCING DEMO"
    if run_number is not None:
        header += f" (Run #{run_number})"
    
    print("=" * 80)
    print(header)
    print("=" * 80)
    print("This demo shows how requests are distributed across multiple Claude Sonnet models")
    print("Requests to 'claude-sonnet-loadbalance-demo' are load balanced between Sonnet 4 and Sonnet 3.7 (Primary models)")
    print("When primary models reach their rate limits (3 RPM each), requests fallback to Sonnet 3.5 (50 RPM)")
    print()
    
    # Print router settings table
    print_router_settings()
    
    # Test questions
    questions = [
        "What is machine learning?",
        "Explain cloud computing briefly.",
        "What are microservices?",
        "Define artificial intelligence.",
        "What is serverless computing?",
        "Explain containerization.",
        "What is DevOps?",
        "Define data science.",
        "What is edge computing?",
        "Explain API design."
    ]
    
    results = []
    model_distribution = defaultdict(int)
    
    log_with_timestamp("Starting 10 concurrent requests to demonstrate load balancing...", "blue")
    print()
    
    # Send requests concurrently to better show load balancing
    threads = []
    results_lock = threading.Lock()
    
    def worker(req_id, question):
        result = send_request(req_id, question)
        with results_lock:
            results.append(result)
            if result["success"]:
                model_distribution[result["model_used"]] += 1
    
    # Launch concurrent requests
    for i, question in enumerate(questions, 1):
        thread = threading.Thread(target=worker, args=(i, question))
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Small delay to show request ordering
    
    # Wait for all requests to complete
    for thread in threads:
        thread.join()
    
    print()
    print("=" * 80)
    print("LOAD BALANCING RESULTS")
    print("=" * 80)
    
    successful_requests = len([r for r in results if r["success"]])
    failed_requests = len([r for r in results if not r["success"]])
    
    print(f"Total Requests:     {len(results)}")
    print(f"Successful:         {successful_requests}")
    print(f"Failed:             {failed_requests}")
    print()
    
    if model_distribution:
        print("Model Distribution:")
        for model, count in sorted(model_distribution.items()):
            percentage = (count / successful_requests) * 100 if successful_requests > 0 else 0
            print(f"  {model:50s} : {count:2d} requests ({percentage:5.1f}%)")
        
        print()
        if len(model_distribution) > 1:
            log_with_timestamp("LOAD BALANCING WORKING: Requests distributed across multiple models!", "green")
        else:
            log_with_timestamp("Only one model responded (normal if only one instance configured)", "yellow")
    
    avg_response_time = sum(r["response_time"] for r in results if r["success"]) / successful_requests if successful_requests > 0 else 0
    print(f"Average Response Time: {avg_response_time:.2f}s")
    
    print("=" * 80)
    
    return {
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "model_distribution": dict(model_distribution),
        "avg_response_time": avg_response_time
    }

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    shutdown_requested = True
    log_with_timestamp("🛑 Shutdown requested. Finishing current run...", "yellow")

def run_loop_mode(interval_seconds):
    """Run the demo in a continuous loop"""
    global shutdown_requested
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("CONTINUOUS LOAD BALANCING DEMO")
    print("=" * 80)
    print(f"Running load balancing demo every {interval_seconds} seconds")
    print("Press Ctrl+C to stop gracefully")
    print("=" * 80)
    print()
    
    run_count = 0
    total_stats = {
        "total_successful": 0,
        "total_failed": 0,
        "total_runs": 0,
        "cumulative_model_distribution": defaultdict(int)
    }
    
    try:
        while not shutdown_requested:
            run_count += 1
            
            log_with_timestamp(f"🚀 Starting run #{run_count}", "blue")
            
            # Run the demo
            results = demo_load_balancing(run_number=run_count)
            
            # Update cumulative stats
            total_stats["total_successful"] += results["successful_requests"]
            total_stats["total_failed"] += results["failed_requests"]
            total_stats["total_runs"] += 1
            
            for model, count in results["model_distribution"].items():
                total_stats["cumulative_model_distribution"][model] += count
            
            # Show cumulative stats
            print()
            print("CUMULATIVE STATISTICS")
            print("-" * 40)
            print(f"Total Runs:           {total_stats['total_runs']}")
            print(f"Total Successful:     {total_stats['total_successful']}")
            print(f"Total Failed:         {total_stats['total_failed']}")
            
            if total_stats["cumulative_model_distribution"]:
                print("\nCumulative Model Distribution:")
                total_requests = total_stats["total_successful"]
                for model, count in sorted(total_stats["cumulative_model_distribution"].items()):
                    percentage = (count / total_requests) * 100 if total_requests > 0 else 0
                    print(f"  {model:50s} : {count:3d} requests ({percentage:5.1f}%)")
            
            print("-" * 40)
            
            if not shutdown_requested:
                log_with_timestamp(f"⏳ Waiting {interval_seconds} seconds until next run...", "blue")
                
                # Wait with periodic checks for shutdown
                for _ in range(interval_seconds):
                    if shutdown_requested:
                        break
                    time.sleep(1)
                
                print()  # Add spacing between runs
    
    except KeyboardInterrupt:
        pass  # Handled by signal handler
    
    print()
    log_with_timestamp("✅ Demo stopped gracefully", "green")
    print()
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total Runs Completed: {total_stats['total_runs']}")
    print(f"Total Requests:       {total_stats['total_successful'] + total_stats['total_failed']}")
    print(f"Success Rate:         {(total_stats['total_successful'] / (total_stats['total_successful'] + total_stats['total_failed']) * 100):5.1f}%" if (total_stats['total_successful'] + total_stats['total_failed']) > 0 else "N/A")
    print("=" * 80)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Load Balancing Demo - Shows request distribution across model instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_load_balancing.py                    # Run once
  python demo_load_balancing.py --loop             # Run continuously every 30 seconds
  python demo_load_balancing.py --loop --interval 60  # Run every 60 seconds
        """
    )
    
    parser.add_argument(
        "--loop", 
        action="store_true",
        help="Run the demo continuously in a loop"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=30,
        help="Interval between runs in seconds (default: 30, only used with --loop)"
    )
    
    args = parser.parse_args()
    
    if args.loop:
        if args.interval < 5:
            print("Error: Minimum interval is 5 seconds")
            sys.exit(1)
        run_loop_mode(args.interval)
    else:
        demo_load_balancing()

if __name__ == "__main__":
    main()
