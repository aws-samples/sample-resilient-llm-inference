#!/usr/bin/env python3
"""
Demo: Clean Quota Isolation Demonstration
Shows how quota isolation prevents noisy neighbors from affecting other consumers
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load configuration
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    port = config.get('litellm', {}).get('port', 4000)

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

def send_consumer_request(consumer_id, api_key, request_id, question):
    """Send a single request for a specific consumer"""
    try:
        client = openai.OpenAI(api_key=api_key, base_url=f"http://0.0.0.0:{port}")
        start_time = time.time()
        
        # Each consumer uses their own model with different RPM limits
        model_mapping = {
            "A": "consumer-a-model",  # 3 RPM - gets throttled
            "B": "consumer-b-model",  # 10 RPM - should work fine  
            "C": "consumer-c-model"   # 10 RPM - should work fine
        }
        
        response = client.chat.completions.create(
            model=model_mapping[consumer_id],
            messages=[{"role": "user", "content": question}],
            timeout=10  # Reduced timeout for faster demo
        )
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        model_used = getattr(response, 'model', 'unknown')
        
        # Color-code by consumer
        consumer_colors = {"A": "magenta", "B": "cyan", "C": "blue"}
        color = consumer_colors.get(consumer_id, "")
        
        log_with_timestamp(
            f"{consumer_id} | SUCCESS      | Req #{request_id:2d} | {response_time:.2f}s", 
            "green"
        )
        
        return {
            "consumer_id": consumer_id,
            "request_id": request_id,
            "success": True,
            "model_used": model_used,
            "response_time": response_time,
            "error": None
        }
        
    except openai.RateLimitError as e:
        log_with_timestamp(
            f"{consumer_id} | RATE LIMITED | Req #{request_id:2d}", 
            "red"
        )
        return {
            "consumer_id": consumer_id,
            "request_id": request_id,
            "success": False,
            "model_used": None,
            "response_time": 0,
            "error": "RateLimitError"
        }
        
    except Exception as e:
        log_with_timestamp(
            f"{consumer_id} | ERROR        | Req #{request_id:2d} → {str(e)[:50]}...", 
            "red"
        )
        return {
            "consumer_id": consumer_id,
            "request_id": request_id,
            "success": False,
            "model_used": None,
            "response_time": 0,
            "error": str(e)
        }

def run_consumer_workload(consumer_id, api_key, num_requests, request_interval, consumer_type, start_time):
    """Run a workload for a specific consumer - all requests sent in parallel"""
    
    questions = [
        f"Hi #{consumer_id}!",
        f"Hello #{consumer_id}",
        f"Hey #{consumer_id}", 
        f"Test #{consumer_id}",
        f"Quick #{consumer_id}",
        f"Fast #{consumer_id}",
        f"Demo #{consumer_id}",
        f"Quota #{consumer_id}",
        f"Limit #{consumer_id}",
        f"Check #{consumer_id}"
    ]
    
    results = []
    results_lock = threading.Lock()
    
    def worker(req_id):
        question = questions[req_id % len(questions)]
        result = send_consumer_request(consumer_id, api_key, req_id + 1, question)
        with results_lock:
            results.append(result)
        return result
    
    # Send all requests in parallel immediately
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(worker, i) for i in range(num_requests)]
        # Wait for all to complete
        for future in as_completed(futures):
            pass  # Results already collected in worker
    
    return results

def demo_quota_isolation(run_number=None):
    """Demonstrate quota isolation preventing noisy neighbor problems"""
    
    header = "LITELLM QUOTA ISOLATION DEMO - THROTTLING NOISY NEIGHBORS"
    if run_number is not None:
        header += f" (Run #{run_number})"
    
    print("=" * 80)
    print(header)
    print("=" * 80)
    print("This demo shows how quota isolation prevents one noisy consumer from")
    print("affecting other consumers.")
    print()
    
    # Display consumer configuration table with actual config values
    print("| Consumer | RPM (quota) | # Requests | Type   |")
    print("|----------|-------------|------------|--------|")
    
    # Read RPM values from config for each consumer model
    consumer_models = {
        "A": "consumer-a-model",
        "B": "consumer-b-model", 
        "C": "consumer-c-model"
    }
    
    for consumer_id in ["A", "B", "C"]:
        model_name = consumer_models[consumer_id]
        # Find the RPM value in config
        rpm_value = "N/A"
        num_requests = 5  # Default from consumers list
        
        for model_config in config.get('model_list', []):
            if model_config.get('model_name') == model_name:
                rpm_value = model_config.get('rpm', 'N/A')
                break
        
        consumer_type = "NOISY" if consumer_id == "A" else "NORMAL"
        print(f"| {consumer_id}        | {rpm_value:<11} | {num_requests:<10} | {consumer_type:<6} |")
    
    print()
    
    # Define consumers with 5 requests each sent in parallel
    consumers = [
        {"id": "A", "key": "consumer-a-key", "requests": 5, "interval": 0, "type": "NOISY"},     # 5 parallel requests (exceeds 3 RPM limit)  
        {"id": "B", "key": "consumer-b-key", "requests": 5, "interval": 0, "type": "NORMAL"},   # 5 parallel requests (within 10 RPM limit)
        {"id": "C", "key": "consumer-c-key", "requests": 5, "interval": 0, "type": "NORMAL"}    # 5 parallel requests (within 10 RPM limit)
    ]
    
    print("All consumers will send 5 requests in parallel simultaneously.")
    print("Quota isolation prevents Consumer A's throttling from affecting B and C.")
    print("Expected behavior based on individual quotas:")
    print("  • Consumer A (NOISY):  5 requests > 3 RPM quota  → 2 requests should be throttled")
    print("  • Consumer B (NORMAL): 5 requests < 10 RPM quota → all 5 requests should succeed")
    print("  • Consumer C (NORMAL): 5 requests < 10 RPM quota → all 5 requests should succeed")
    print()
    
    log_with_timestamp("Starting consumers...", "blue")
    
    # Record start time for synchronized execution
    demo_start_time = time.time()
    all_results = {}
    
    # Run all consumers concurrently to simulate real-world noisy neighbor scenario
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_consumer = {}
        
        for consumer in consumers:
            future = executor.submit(
                run_consumer_workload,
                consumer["id"],
                consumer["key"], 
                consumer["requests"],
                consumer["interval"],
                consumer["type"],
                demo_start_time
            )
            future_to_consumer[future] = consumer["id"]
        
        for future in as_completed(future_to_consumer):
            consumer_id = future_to_consumer[future]
            results = future.result()
            all_results[consumer_id] = results
    
    total_time = time.time() - demo_start_time
    log_with_timestamp(f"Demo completed in {total_time:.1f} seconds", "blue")
    
    return analyze_and_display_results(all_results, consumers)

def analyze_and_display_results(all_results, consumers):
    """Analyze results and display clear statistics"""
    
    # Calculate stats for all consumers
    consumer_stats = {}
    
    for consumer in consumers:
        consumer_id = consumer["id"]
        results = all_results.get(consumer_id, [])
        
        successful = len([r for r in results if r["success"]])
        failed = len([r for r in results if not r["success"]])
        rate_limited = len([r for r in results if r.get("error") == "RateLimitError"])
        
        success_rate = (successful / len(results)) * 100 if results else 0
        avg_response_time = sum(r["response_time"] for r in results if r["success"]) / successful if successful > 0 else 0
        
        consumer_stats[consumer_id] = {
            "type": consumer["type"],
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "rate_limited": rate_limited,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time
        }
    
    print()
    print("=" * 80)
    print("QUOTA ISOLATION ANALYSIS")
    print("=" * 80)
    
    # Show consolidated success rate summary as a table
    print("Consumer   | Type   | Success Rate | Total | Success | Failed | Rate Limited | Avg Time")
    print("-----------+--------+--------------+-------+---------+--------+--------------+---------")
    
    for consumer_id in ["A", "B", "C"]:
        if consumer_id in consumer_stats:
            stats = consumer_stats[consumer_id]
            
            # Format the line content
            line_content = (f"{consumer_id}        | {stats['type']:6s} | "
                          f"{stats['success_rate']:5.1f}%{' ':>7s} | "
                          f"{stats['total']:5d} | {stats['successful']:7d} | "
                          f"{stats['failed']:6d} | {stats['rate_limited']:12d} | "
                          f"{stats['avg_response_time']:7.2f}s")
            
            # For 100% success rate, show entire line in white (no color)
            if stats['success_rate'] == 100.0:
                print(line_content)
            else:
                # Highlight entire line in red for non-100% success rates
                print(f"\033[91m{line_content}\033[0m")
    
    print()

    # Analyze isolation effectiveness with enhanced feedback
    noisy_rate_limited = consumer_stats.get("A", {}).get("rate_limited", 0) > 0
    normal_b_success_rate = consumer_stats.get("B", {}).get("success_rate", 0)
    normal_c_success_rate = consumer_stats.get("C", {}).get("success_rate", 0)
    
    return {
        "consumer_stats": consumer_stats,
        "isolation_effective": normal_b_success_rate >= 80 and normal_c_success_rate >= 80,
        "noisy_rate_limited": noisy_rate_limited
    }

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    shutdown_requested = True
    log_with_timestamp("Shutdown requested. Finishing current run...", "yellow")

def run_loop_mode(interval_seconds):
    """Run the demo in a continuous loop"""
    global shutdown_requested
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("CONTINUOUS QUOTA ISOLATION DEMO")
    print("=" * 80)
    print(f"Running quota isolation demo every {interval_seconds} seconds")
    print("Press Ctrl+C to stop gracefully")
    print("=" * 80)
    print()
    
    run_count = 0
    total_stats = {
        "total_runs": 0,
        "effective_isolation_runs": 0,
        "consumer_cumulative_stats": defaultdict(lambda: {"successful": 0, "failed": 0, "rate_limited": 0})
    }
    
    try:
        while not shutdown_requested:
            run_count += 1
            
            log_with_timestamp(f"Starting run #{run_count}", "blue")
            
            # Run the demo
            results = demo_quota_isolation(run_number=run_count)
            
            # Update cumulative stats
            total_stats["total_runs"] += 1
            if results["isolation_effective"]:
                total_stats["effective_isolation_runs"] += 1
            
            for consumer_id, stats in results["consumer_stats"].items():
                cumulative = total_stats["consumer_cumulative_stats"][consumer_id]
                cumulative["successful"] += stats["successful"]
                cumulative["failed"] += stats["failed"]
                cumulative["rate_limited"] += stats["rate_limited"]
            
            # Show cumulative stats
            print()
            print("CUMULATIVE STATISTICS")
            print("-" * 40)
            print(f"Total Runs:                {total_stats['total_runs']}")
            print(f"Effective Isolation Runs:  {total_stats['effective_isolation_runs']}")
            
            isolation_percentage = (total_stats['effective_isolation_runs'] / total_stats['total_runs'] * 100) if total_stats['total_runs'] > 0 else 0
            print(f"Isolation Success Rate:    {isolation_percentage:5.1f}%")
            
            print("\nCumulative Consumer Stats:")
            for consumer_id in ["A", "B", "C"]:
                if consumer_id in total_stats["consumer_cumulative_stats"]:
                    cum_stats = total_stats["consumer_cumulative_stats"][consumer_id]
                    total_requests = cum_stats["successful"] + cum_stats["failed"]
                    success_rate = (cum_stats["successful"] / total_requests * 100) if total_requests > 0 else 0
                    
                    print(f"  Consumer-{consumer_id}: {cum_stats['successful']:3d} success, {cum_stats['rate_limited']:3d} rate limited ({success_rate:5.1f}% success rate)")
            
            print("-" * 40)
            
            if not shutdown_requested:
                log_with_timestamp(f"Waiting {interval_seconds} seconds until next run...", "blue")
                
                # Wait with periodic checks for shutdown
                for _ in range(interval_seconds):
                    if shutdown_requested:
                        break
                    time.sleep(1)
                
                print()  # Add spacing between runs
    
    except KeyboardInterrupt:
        pass  # Handled by signal handler
    
    print()
    log_with_timestamp("Demo stopped gracefully", "green")
    print()
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total Runs Completed:      {total_stats['total_runs']}")
    print(f"Effective Isolation Runs:  {total_stats['effective_isolation_runs']}")
    
    final_isolation_rate = (total_stats['effective_isolation_runs'] / total_stats['total_runs'] * 100) if total_stats['total_runs'] > 0 else 0
    print(f"Overall Isolation Success: {final_isolation_rate:5.1f}%")
    print("=" * 80)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Quota Isolation Demo - Shows how quotas prevent noisy neighbor problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_quota_isolation_clean.py                    # Run once
  python demo_quota_isolation_clean.py --loop             # Run continuously every 30 seconds
  python demo_quota_isolation_clean.py --loop --interval 60  # Run every 60 seconds
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
        demo_quota_isolation()

if __name__ == "__main__":
    main()