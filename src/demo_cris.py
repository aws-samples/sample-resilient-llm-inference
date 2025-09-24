#!/usr/bin/env python3
"""
Demo: CRIS (Cross-Region Inference Service) Demonstration
Shows how cross-region inference automatically distributes requests across multiple AWS regions
"""

import json
import time
import threading
import argparse
import signal
import sys
import yaml
import re
import boto3
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load configuration
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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

def get_aws_profile():
    """Get AWS profile from config"""
    try:
        profile = config.get('aws', {}).get('profile_name', 'default')
        # Basic validation for profile name
        if not isinstance(profile, str) or not profile.strip():
            return 'default'
        return profile.strip()
    except:
        return 'default'

def get_aws_region():
    """Get AWS region from config"""
    try:
        region = config.get('aws', {}).get('region_name', 'us-east-1')
        # Basic validation for AWS region format
        if not isinstance(region, str) or not region.strip():
            return 'us-east-1'
        # Simple region format validation (basic pattern)
        import re
        if not re.match(r'^[a-z]{2}(-[a-z]+)*-[0-9]+$', region.strip()):
            log_with_timestamp(f"Invalid region format '{region}', using default", "yellow")
            return 'us-east-1'
        return region.strip()
    except:
        return 'us-east-1'

def get_log_group_name():
    """Get CloudWatch log group name from config"""
    try:
        log_group = config.get('aws', {}).get('bedrock_log_group_name', 'BedrockModelInvocation')
        # Basic validation for log group name
        if not isinstance(log_group, str) or not log_group.strip():
            return 'BedrockModelInvocation'
        # CloudWatch log group name validation (basic pattern)
        import re
        if not re.match(r'^[\w\.\-_/]+$', log_group.strip()):
            log_with_timestamp(f"Invalid log group name format, using default", "yellow")
            return 'BedrockModelInvocation'
        return log_group.strip()
    except:
        return 'BedrockModelInvocation'

def start_cloudwatch_query(logs_client, model_id, start_timestamp_ms, expected_count):
    """Start a CloudWatch Logs Insights query for the specific model"""
    try:
        # Query specifically for the model used in this demo run
        # The modelId in logs is stored as ARN with inference-profile, so we need to match on the suffix
        query_string = f"""
        filter modelId like /inference-profile\/{model_id}/
        and toMillis(@timestamp) >= {start_timestamp_ms}
        | stats count(*) as invocationCount by inferenceRegion
        | sort inferenceRegion
        """
        
        # Calculate end time (start + 5 minutes buffer)
        end_timestamp = int(time.time())
        
        log_with_timestamp(f"Starting CloudWatch query for model: {model_id}", "blue")
        
        log_group_name = get_log_group_name()
        response = logs_client.start_query(
            logGroupName=log_group_name,
            startTime=int(start_timestamp_ms / 1000),
            endTime=end_timestamp,
            queryString=query_string.strip()
        )
        
        return response['queryId']
    except Exception as e:
        log_with_timestamp(f"Failed to start CloudWatch query: {str(e)}", "red")
        return None

def wait_for_query_completion(logs_client, query_id, timeout_seconds=60):
    """Wait for CloudWatch query to complete"""
    try:
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = logs_client.describe_queries()
                
                # Find our query by ID
                our_query = None
                for query in response.get('queries', []):
                    if query.get('queryId') == query_id:
                        our_query = query
                        break
                
                if not our_query:
                    log_with_timestamp("Query not found", "red")
                    return False
                
                status = our_query['status']
                
                if status == 'Complete':
                    log_with_timestamp("CloudWatch query completed", "green")
                    return True
                elif status in ['Failed', 'Cancelled']:
                    log_with_timestamp(f"CloudWatch query {status.lower()}", "red")
                    return False
                elif status == 'Running':
                    pass
                
                time.sleep(2)
            except Exception as e:
                log_with_timestamp(f"Error checking query status: {str(e)}", "red")
                return False
        
        log_with_timestamp("CloudWatch query timed out", "yellow")
        return False
    except Exception as e:
        log_with_timestamp(f"Error waiting for query: {str(e)}", "red")
        return False

def get_cloudwatch_results(logs_client, query_id):
    """Retrieve and parse CloudWatch query results"""
    try:
        response = logs_client.get_query_results(queryId=query_id)
        
        if not response.get('results'):
            return {}
        
        # Parse results into region -> count mapping
        region_distribution = {}
        total_invocations = 0
        
        for row in response['results']:
            if len(row) >= 2:
                region = row[0]['value']
                count = int(row[1]['value'])
                region_distribution[region] = count
                total_invocations += count
        
        return {
            'region_distribution': region_distribution,
            'total_invocations': total_invocations
        }
    except Exception as e:
        log_with_timestamp(f"Error retrieving CloudWatch results: {str(e)}", "red")
        return {}

def query_cloudwatch_distribution(model_id, start_timestamp_ms, expected_count):
    """Query CloudWatch for actual regional distribution"""
    try:
        # Get AWS region and create CloudWatch client
        aws_profile = get_aws_profile()
        aws_region = get_aws_region()

        session = boto3.Session(profile_name=aws_profile)
        logs_client = session.client('logs', region_name=aws_region)
        
        # Test AWS credentials
        try:
            logs_client.describe_log_groups(limit=1)
        except Exception as e:
            log_with_timestamp("AWS credentials not configured or insufficient permissions", "yellow")
            return None
        
        # Start the query
        query_id = start_cloudwatch_query(logs_client, model_id, start_timestamp_ms, expected_count)
        if not query_id:
            return None
        
        # Wait for completion
        if not wait_for_query_completion(logs_client, query_id):
            return None
        
        # Get results
        return get_cloudwatch_results(logs_client, query_id)
        
    except Exception as e:
        log_with_timestamp(f"CloudWatch analysis failed: {str(e)}", "yellow")
        return None

def send_bedrock_request(request_id, model_id, question):
    """Send a single request directly to Bedrock"""
    try:
        # Create Bedrock client with profile and region from config
        aws_profile = get_aws_profile()
        aws_region = get_aws_region()

        session = boto3.Session(profile_name=aws_profile)
        bedrock_client = session.client('bedrock-runtime', region_name=aws_region)

        start_time = time.time()

        # Prepare the request body for Claude models
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": question}]
        }

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        # Parse response
        response_body = json.loads(response['body'].read())
        content = response_body.get('content', [{}])[0].get('text', 'No content')

        log_with_timestamp(
            f"Request #{request_id:2d} | SUCCESS | {response_time:.2f}s",
            "green"
        )

        return {
            "request_id": request_id,
            "success": True,
            "model_used": model_id,
            "response_time": response_time,
            "content": content[:50] + "..." if len(content) > 50 else content,
            "error": None
        }

    except Exception as e:
        error_msg = str(e)
        if "ThrottlingException" in error_msg or "ServiceQuota" in error_msg:
            log_with_timestamp(
                f"Request #{request_id:2d} | THROTTLED",
                "yellow"
            )
            error_type = "ThrottlingException"
        else:
            # Sanitize error message to avoid information leakage
            sanitized_error = "API Error"
            if "ValidationException" in error_msg:
                sanitized_error = "Validation Error"
            elif "AccessDenied" in error_msg:
                sanitized_error = "Access Denied"
            elif "ResourceNotFound" in error_msg:
                sanitized_error = "Resource Not Found"

            log_with_timestamp(
                f"Request #{request_id:2d} | ERROR | {sanitized_error}",
                "red"
            )
            error_type = "Error"

        return {
            "request_id": request_id,
            "success": False,
            "model_used": None,
            "response_time": 0,
            "content": None,
            "error": error_type
        }

def run_cris_demo(num_requests=20):
    """Demonstrate CRIS cross-region distribution"""
    
    print("=" * 80)
    print("AMAZON BEDROCK CROSS-REGION INFERENCE (CRIS) DEMO")
    print("=" * 80)
    
    # Get model ID from config
    model_id = config.get('cris', {}).get('model_id', 'us.anthropic.claude-sonnet-4-20250514-v1:0')
    
    
    print(f"Sending {num_requests} requests in parallel to demonstrate cross-region distribution.")
    print(f"Cross-Region inference profile: {model_id}")
    print("CRIS will automatically route requests to available regions for optimal performance")
    print()
    
    log_with_timestamp("Starting CRIS requests...", "blue")
    
    # Prepare questions for requests
    questions = [
        "Hello from CRIS request!",
        "Test message for cross-region demo",
        "CRIS demo request",
        "Cross-region test message",
        "Regional distribution test"
    ]
    
    results = []
    results_lock = threading.Lock()
    demo_start_time = time.time()
    demo_start_timestamp_ms = int(demo_start_time * 1000)  # Convert to milliseconds for CloudWatch
    
    def worker(req_id):
        question = questions[req_id % len(questions)]
        result = send_bedrock_request(req_id + 1, model_id, question)
        with results_lock:
            results.append(result)
        return result
    
    # Send all requests in parallel
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(worker, i) for i in range(num_requests)]
        # Wait for all to complete
        for future in as_completed(futures):
            pass  # Results already collected in worker
    
    total_time = time.time() - demo_start_time
    log_with_timestamp(f"Demo completed in {total_time:.1f} seconds", "blue")
    
    return analyze_cris_results(results, total_time, model_id, demo_start_timestamp_ms)

def analyze_cris_results(results, total_time, model_id, demo_start_timestamp_ms):
    """Analyze CRIS results and display regional distribution"""
    
    print()
    print("=" * 80)
    print("CRIS ANALYSIS")
    print("=" * 80)
    
    # Calculate basic stats
    successful = len([r for r in results if r["success"]])
    failed = len([r for r in results if not r["success"]])
    total = len(results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    # Calculate average response times
    successful_results = [r for r in results if r["success"]]
    avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results) if successful_results else 0
    
    
    # CloudWatch Logs Analysis - Real Regional Distribution
    if successful > 0:
        
        # Display query parameters
        log_group_name = get_log_group_name()
        generous_timestamp_ms = demo_start_timestamp_ms - (60 * 1000)
        start_time_readable = datetime.fromtimestamp(generous_timestamp_ms / 1000).strftime('%H:%M:%S')
        end_time_readable = datetime.fromtimestamp(int(time.time())).strftime('%H:%M:%S')
        
        print(f"Query Parameters:")
        print(f"  Log Group:    {log_group_name}")
        print(f"  Model Filter: {model_id}")
        print(f"  Time Range:   {start_time_readable} - {end_time_readable}")
        print()
        
        log_with_timestamp("Waiting for logs to propagate to CloudWatch...", "blue")
        
        # Try multiple times with longer waits for logs to propagate
        cloudwatch_results = None
        max_attempts = 5
        for attempt in range(max_attempts):
            if attempt == 0:
                wait_time = 60  # First attempt: 60s after demo completion
                log_with_timestamp(f"Waiting 60s for logs to propagate... (1/{max_attempts})", "blue")
            else:
                wait_time = 30  # Subsequent attempts: 30s each
                log_with_timestamp(f"Waiting 30s for logs to propagate... ({attempt + 1}/{max_attempts})", "blue")
            time.sleep(wait_time)
            
            # Use a more generous timestamp (subtract 1 minute from start time)
            generous_timestamp_ms = demo_start_timestamp_ms - (60 * 1000)
            cloudwatch_results = query_cloudwatch_distribution(model_id, generous_timestamp_ms, successful)
            
            # Only consider results complete if we get the expected number of invocations
            if cloudwatch_results and cloudwatch_results.get('total_invocations', 0) == successful:
                break
            elif attempt < 4:
                found_count = cloudwatch_results.get('total_invocations', 0) if cloudwatch_results else 0
                if found_count > 0:
                    log_with_timestamp(f"Found {found_count}/{successful} requests, will retry for complete data...", "yellow")
                else:
                    log_with_timestamp("No data found yet, will retry...", "yellow")
        
        if cloudwatch_results and cloudwatch_results.get('region_distribution'):
            region_dist = cloudwatch_results['region_distribution']
            total_cw_invocations = cloudwatch_results['total_invocations']
            
            # Display CloudWatch results in table format
            print()
            print("Region       | Invocations | Percentage")
            print("-------------|-------------|------------")
            
            for region in sorted(region_dist.keys()):
                count = region_dist[region]
                percentage = (count / total_cw_invocations * 100) if total_cw_invocations > 0 else 0
                print(f"{region:<12s} | {count:>11d} | {percentage:>9.1f}%")
            
        else:
            log_with_timestamp("CloudWatch analysis unavailable (check AWS credentials/permissions)", "yellow")
    
    print("=" * 80)
    
    return {
        "total_requests": total,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "total_time": total_time
    }

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="CRIS Demo - Shows cross-region inference request distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_cris.py                    # Run with 20 requests (default)
  python demo_cris.py --requests 50      # Run with 50 requests
  python demo_cris.py --requests 10      # Run with 10 requests for quick test
        """
    )
    
    parser.add_argument(
        "--requests", 
        type=int, 
        default=20,
        help="Number of parallel requests to send (default: 20)"
    )
    
    args = parser.parse_args()

    # Enhanced input validation
    if not isinstance(args.requests, int):
        print("Error: Number of requests must be an integer")
        sys.exit(1)

    if args.requests < 1:
        print("Error: Number of requests must be at least 1")
        sys.exit(1)

    if args.requests > 100:
        print("Error: Maximum 100 requests allowed to avoid overwhelming the system")
        sys.exit(1)
    
    run_cris_demo(args.requests)

if __name__ == "__main__":
    main()