#!/usr/bin/env python3
"""
Demo: Account Sharding for LLM Inference
Shows how to distribute requests across multiple AWS accounts (account sharding) for quota isolation and increased capacity
Uses CloudWatch logs to show real regional distribution across both accounts
"""

import json
import time
import threading
import argparse
import sys
import yaml
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

def get_aws_profiles():
    """Get primary and secondary AWS profiles from config"""
    aws_config = config.get('aws', {})
    primary = aws_config.get('profile_name', 'default')
    secondary = aws_config.get('secondary_profile_name', 'default-secondary')
    return primary, secondary

def get_aws_region():
    """Get AWS region from config"""
    return config.get('aws', {}).get('region_name', 'us-east-1')

def get_log_group_name():
    """Get CloudWatch log group name from config"""
    return config.get('aws', {}).get('bedrock_log_group_name', 'BedrockModelInvocation')

def get_model_id():
    """Get model ID for cross-account demo"""
    return config.get('cris', {}).get('model_id', 'us.anthropic.claude-sonnet-4-20250514-v1:0')

def verify_aws_credentials(profile_name, account_type):
    """Verify AWS credentials for a profile"""
    try:
        session = boto3.Session(profile_name=profile_name)
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        account_id = identity['Account']
        return account_id
    except Exception as e:
        log_with_timestamp(f"Failed to verify {account_type} credentials: {str(e)}", "red")
        return None

def start_cloudwatch_query(logs_client, model_id, start_timestamp_ms, expected_count, account_type):
    """Start a CloudWatch Logs Insights query for the specific model"""
    try:
        # Query specifically for the model used in this demo run
        # The modelId in logs is stored as ARN with inference-profile, so we need to match on the suffix
        query_string = f"""
        filter modelId like /inference-profile\\/{model_id}/
        and toMillis(@timestamp) >= {start_timestamp_ms}
        | stats count(*) as invocationCount by inferenceRegion
        | sort inferenceRegion
        """

        # Calculate end time (start + 5 minutes buffer)
        end_timestamp = int(time.time())

        log_with_timestamp(f"Starting CloudWatch query for {account_type} account, model: {model_id}", "blue")

        log_group_name = get_log_group_name()
        response = logs_client.start_query(
            logGroupName=log_group_name,
            startTime=int(start_timestamp_ms / 1000),
            endTime=end_timestamp,
            queryString=query_string.strip()
        )

        return response['queryId']
    except Exception as e:
        log_with_timestamp(f"Failed to start CloudWatch query for {account_type}: {str(e)}", "red")
        return None

def wait_for_query_completion(logs_client, query_id, account_type, timeout_seconds=60):
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
                    log_with_timestamp(f"{account_type} query not found", "red")
                    return False

                status = our_query['status']

                if status == 'Complete':
                    log_with_timestamp(f"{account_type} CloudWatch query completed", "green")
                    return True
                elif status in ['Failed', 'Cancelled']:
                    log_with_timestamp(f"{account_type} CloudWatch query {status.lower()}", "red")
                    return False
                elif status == 'Running':
                    pass

                time.sleep(2)
            except Exception as e:
                log_with_timestamp(f"Error checking {account_type} query status: {str(e)}", "red")
                return False

        log_with_timestamp(f"{account_type} CloudWatch query timed out", "yellow")
        return False
    except Exception as e:
        log_with_timestamp(f"Error waiting for {account_type} query: {str(e)}", "red")
        return False

def get_cloudwatch_results(logs_client, query_id, account_type):
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
            'total_invocations': total_invocations,
            'account_type': account_type
        }
    except Exception as e:
        log_with_timestamp(f"Error retrieving {account_type} CloudWatch results: {str(e)}", "red")
        return {}

def query_cloudwatch_distribution_for_account(profile_name, account_type, model_id, start_timestamp_ms, expected_count):
    """Query CloudWatch for actual regional distribution for a specific account"""
    try:
        # Get AWS region and create CloudWatch client for specific profile
        aws_region = get_aws_region()
        session = boto3.Session(profile_name=profile_name)
        logs_client = session.client('logs', region_name=aws_region)

        # Test AWS credentials
        try:
            logs_client.describe_log_groups(limit=1)
        except Exception as e:
            log_with_timestamp(f"{account_type} AWS credentials not configured or insufficient permissions", "yellow")
            return None

        # Start the query
        query_id = start_cloudwatch_query(logs_client, model_id, start_timestamp_ms, expected_count, account_type)
        if not query_id:
            return None

        # Wait for completion
        if not wait_for_query_completion(logs_client, query_id, account_type):
            return None

        # Get results
        return get_cloudwatch_results(logs_client, query_id, account_type)

    except Exception as e:
        log_with_timestamp(f"{account_type} CloudWatch analysis failed: {str(e)}", "yellow")
        return None

def send_bedrock_request(request_id, model_id, question, profile_name, account_type):
    """Send a single request directly to Bedrock using specified AWS profile"""
    try:
        # Create Bedrock client with specific profile
        aws_region = get_aws_region()
        session = boto3.Session(profile_name=profile_name)
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
            f"Request #{request_id:2d} | {account_type:8s} | SUCCESS | {response_time:.2f}s",
            "green"
        )

        return {
            "request_id": request_id,
            "success": True,
            "account_type": account_type,
            "profile": profile_name,
            "model_used": model_id,
            "response_time": response_time,
            "content": content[:50] + "..." if len(content) > 50 else content,
            "error": None
        }

    except Exception as e:
        error_msg = str(e)
        if "ThrottlingException" in error_msg or "ServiceQuota" in error_msg:
            log_with_timestamp(
                f"Request #{request_id:2d} | {account_type:8s} | THROTTLED",
                "yellow"
            )
            error_type = "ThrottlingException"
        else:
            log_with_timestamp(
                f"Request #{request_id:2d} | {account_type:8s} | ERROR | {error_msg[:50]}...",
                "red"
            )
            error_type = "Error"

        return {
            "request_id": request_id,
            "success": False,
            "account_type": account_type,
            "profile": profile_name,
            "model_used": None,
            "response_time": 0,
            "content": None,
            "error": error_type
        }

def run_cross_account_demo(num_requests=20, distribution_strategy="round-robin"):
    """Demonstrate cross-account request distribution"""

    print("=" * 80)
    print("AWS ACCOUNT SHARDING DEMO")
    print("=" * 80)

    # Get AWS profiles and model
    primary_profile, secondary_profile = get_aws_profiles()
    model_id = get_model_id()

    print(f"Sending {num_requests} requests in parallel ({num_requests//2} per account) to demonstrate account sharding.")
    print(f"Cross-Region inference profile: {model_id}")
    print("CRIS will automatically route requests to available regions within each AWS account for optimal performance")
    print()

    # Verify both AWS accounts
    print("AWS Account Configuration:")
    print("-------------|------------|--------------------")
    print("Account Name | Account ID | Profile Name")
    print("-------------|------------|--------------------")
    primary_account = verify_aws_credentials(primary_profile, "ACCOUNT1")
    secondary_account = verify_aws_credentials(secondary_profile, "ACCOUNT2")
    if primary_account and secondary_account:
        masked_primary = f"...{primary_account[-4:]}"
        masked_secondary = f"...{secondary_account[-4:]}"
        print(f"ACCOUNT1     | {masked_primary:>10} | {primary_profile:<18}")
        print(f"ACCOUNT2     | {masked_secondary:>10} | {secondary_profile:<18}")
        print("-------------|------------|--------------------")

    if not primary_account or not secondary_account:
        print("\nError: Failed to verify AWS credentials for both accounts.")
        print("Please ensure both profiles are configured in ~/.aws/credentials")
        print(f"  Primary profile: {primary_profile}")
        print(f"  Secondary profile: {secondary_profile}")
        sys.exit(1)

    if primary_account == secondary_account:
        print("\nWarning: Both profiles point to the same AWS account!")
        print("For true cross-account demonstration, configure different accounts.")

    print()
    log_with_timestamp("Starting cross-account requests...", "blue")

    # Prepare questions for requests
    questions = [
        "Hello from cross-account request!",
        "Test message for account isolation demo",
        "Cross-account demo request",
        "Multi-account test message",
        "Account distribution test"
    ]

    results = []
    results_lock = threading.Lock()
    demo_start_time = time.time()
    demo_start_timestamp_ms = int(demo_start_time * 1000)  # Convert to milliseconds for CloudWatch

    def worker(req_id):
        question = questions[req_id % len(questions)]

        # Determine which account to use based on strategy
        if distribution_strategy == "round-robin":
            # Alternate between accounts
            use_primary = (req_id % 2 == 0)
        elif distribution_strategy == "split":
            # First half to account1, second half to account2
            use_primary = (req_id < num_requests // 2)
        else:  # random
            import random
            use_primary = random.choice([True, False])

        profile = primary_profile if use_primary else secondary_profile
        account_type = "ACCOUNT1" if use_primary else "ACCOUNT2"

        result = send_bedrock_request(req_id + 1, model_id, question, profile, account_type)
        with results_lock:
            results.append(result)
        return result

    # Send all requests in parallel
    with ThreadPoolExecutor(max_workers=min(num_requests, 20)) as executor:
        futures = [executor.submit(worker, i) for i in range(num_requests)]
        # Wait for all to complete
        for future in as_completed(futures):
            pass  # Results already collected in worker

    total_time = time.time() - demo_start_time
    log_with_timestamp(f"Demo completed in {total_time:.1f} seconds", "blue")

    return analyze_cross_account_results(results, total_time, primary_account, secondary_account,
                                       primary_profile, secondary_profile, model_id, demo_start_timestamp_ms)

def analyze_cross_account_results(results, total_time, primary_account, secondary_account,
                                primary_profile, secondary_profile, model_id, demo_start_timestamp_ms):
    """Analyze cross-account results and display distribution"""

    print()
    print("=" * 80)
    print("ACCOUNT SHARDING ANALYSIS")
    print("=" * 80)

    # Calculate stats per account
    primary_results = [r for r in results if r["account_type"] == "ACCOUNT1"]
    secondary_results = [r for r in results if r["account_type"] == "ACCOUNT2"]

    primary_successful = len([r for r in primary_results if r["success"]])
    primary_failed = len([r for r in primary_results if not r["success"]])
    primary_throttled = len([r for r in primary_results if r.get("error") == "ThrottlingException"])

    secondary_successful = len([r for r in secondary_results if r["success"]])
    secondary_failed = len([r for r in secondary_results if not r["success"]])
    secondary_throttled = len([r for r in secondary_results if r.get("error") == "ThrottlingException"])

    total = len(results)
    total_successful = primary_successful + secondary_successful
    success_rate = (total_successful / total) * 100 if total > 0 else 0

    # Calculate average response times
    primary_avg_time = sum(r["response_time"] for r in primary_results if r["success"]) / primary_successful if primary_successful else 0
    secondary_avg_time = sum(r["response_time"] for r in secondary_results if r["success"]) / secondary_successful if secondary_successful else 0

    # Display results in table format
    print()
    print("-" * 70)
    print("Account    | Total | Success | Failed | Success Rate | Avg Time")
    print("-----------|-------|---------|--------|--------------|----------")

    if len(primary_results) > 0:
        primary_success_rate = (primary_successful / len(primary_results)) * 100
        color = "green" if primary_success_rate == 100.0 else "red"
        primary_line = f"ACCOUNT1   | {len(primary_results):>5d} | {primary_successful:>7d} | {primary_failed:>6d} | {primary_success_rate:>11.1f}% | {primary_avg_time:>7.2f}s"
        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "reset": "\033[0m"
        }
        color_code = colors.get(color, "")
        reset_code = colors["reset"]
        print(f"{color_code}{primary_line}{reset_code}")

    if len(secondary_results) > 0:
        secondary_success_rate = (secondary_successful / len(secondary_results)) * 100
        color = "green" if secondary_success_rate == 100.0 else "red"
        secondary_line = f"ACCOUNT2   | {len(secondary_results):>5d} | {secondary_successful:>7d} | {secondary_failed:>6d} | {secondary_success_rate:>11.1f}% | {secondary_avg_time:>7.2f}s"
        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "reset": "\033[0m"
        }
        color_code = colors.get(color, "")
        reset_code = colors["reset"]
        print(f"{color_code}{secondary_line}{reset_code}")

    print("-" * 70)
    color = "green" if success_rate == 100.0 else "red"
    total_line = f"TOTAL      | {total:>5d} | {total_successful:>7d} | {total - total_successful:>6d} | {success_rate:>11.1f}% | {(primary_avg_time + secondary_avg_time) / 2:>7.2f}s"
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    color_code = colors.get(color, "")
    reset_code = colors["reset"]
    print(f"{color_code}{total_line}{reset_code}")


    # CloudWatch Logs Analysis - Real Regional Distribution per Account
    if total_successful > 0:

        # Display query parameters
        log_group_name = get_log_group_name()
        generous_timestamp_ms = demo_start_timestamp_ms - (60 * 1000)
        start_time_readable = datetime.fromtimestamp(generous_timestamp_ms / 1000).strftime('%H:%M:%S')
        end_time_readable = datetime.fromtimestamp(int(time.time())).strftime('%H:%M:%S')

        print()
        print("CloudWatch Query Parameters:")
        print(f"  Log Group:    {log_group_name}")
        print(f"  Model Filter: {model_id}")
        print(f"  Time Range:   {start_time_readable} - {end_time_readable}")
        print()

        log_with_timestamp("Waiting for logs to propagate to CloudWatch...", "blue")

        # Query both accounts in parallel
        all_cloudwatch_results = {}
        max_attempts = 5

        for attempt in range(max_attempts):
            if attempt == 0:
                wait_time = 60  # First attempt: 60s after demo completion
                log_with_timestamp(f"Waiting 60s for logs to propagate... (1/{max_attempts})", "blue")
            else:
                wait_time = 30  # Subsequent attempts: 30s each
                log_with_timestamp(f"Waiting 30s for logs to propagate... ({attempt + 1}/{max_attempts})", "blue")
            time.sleep(wait_time)

            # Query both accounts
            futures = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                if primary_successful > 0:
                    futures.append(executor.submit(
                        query_cloudwatch_distribution_for_account,
                        primary_profile, "ACCOUNT1", model_id, generous_timestamp_ms, primary_successful
                    ))
                if secondary_successful > 0:
                    futures.append(executor.submit(
                        query_cloudwatch_distribution_for_account,
                        secondary_profile, "ACCOUNT2", model_id, generous_timestamp_ms, secondary_successful
                    ))

                # Collect results
                for future in as_completed(futures):
                    result = future.result()
                    if result and result.get('region_distribution'):
                        all_cloudwatch_results[result['account_type']] = result

            # Check if we have complete results
            expected_primary = primary_successful > 0
            expected_secondary = secondary_successful > 0
            got_primary = 'ACCOUNT1' in all_cloudwatch_results and all_cloudwatch_results['ACCOUNT1'].get('total_invocations', 0) == primary_successful
            got_secondary = 'ACCOUNT2' in all_cloudwatch_results and all_cloudwatch_results['ACCOUNT2'].get('total_invocations', 0) == secondary_successful

            # Break if we have all expected results
            if (not expected_primary or got_primary) and (not expected_secondary or got_secondary):
                break
            elif attempt < 4:
                missing = []
                if expected_primary and not got_primary:
                    found = all_cloudwatch_results.get('ACCOUNT1', {}).get('total_invocations', 0)
                    missing.append(f"ACCOUNT1 ({found}/{primary_successful})")
                if expected_secondary and not got_secondary:
                    found = all_cloudwatch_results.get('ACCOUNT2', {}).get('total_invocations', 0)
                    missing.append(f"ACCOUNT2 ({found}/{secondary_successful})")
                log_with_timestamp(f"Incomplete data for: {', '.join(missing)}, will retry...", "yellow")

        # Display CloudWatch results for each account
        if all_cloudwatch_results:

            # Combine all regions from both accounts
            all_regions = set()
            for account_results in all_cloudwatch_results.values():
                all_regions.update(account_results.get('region_distribution', {}).keys())

            for account_type in ['ACCOUNT1', 'ACCOUNT2']:
                if account_type in all_cloudwatch_results:
                    account_results = all_cloudwatch_results[account_type]
                    region_dist = account_results['region_distribution']
                    total_cw_invocations = account_results['total_invocations']

                    print(f"\n{account_type}:")
                    print("-------------|-------------|------------")
                    print("Region       | Invocations | Percentage")
                    print("-------------|-------------|------------")

                    for region in sorted(region_dist.keys()):
                        count = region_dist[region]
                        percentage = (count / total_cw_invocations * 100) if total_cw_invocations > 0 else 0
                        print(f"{region:<12s} | {count:>11d} | {percentage:>9.1f}%")

        else:
            log_with_timestamp("CloudWatch analysis unavailable (check AWS credentials/permissions)", "yellow")

    if primary_account == secondary_account:
        print("\n⚠️  Note: Both profiles use the same AWS account.")
        print("   For true quota isolation, configure different AWS accounts.")

    print("=" * 80)

    return {
        "total_requests": total,
        "successful": total_successful,
        "failed": total - total_successful,
        "success_rate": success_rate,
        "primary_requests": len(primary_results),
        "secondary_requests": len(secondary_results),
        "total_time": total_time
    }

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Account Sharding Demo - Shows multi-account inference distribution with CloudWatch analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_cross_account.py                           # 20 requests, round-robin
  python demo_cross_account.py --requests 30             # 30 requests, round-robin
  python demo_cross_account.py --strategy split          # Split requests between accounts
  python demo_cross_account.py --strategy random         # Random distribution

Distribution Strategies:
  round-robin: Alternate requests between accounts (default)
  split:       First half to primary, second half to secondary
  random:      Random distribution between accounts

Prerequisites:
  • Configure two AWS profiles in ~/.aws/credentials
  • Update config/config.yaml with profile names
  • Enable Bedrock model invocation logging in both accounts
  • Ensure CloudWatch log groups exist in both accounts
        """
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of parallel requests to send (default: 20)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["round-robin", "split", "random"],
        default="round-robin",
        help="Distribution strategy for requests (default: round-robin)"
    )

    args = parser.parse_args()

    if args.requests < 1:
        print("Error: Number of requests must be at least 1")
        sys.exit(1)

    if args.requests > 100:
        print("Error: Maximum 100 requests allowed to avoid overwhelming the system")
        sys.exit(1)

    run_cross_account_demo(args.requests, args.strategy)

if __name__ == "__main__":
    main()