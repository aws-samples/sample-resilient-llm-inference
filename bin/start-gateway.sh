#!/bin/bash

# Read configuration from config.yaml using uv's Python environment with validation
CONFIG=$(uv run python -c "
import yaml
import sys
import re

try:
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        aws_config = config.get('aws', {})
        port = config.get('litellm', {}).get('port', 4000)
        profile = aws_config.get('profile_name', 'genai')
        region = aws_config.get('region_name', 'us-east-1')

        # Validate port
        if not isinstance(port, int) or port < 1024 or port > 65535:
            port = 4000

        # Validate profile name (basic validation)
        if not isinstance(profile, str) or not profile.strip() or not re.match(r'^[a-zA-Z0-9_-]+$', profile):
            profile = 'genai'

        # Validate region name (basic AWS region format)
        if not isinstance(region, str) or not re.match(r'^[a-z]{2}(-[a-z]+)*-[0-9]+$', region):
            region = 'us-east-1'

        print(f'{port} {profile} {region}')
except Exception as e:
    print('4000 genai us-east-1', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

# Check if config parsing was successful
if [ $? -ne 0 ] || [ -z "$CONFIG" ]; then
    echo "Error: Failed to parse configuration, using defaults"
    CONFIG="4000 genai us-east-1"
fi

# Parse the output
read PORT AWS_PROFILE AWS_REGION <<< "$CONFIG"

# Set AWS environment variables
export AWS_PROFILE="$AWS_PROFILE"
export AWS_DEFAULT_REGION="$AWS_REGION"

# Show AWS configuration being used
echo "================================================================================"
echo "AWS Configuration: Profile='$AWS_PROFILE', Region='$AWS_REGION'"
echo "================================================================================"

# Start the gateway with single worker for cleaner demo output
uv run litellm --config ./config/config.yaml --port $PORT --num_workers 1