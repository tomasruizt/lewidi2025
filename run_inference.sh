#!/bin/bash

# Function to check if the server is ready
wait_for_server() {
    local port=$1
    local pid=$2
    local max_attempts=60  # 300s total / 5s interval = 60 attempts
    local attempt=1
    local wait_time=5

    echo "[run_inference.sh] Waiting for vLLM server to start on port $port (timeout: 5min)..."
    
    while [ $attempt -le $max_attempts ]; do
        # Check if process is still running
        if ! kill -0 $pid 2>/dev/null; then
            echo "[run_inference.sh] Error: Server process (PID: $pid) died unexpectedly"
            return 1
        fi

        if curl -s "http://localhost:$port/v1/models" > /dev/null; then
            echo "[run_inference.sh] Server is ready!"
            return 0
        fi
        
        echo "[run_inference.sh] Attempt $attempt/$max_attempts: Server not ready yet, waiting ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
    done
    
    echo "[run_inference.sh] Error: Server failed to start after $max_attempts attempts (5min timeout)"
    return 1
}

# Default values
VLLM_PORT="8000"
MODEL_ID=""
REASONING=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm_port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --reasoning)
            REASONING="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --vllm_port <port> --model_id <model> --reasoning <0|1>"
            echo "Example: $0 --vllm_port 8000 --model_id Qwen/Qwen3-8B --reasoning 0"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$VLLM_PORT" ] || [ -z "$MODEL_ID" ] || [ -z "$REASONING" ]; then
    echo "[run_inference.sh] Error: Missing required arguments"
    echo "[run_inference.sh] Usage: $0 --vllm_port <port> --model_id <model> --reasoning <0|1>"
    echo "[run_inference.sh] Example: $0 --vllm_port 8000 --model_id Qwen/Qwen3-8B --reasoning 0"
    exit 1
fi

# Validate reasoning argument
if [ "$REASONING" != "0" ] && [ "$REASONING" != "1" ]; then
    echo "[run_inference.sh] Error: reasoning must be either 0 or 1"
    exit 1
fi

# Start vLLM server with appropriate parameters based on reasoning mode
if [ "$REASONING" = "1" ]; then
    echo "[run_inference.sh] Starting vLLM server in reasoning mode..."
    vllm serve $MODEL_ID \
        --dtype auto \
        --enable-reasoning \
        --reasoning-parser deepseek_r1 \
        --task generate \
        --disable-log-requests \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --port $VLLM_PORT &
else
    echo "[run_inference.sh] Starting vLLM server in non-reasoning mode..."
    vllm serve $MODEL_ID \
        --dtype auto \
        --chat-template ./qwen3_nonthinking.jinja \
        --task generate \
        --disable-log-requests \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --port $VLLM_PORT &
fi
SERVER_PID=$!

# Wait for the server to be ready
if ! wait_for_server $VLLM_PORT $SERVER_PID; then
    echo "[run_inference.sh] Failed to start server, killing process $SERVER_PID (if exists) and exiting..."
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run the inference script with appropriate parameters based on reasoning mode
echo "[run_inference.sh] Starting inference..."
if [ "$REASONING" = "1" ]; then
    echo "[run_inference.sh] Running reasoning mode in vLLM..."
    # Reasoning mode parameters from inference.py docstring
    python inference.py \
        --vllm_port $VLLM_PORT \
        --model_id $MODEL_ID \
        --temperature 0.6 \
        --top_p 0.95 \
        --presence_penalty 1.5 \
        --max_tokens 6000 \
        --remote_call_concurrency 32
else
    echo "[run_inference.sh] Running non-reasoning mode in vLLM..."
    # Non-reasoning mode parameters from inference.py docstring
    python inference.py \
        --vllm_port $VLLM_PORT \
        --model_id $MODEL_ID \
        --temperature 0.7 \
        --top_p 0.8 \
        --presence_penalty 1.5 \
        --max_tokens 6000 \
        --remote_call_concurrency 128
fi
echo "[run_inference.sh] Inference completed"

# Kill the specific vLLM server instance we started
echo "[run_inference.sh] Killing vLLM server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null

sleep 10 