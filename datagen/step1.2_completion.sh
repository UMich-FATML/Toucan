#!/bin/bash
#
# Step 1.2: Run completions for question generation.
#
# Usage: bash step1.2_completion.sh <input_file> [model_path] [engine] [step]
#
# If engine is vllm_api, this script checks whether a vLLM server is already
# running. If so, it reuses it. If not, it starts one via start_vllm.sh and
# cleans it up on exit.

# export CUDA_VISIBLE_DEVICES=0,1,2,3

input_file=${1}
model_path=${2:-"/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Instruct-0905/snapshots/ac6c49f04883bd0a0598b790693a72061c676629"}
engine=${3:-"vllm_api"}
step=${4:-"1.2"}
# Extra args (positional $5+) are forwarded to start_vllm.sh
shift $(( $# < 4 ? $# : 4 ))
vllm_extra_args=("$@")

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step1.2_completion.sh <input_file> [model_path] [engine] [step]"
    exit 1
fi

VLLM_PID=""

if [ "$engine" == "vllm_api" ]; then
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}[step1.2] Found existing vLLM server on port 8000, reusing it.${NC}"
    else
        echo -e "${BLUE}[step1.2] No vLLM server found, starting one...${NC}"
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        VLLM_PID=$(bash "${SCRIPT_DIR}/start_vllm.sh" "$model_path" "${vllm_extra_args[@]}")
        if [ $? -ne 0 ] || [ -z "$VLLM_PID" ]; then
            echo "Error: Failed to start vLLM server."
            exit 1
        fi
        echo -e "${GREEN}[step1.2] vLLM server started with PID: ${VLLM_PID}${NC}"

        # Clean up the server we started on exit
        cleanup() {
            echo "[step1.2] Cleaning up vLLM server (PID: $VLLM_PID)..."
            kill "$VLLM_PID" 2>/dev/null || true
        }
        trap cleanup EXIT INT TERM
    fi
fi

python completion_endpoint.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    --engine "${engine}" \
    --step "${step}"
