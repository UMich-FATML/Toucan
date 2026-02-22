#!/bin/bash
#
# Step 1.2 (O*NET): Run completions for O*NET question generation.
#
# Usage: bash step1.2_onet_completion.sh <input_file> [model_path] [step]
#
# This script ALWAYS starts a vLLM server via start_vllm.sh and cleans it up on exit.

input_file=${1}
model_path=${2:-"/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Instruct-0905/snapshots/ac6c49f04883bd0a0598b790693a72061c676629"}
step=${3:-"1.2_onet"}
# Extra args (positional $4+) are forwarded to start_vllm.sh
shift $(( $# < 3 ? $# : 3 ))
vllm_extra_args=("$@")

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step1.2_onet_completion.sh <input_file> [model_path] [step]"
    exit 1
fi

VLLM_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}[step1.2_onet] Starting vLLM server via start_vllm.sh...${NC}"
VLLM_PID=$(bash "${SCRIPT_DIR}/start_vllm.sh" "$model_path" "${vllm_extra_args[@]}")
if [ $? -ne 0 ] || [ -z "$VLLM_PID" ]; then
    echo "Error: Failed to start vLLM server."
    exit 1
fi
echo -e "${GREEN}[step1.2_onet] vLLM server started with PID: ${VLLM_PID}${NC}"

cleanup() {
    echo "[step1.2_onet] Cleaning up vLLM server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python structured_completions_endpoint.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    --engine "vllm_api" \
    --step "${step}"
