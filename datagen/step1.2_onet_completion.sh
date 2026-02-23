#!/bin/bash
#
# Step 1.2 (O*NET): Run completions for O*NET question generation.
#
# Usage: bash step1.2_onet_completion.sh <input_file> <start_idx> <batch_size> [model_name] [step] [output_schema_file]
#
# This script ALWAYS starts a vLLM server via run_vllm_image.sh and cleans it up on exit.
# vLLM is started from a fixed local model path for Kimi-K2-Thinking.
# model_name is passed to structured_completions_endpoint.py as the served model identifier.

input_file=${1}
start_idx=${2}
batch_size=${3}
model_name=${4:-"Kimi-K2-Thinking"}
model_path="/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"
step=${5:-"1.2_onet"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_schema_file="${SCRIPT_DIR}/prompts/genq_from_onet_tasks_output_schema.json"
schema_file=${6:-"${default_schema_file}"}
# Extra args (positional $7+) are forwarded to run_vllm_image.sh
shift $(( $# < 6 ? $# : 6 ))
vllm_extra_args=("$@")

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step1.2_onet_completion.sh <input_file> <start_idx> <batch_size> [model_name] [step] [output_schema_file]"
    exit 1
fi
if [ -z "$start_idx" ] || [ -z "$batch_size" ]; then
    echo "Error: start_idx and batch_size are required."
    echo "Usage: bash step1.2_onet_completion.sh <input_file> <start_idx> <batch_size> [model_name] [step] [output_schema_file]"
    exit 1
fi
if ! [[ "$start_idx" =~ ^[0-9]+$ ]]; then
    echo "Error: start_idx must be a non-negative integer."
    exit 1
fi
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: batch_size must be a positive integer."
    exit 1
fi
if [ ! -f "$schema_file" ]; then
    echo "Error: output schema file not found: $schema_file"
    exit 1
fi

VLLM_PID=""

echo -e "${BLUE}[step1.2_onet] Starting vLLM server via run_vllm_image.sh...${NC}"
VLLM_PID=$(bash "${SCRIPT_DIR}/run_vllm_image.sh" "$model_path" "${vllm_extra_args[@]}")
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
    --start_idx "${start_idx}" \
    --batch_size "${batch_size}" \
    --model_name "${model_name}" \
    --output_schema_file "${schema_file}" \
    --step "${step}"
