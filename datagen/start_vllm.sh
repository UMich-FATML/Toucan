#!/bin/bash
#
# Shared vLLM server startup script.
#
# Usage: bash start_vllm.sh <model_path> [options]
#   --port PORT                  (default: 8000)
#   --max-model-len LEN          (default: 32768)
#   --tensor-parallel-size N     (default: 4)
#   --gpu-memory-utilization F   (default: 0.9)
#   --enable-tool-call           Add --tool-call-parser and --enable-auto-tool-choice for Mistral/Devstral
#   --log-file PATH              (default: auto-generated in ../logs/vllm/)
#
# On success: prints the vLLM server PID to stdout and exits 0.
# On failure: exits with code 1.
# The server keeps running independently — the caller decides whether to kill it.

set -euo pipefail

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# --- Parse arguments ---
model_path="${1:?Usage: bash start_vllm.sh <model_path> [options]}"
shift

port=8000
max_model_len=40960
log_file=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            port="$2"; shift 2 ;;
        --max-model-len)
            max_model_len="$2"; shift 2 ;;
        --log-file)
            log_file="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# --- Log file setup ---
if [ -z "$log_file" ]; then
    mkdir -p ../logs/vllm
    log_file="../logs/vllm/$(date +%Y%m%d_%H%M%S)_vllm_$(basename "$model_path").log"
fi
mkdir -p "$(dirname "$log_file")"

# --- CUDA forward compatibility (needed when vLLM is built with a newer CUDA than the driver supports) ---
if [[ -d "${CONDA_PREFIX:-}/cuda-compat" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo -e "${BLUE}[start_vllm] Using cuda-compat from ${CONDA_PREFIX}/cuda-compat${NC}" >&2
fi

# --- Model-specific configuration ---
vllm_extra_args=()

if [[ "$model_path" == *"Kimi-K2-Thinking"* ]]; then
    vllm_extra_args+=(--tensor-parallel-size 8 --decode-context-parallel-size 8 --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code)
    echo -e "${BLUE}[start_vllm] Applying Kimi-K2-Thinking configuration${NC}" >&2
elif [[ "$model_path" == *"Kimi-K2.5"* ]]; then
    vllm_extra_args+=(--tensor-parallel-size 8 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code)
    echo -e "${BLUE}[start_vllm] Applying Kimi-K2.5 configuration${NC}" >&2
fi

# --- Activate conda environment ---
eval "$(conda shell.bash hook)"
conda activate vllm

# --- Start vLLM server ---
echo -e "${BLUE}[start_vllm] Starting vLLM server for ${model_path}...${NC}" >&2
echo -e "${BLUE}[start_vllm] Log file: ${log_file}${NC}" >&2

vllm serve "$model_path" \
    --port "$port" \
    --host 0.0.0.0 \
    --max-model-len "$max_model_len" \
    "${vllm_extra_args[@]}" > "$log_file" 2>&1 &
VLLM_PID=$!

echo -e "${BLUE}[start_vllm] vLLM server started with PID: $VLLM_PID${NC}" >&2

# --- Health check loop ---
echo -e "${BLUE}[start_vllm] Waiting for vLLM server to be ready...${NC}" >&2
MAX_RETRIES=200
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}[start_vllm] vLLM server is ready!${NC}" >&2
        # Print PID to stdout for the caller to capture
        echo "$VLLM_PID"
        exit 0
    else
        echo -e "${BLUE}[start_vllm] Waiting for vLLM server to start... ($((RETRY_COUNT+1))/$MAX_RETRIES)${NC}" >&2
        sleep 15
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

echo -e "${RED}[start_vllm] Failed to start vLLM server after $MAX_RETRIES attempts.${NC}" >&2
kill "$VLLM_PID" 2>/dev/null || true
exit 1
