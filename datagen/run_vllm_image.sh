#!/bin/bash
#
# Shared vLLM server startup script (Docker/Enroot version).
#
# Identical interface to start_vllm.sh, but runs vLLM inside a container
# via enroot (for Slurm+Pyxis clusters without a local vLLM install).
#
# Usage: bash start_vllm_docker.sh <model_path> [options]
#   --port PORT                  (default: 8000)
#   --max-model-len LEN          (default: 40960)
#   --log-file PATH              (default: auto-generated in ../logs/vllm/)
#   --image IMAGE                (default: vllm/vllm-openai:latest)
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
model_path="${1:?Usage: bash start_vllm_docker.sh <model_path> [options]}"
shift

port=8000
max_model_len=40960
log_file=""
image="vllm/vllm-openai:latest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            port="$2"; shift 2 ;;
        --max-model-len)
            max_model_len="$2"; shift 2 ;;
        --log-file)
            log_file="$2"; shift 2 ;;
        --image)
            image="$2"; shift 2 ;;
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
log_dir="$(cd "$(dirname "$log_file")" && pwd)"
log_file="${log_dir}/$(basename "$log_file")"

# --- Enroot container setup (idempotent) ---
# Derive container name (/ and : -> -)
container_name="${image//\//-}"
container_name="${container_name//:/-}"
# enroot import produces sqsh filenames with + as separator
sqsh_file="${image//\/\//}"   # strip leading //
sqsh_file="${sqsh_file//\//-}"
sqsh_file="${sqsh_file//:/-}"
sqsh_file="${sqsh_file}.sqsh"

if enroot list | grep -qx "$container_name"; then
    echo -e "${BLUE}[start_vllm_docker] Enroot container '$container_name' already exists, skipping import${NC}" >&2
else
    if [[ -f "$sqsh_file" ]]; then
        echo -e "${BLUE}[start_vllm_docker] Reusing existing sqsh file '$sqsh_file'${NC}" >&2
    else
        echo -e "${BLUE}[start_vllm_docker] Importing docker://${image} ...${NC}" >&2
        enroot import --output "$sqsh_file" "docker://${image}"
    fi
    echo -e "${BLUE}[start_vllm_docker] Creating enroot container '$container_name' ...${NC}" >&2
    enroot create --name "$container_name" "$sqsh_file"
    rm -f "$sqsh_file"
fi

# --- HF cache directory ---
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
mkdir -p "$HF_CACHE_DIR"

# --- Model-specific configuration ---
vllm_extra_args=()

if [[ "$model_path" == *"Mistral-Small-3"* ]]; then
    export VLLM_ATTENTION_BACKEND=XFORMERS
    vllm_extra_args+=(--tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 4 --gpu-memory-utilization 0.9)
    echo -e "${BLUE}[start_vllm_docker] Applying XFORMERS backend for Mistral-Small-3${NC}" >&2
elif [[ "$model_path" == *"Devstral-Small"* ]]; then
    export VLLM_ATTENTION_BACKEND=XFORMERS
    vllm_extra_args+=(--tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 4 --gpu-memory-utilization 0.9)
    echo -e "${BLUE}[start_vllm_docker] Applying XFORMERS backend for Devstral-Small${NC}" >&2
elif [[ "$model_path" == *"Kimi-K2-Thinking"* ]]; then
    vllm_extra_args+=(--tensor-parallel-size 8 --decode-context-parallel-size 8 --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code)
    echo -e "${BLUE}[start_vllm_docker] Applying Kimi-K2-Thinking configuration${NC}" >&2
elif [[ "$model_path" == *"Kimi-K2.5"* ]]; then
    vllm_extra_args+=(--tensor-parallel-size 8 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code)
    echo -e "${BLUE}[start_vllm_docker] Applying Kimi-K2.5 configuration${NC}" >&2
fi

# --- Build enroot options ---
enroot_opts=(
    --rw
    --mount "$HF_CACHE_DIR":/root/.cache/huggingface
    --mount "$log_dir":"$log_dir"
)

# Remap local model path for the container.
# If it lives inside $HF_CACHE_DIR, rewrite it to the container mount point
# so that symlinks (e.g. snapshots -> blobs) resolve correctly.
if [[ -d "$model_path" ]]; then
    model_path="$(cd "$model_path" && pwd)"
    hf_cache_real="$(cd "$HF_CACHE_DIR" && pwd)"
    if [[ "$model_path" == "$hf_cache_real"/* ]]; then
        model_path="/root/.cache/huggingface${model_path#"$hf_cache_real"}"
    else
        enroot_opts+=(--mount "$model_path":"$model_path")
    fi
fi

# Pass environment variables (only if non-empty to avoid parser issues)
if [[ -n "${HF_TOKEN:-}" ]]; then
    enroot_opts+=(--env "HF_TOKEN=$HF_TOKEN")
fi
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
    enroot_opts+=(--env "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND")
fi

# --- Start vLLM server inside enroot container ---
echo -e "${BLUE}[start_vllm_docker] Starting vLLM server for ${model_path} (image: ${image})...${NC}" >&2
echo -e "${BLUE}[start_vllm_docker] Log file: ${log_file}${NC}" >&2

enroot start \
    "${enroot_opts[@]}" \
    -- "$container_name" \
    "$model_path" \
        --port "$port" \
        --host 0.0.0.0 \
        --max-model-len "$max_model_len" \
        "${vllm_extra_args[@]}" > "$log_file" 2>&1 &
VLLM_PID=$!

echo -e "${BLUE}[start_vllm_docker] vLLM server started with PID: $VLLM_PID${NC}" >&2

# --- Health check loop ---
echo -e "${BLUE}[start_vllm_docker] Waiting for vLLM server to be ready...${NC}" >&2
MAX_RETRIES=200
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}[start_vllm_docker] vLLM server is ready!${NC}" >&2
        # Print PID to stdout for the caller to capture
        echo "$VLLM_PID"
        exit 0
    else
        echo -e "${BLUE}[start_vllm_docker] Waiting for vLLM server to start... ($((RETRY_COUNT+1))/$MAX_RETRIES)${NC}" >&2
        sleep 15
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

echo -e "${RED}[start_vllm_docker] Failed to start vLLM server after $MAX_RETRIES attempts.${NC}" >&2
kill "$VLLM_PID" 2>/dev/null || true
exit 1
