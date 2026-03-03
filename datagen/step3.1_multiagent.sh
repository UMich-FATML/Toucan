#!/bin/bash
#
# Step 3.1m: Multi-agent completions with REAL MCP servers.
#
# A "User" LLM drives the conversation, answering follow-up questions and
# evaluating the Student agent's progress turn-by-turn. Real MCP servers
# are contacted via Smithery.
#
# Usage: bash step3.1_multiagent.sh <input_file> [model_path] [engine] [start_vllm] [user_model]
#
# Examples:
#   bash step3.1_multiagent.sh input.jsonl
#   bash step3.1_multiagent.sh input.jsonl "moonshotai/kimi-k2-thinking" openrouter_api false openai/gpt-4o
#   bash step3.1_multiagent.sh input.jsonl "/path/to/model" vllm_api true
#

input_file=${1}
model_path=${2:-"moonshotai/kimi-k2-thinking"}
engine=${3:-"openrouter_api"}
start_vllm_service=${4:-"false"}
user_model=${5:-"openai/gpt-4o"}

# Hardcoded settings
step="3.1m"
agent="openai_agent"
timeout=900
max_turns=15
max_workers=4
user_max_turns=8
smithery_api_pool="smithery_api_pool.json"

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step3.1_multiagent.sh <input_file> [model_path] [engine] [start_vllm] [user_model]"
    exit 1
fi

# --- vLLM server management (only when engine=vllm_api) ---
if [ "$engine" == "vllm_api" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1

    if [ "$start_vllm_service" == "true" ]; then
        input_file_basename=$(basename "${input_file}")
        input_file_with_timestamp=$(date +%Y%m%d_%H%M%S)_${input_file_basename}

        cleanup() {
            echo "Cleaning up background processes..."
            jobs -p | xargs -r kill
            exit 0
        }
        trap cleanup EXIT INT TERM

        mkdir -p ../logs/vllm
        log_file="../logs/vllm/${input_file_with_timestamp}.log"

        echo -e "${BLUE}[VLLM API] Initializing vllm server...${NC}"
        if [[ "$model_path" == *"Mistral-Small-3"* ]]; then
            export VLLM_ATTENTION_BACKEND=XFORMERS
            vllm serve "$model_path" \
                --tokenizer_mode mistral \
                --config_format mistral \
                --load_format mistral \
                --limit_mm_per_prompt 'image=10' \
                --tensor-parallel-size 4 \
                --port 8000 --host 0.0.0.0 \
                --max-model-len 32768 \
                --gpu-memory-utilization 0.9 > "$log_file" 2>&1 &
        elif [[ "$model_path" == *"Devstral-Small"* ]]; then
            export VLLM_ATTENTION_BACKEND=XFORMERS
            vllm serve "$model_path" \
                --tokenizer_mode mistral \
                --config_format mistral \
                --load_format mistral \
                --tensor-parallel-size 4 \
                --port 8000 --host 0.0.0.0 \
                --max-model-len 40960 \
                --gpu-memory-utilization 0.9 > "$log_file" 2>&1 &
        else
            vllm serve "$model_path" \
                --tensor-parallel-size 4 \
                --port 8000 --host 0.0.0.0 \
                --max-model-len 32768 \
                --gpu-memory-utilization 0.9 > "$log_file" 2>&1 &
        fi
        VLLM_PID=$!
        echo -e "${BLUE}[VLLM API] VLLM server PID: $VLLM_PID${NC}"

        echo -e "${BLUE}[VLLM API] Waiting for server to be ready...${NC}"
        MAX_RETRIES=50
        RETRY_COUNT=0
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if curl -s http://localhost:8000/v1/models > /dev/null; then
                echo -e "${GREEN}[VLLM API] Server is ready!${NC}"
                break
            fi
            echo -e "${BLUE}[VLLM API] Waiting... ($((RETRY_COUNT+1))/$MAX_RETRIES)${NC}"
            sleep 10
            RETRY_COUNT=$((RETRY_COUNT+1))
        done

        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "${RED}[VLLM API] Failed to start server after $MAX_RETRIES attempts. Exiting.${NC}"
            exit 1
        fi
    else
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo -e "${GREEN}[VLLM API] Found existing server on port 8000, reusing it.${NC}"
        else
            echo -e "${YELLOW}[VLLM API] No server found on port 8000. Set start_vllm=true or start one manually.${NC}"
            exit 1
        fi
    fi
fi

echo -e "${BLUE}[Step 3.1m] Multi-Agent Evaluation${NC}"
echo -e "  Input:        ${input_file}"
echo -e "  Model:        ${model_path}"
echo -e "  Engine:       ${engine}"
echo -e "  Agent:        ${agent}"
echo -e "  User Model:   ${user_model}"
echo -e "  User Turns:   ${user_max_turns}"
echo -e "  Timeout:      ${timeout}s"
echo -e "  Max turns:    ${max_turns}"
echo -e "  Workers:      ${max_workers}"

python completion_multiagent.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    --engine "${engine}" \
    --step "${step}" \
    --agent "${agent}" \
    --user_model "${user_model}" \
    --user_max_turns ${user_max_turns} \
    --smithery_api_pool "${smithery_api_pool}" \
    --max_workers ${max_workers} \
    --timeout ${timeout} \
    --max_turns ${max_turns}
