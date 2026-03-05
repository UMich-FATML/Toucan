#!/bin/bash
#
# Step 3.1: Run agent-based completions for evaluation with REAL MCP servers.
#
# Usage: bash step3.1_completion_agent.sh <input_file> [model_path] [engine] [start_vllm]
#
# Examples:
#   bash step3.1_completion_agent.sh input.jsonl                                          # OpenRouter + kimi-k2
#   bash step3.1_completion_agent.sh input.jsonl "moonshotai/kimi-k2-thinking" openrouter_api
#   bash step3.1_completion_agent.sh input.jsonl "/path/to/model" vllm_api true           # vLLM with auto-start
#   bash step3.1_completion_agent.sh input.jsonl "/path/to/model" vllm_api false          # vLLM, server already running
#

input_file=${1}
model_path=${2:-"moonshotai/kimi-k2-thinking"}
engine=${3:-"openrouter_api"}
start_vllm_service=${4:-"false"}

# Hardcoded settings
step="3.1"
agent="openai_agent"
timeout=900
max_turns=15
max_workers=4
smithery_api_pool="smithery_api_pool.json"

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step3.1_completion_agent.sh <input_file> [model_path] [engine] [start_vllm]"
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

if [ "$engine" == "vllm_api" ]; then
    base_url=${BASE_URL:-"http://localhost:8000/v1"}
    api_key=${API_KEY:-"EMPTY"}
elif [ "$engine" == "openrouter_api" ]; then
    base_url=${OPENROUTER_URL:-"https://openrouter.ai/api/v1"}
    api_key=${OPENROUTER_API_KEY:-"EMPTY"}
else
    echo -e "${RED}Error: Unsupported engine '${engine}'. Use 'vllm_api' or 'openrouter_api'.${NC}"
    exit 1
fi

echo -e "${BLUE}[Step 3.1] Agent Evaluation${NC}"
echo -e "  Input:      ${input_file}"
echo -e "  Model:      ${model_path}"
echo -e "  Engine:     ${engine}"
echo -e "  Endpoint:   ${base_url}"
echo -e "  Agent:      ${agent}"
echo -e "  Timeout:    ${timeout}s"
echo -e "  Max turns:  ${max_turns}"
echo -e "  Workers:    ${max_workers}"

python completion_openai_agent.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    --base_url "${base_url}" \
    --api_key "${api_key}" \
    --step "${step}" \
    --agent "${agent}" \
    --smithery_api_pool "${smithery_api_pool}" \
    --max_workers ${max_workers} \
    --timeout ${timeout} \
    --max_turns ${max_turns}
