#!/bin/bash
#
# Step 4.4: Run LLM judge evaluation for one or more eval dimensions.
#
# Discovers prepared prompt files by convention:
#   {base}_eval_{dimension}_prepared.jsonl
# where {base} is --input_file with .jsonl stripped.
#
# Usage:
#   bash step4.4_eval_completion.sh \
#     --input_file path/to/results.jsonl \
#     --dimensions "all" \
#     --model_path "openai/gpt-oss-120b" \
#     --engine "vllm_api" \
#     --start_vllm "true"
#
# Examples:
#   # Run all 3 dimensions
#   bash step4.4_eval_completion.sh --input_file ../data/results/onet_2tasks_results.jsonl --dimensions all
#
#   # Run only grounding
#   bash step4.4_eval_completion.sh --input_file ../data/results/onet_2tasks_results.jsonl --dimensions grounding
#
#   # Run workflow_completion and grounding together
#   bash step4.4_eval_completion.sh --input_file ../data/results/onet_2tasks_results.jsonl --dimensions workflow_completion,grounding
#

# --- Defaults ---
input_file=""
dimensions="all"
model_path="openai/gpt-oss-120b"
engine="vllm_api"
start_vllm="true"
max_tokens=4096

# --- Color definitions ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)  input_file="$2";  shift 2 ;;
        --dimensions)  dimensions="$2";  shift 2 ;;
        --model_path)  model_path="$2";  shift 2 ;;
        --engine)      engine="$2";      shift 2 ;;
        --start_vllm)  start_vllm="$2";  shift 2 ;;
        --max_tokens)  max_tokens="$2";  shift 2 ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

if [ -z "$input_file" ]; then
    echo -e "${RED}Error: --input_file is required.${NC}"
    echo "Usage: bash step4.4_eval_completion.sh --input_file <path.jsonl> [--dimensions all] [--model_path ...] [--engine ...] [--start_vllm true]"
    exit 1
fi

# Derive base path by stripping .jsonl extension
base_path="${input_file%.jsonl}"

# --- Expand dimensions ---
if [ "$dimensions" == "all" ]; then
    dimensions="tool_call,workflow_completion,grounding,followup_quality"
fi

IFS=',' read -ra DIM_ARRAY <<< "$dimensions"

echo -e "${BLUE}[Step 4.4] Evaluation Completion${NC}"
echo -e "  Input file:  ${input_file}"
echo -e "  Dimensions:  ${dimensions}"
echo -e "  Model:       ${model_path}"
echo -e "  Engine:      ${engine}"
echo ""

# --- vLLM server management (only when engine=vllm_api) ---
if [ "$engine" == "vllm_api" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1

    if [ "$start_vllm" == "true" ]; then
        input_file_basename=$(basename "${input_file%.jsonl}")
        input_file_with_timestamp=$(date +%Y%m%d_%H%M%S)_${input_file_basename}

        cleanup() {
            echo "Cleaning up background processes..."
            jobs -p | xargs -r kill
            exit 0
        }
        trap cleanup EXIT INT TERM

        mkdir -p ../logs/vllm
        log_file="../logs/vllm/${input_file_with_timestamp}_eval.log"

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
                --max-model-len 40960 \
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
                --max-model-len 40960 \
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
            echo -e "${YELLOW}[VLLM API] No server found on port 8000. Set --start_vllm true or start one manually.${NC}"
            exit 1
        fi
    fi
fi

# --- Run each dimension ---
FAILED=0
for dim in "${DIM_ARRAY[@]}"; do
    dim=$(echo "$dim" | xargs)  # trim whitespace
    input_file="${base_path}_eval_${dim}_prepared.jsonl"

    if [ ! -f "$input_file" ]; then
        echo -e "${YELLOW}[Step 4.4] Skipping '${dim}': prepared file not found at ${input_file}${NC}"
        continue
    fi

    echo -e "${BLUE}[Step 4.4] Running evaluation for dimension: ${dim}${NC}"
    echo -e "  Input: ${input_file}"

    python completion_endpoint.py \
        --input_file "${input_file}" \
        --model_path "${model_path}" \
        --engine "${engine}" \
        --step "eval_${dim}" \
        --max_tokens ${max_tokens}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[Step 4.4] Completed: ${dim}${NC}"
    else
        echo -e "${RED}[Step 4.4] Failed: ${dim}${NC}"
        FAILED=$((FAILED+1))
    fi
    echo ""
done

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}[Step 4.4] ${FAILED} dimension(s) failed.${NC}"
    exit 1
else
    echo -e "${GREEN}[Step 4.4] All dimensions completed successfully.${NC}"
fi
