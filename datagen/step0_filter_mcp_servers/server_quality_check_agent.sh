#!/bin/bash
# Run MCP server quality check using the OpenAI agent framework.
#
# Usage:
#   bash server_quality_check_agent.sh <prepared_jsonl_file> <model_path> [engine]
#
# Arguments:
#   prepared_jsonl_file   Path to the _prepared.jsonl file (required)
#   model_path            Model to use, e.g. qwen/qwen3-30b-a3b-thinking-2507 (required)
#   engine                vllm_api or openrouter_api (default: openrouter_api)
#
# Environment variables:
#   OPENROUTER_API_KEY  Required if engine=openrouter_api
#   VLLM_API_URL        Required if engine=vllm_api (default: http://localhost:8000/v1)
#   VLLM_API_KEY        Required if engine=vllm_api (default: EMPTY)
#   MAX_WORKERS         Number of parallel workers (default: 4)
#   TIMEOUT             Per-server timeout in seconds (default: 180)
#   MAX_TURNS           Max agent turns per server (default: 20)
#
# Example:
#   bash server_quality_check_agent.sh server_quality_check_prepared.jsonl qwen/qwen3-30b-a3b-thinking-2507 openrouter_api

INPUT_FILE="${1:?Usage: bash server_quality_check_agent.sh <prepared_jsonl_file> <model_path> [engine]}"
MODEL_PATH="${2:?model_path argument is required}"
ENGINE="${3:-openrouter_api}"
: "${MAX_WORKERS:=4}"
: "${TIMEOUT:=300}"
: "${MAX_TURNS:=20}"

# Resolve datagen/ directory (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATAGEN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$ENGINE" = "openrouter_api" ]; then
    python "$DATAGEN_DIR/completion_openai_agent.py" \
        --input_file "$INPUT_FILE" \
        --model_path "$MODEL_PATH" \
        --engine openrouter_api \
        --openrouter_api_key "${OPENROUTER_API_KEY:-}" \
        --smithery_api_pool "$DATAGEN_DIR/smithery_api_pool.json" \
        --max_workers "$MAX_WORKERS" \
        --timeout "$TIMEOUT" \
        --max_turns "$MAX_TURNS" \
        --step "server_quality_check"

elif [ "$ENGINE" = "vllm_api" ]; then
    : "${VLLM_API_URL:=http://localhost:8000/v1}"
    : "${VLLM_API_KEY:=EMPTY}"
    python "$DATAGEN_DIR/completion_openai_agent.py" \
        --input_file "$INPUT_FILE" \
        --model_path "$MODEL_PATH" \
        --engine vllm_api \
        --vllm_api_url "$VLLM_API_URL" \
        --vllm_api_key "$VLLM_API_KEY" \
        --smithery_api_pool "$DATAGEN_DIR/smithery_api_pool.json" \
        --max_workers "$MAX_WORKERS" \
        --timeout "$TIMEOUT" \
        --max_turns "$MAX_TURNS" \
        --step "server_quality_check"

else
    echo "Error: ENGINE must be 'openrouter_api' or 'vllm_api', got: $ENGINE"
    exit 1
fi
