#!/bin/bash
#
# Step 3.21m: Multi-agent virtual-tool completions.
#
# A "User" LLM drives the conversation, answering follow-up questions and
# evaluating the Student agent's progress turn-by-turn. Virtual tools are
# used so no real MCP servers are contacted.
#
# Usage:
#   bash step3.21_multiagent_virtual.sh <input_file> [model_path] [legacy_virtual_model] [engine] [legacy_user_model] [vllm_url] [mcp_server_dir]
#
# Examples:
#   bash step3.21_multiagent_virtual.sh input.jsonl
#   bash step3.21_multiagent_virtual.sh input.jsonl openai/gpt-4o
#   bash step3.21_multiagent_virtual.sh input.jsonl openai/gpt-4o openai/gpt-4o-mini openrouter_api openai/gpt-4o
#

# Configuration
input_file=${1}
model_path=${2:-"openai/gpt-4o"}
virtual_model=${3:-"${model_path}"}
engine=${4:-"openrouter_api"}
user_model=${5:-"${model_path}"}
vllm_url=${6:-"http://localhost:8000/v1"}
mcp_server_dir=${7:-"../mcp_servers/smithery_mcp_servers_0210"}
step="3.21m"
user_max_turns=5

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step3.21_multiagent_virtual.sh <input_file> [model_path] [legacy_virtual_model] [engine] [legacy_user_model]"
    exit 1
fi

echo "👥 Starting Multi-Agent Virtual Generation..."
echo "   - Model (Student/User/VirtualTool): $model_path"
echo "   - Engine:      $engine"
echo "   - User Turns:  $user_max_turns"
echo "   - MCP Dir:     $mcp_server_dir"
if [ -n "${3}" ] && [ "$virtual_model" != "$model_path" ]; then
    echo "   - Legacy virtual_model arg ignored: $virtual_model"
fi
if [ -n "${5}" ] && [ "$user_model" != "$model_path" ]; then
    echo "   - Legacy user_model arg ignored:    $user_model"
fi

# Build engine-specific args
engine_args="--engine ${engine}"
if [ "$engine" = "vllm_api" ]; then
    engine_args="${engine_args} --vllm_api_url ${vllm_url}"
fi

# Run Python Script
python completion_multiagent.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    ${engine_args} \
    --step "${step}" \
    --agent "openai_agent" \
    --virtual_tools \
    --user_max_turns ${user_max_turns} \
    --mcp_server_dir "${mcp_server_dir}" \
    --max_workers 8 \
    --timeout 900
