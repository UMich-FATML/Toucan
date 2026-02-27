#!/bin/bash

# Configuration
input_file=${1}
model_path=${2:-"openai/gpt-4o"}
virtual_model=${3:-"openai/gpt-4o-mini"}
engine=${4:-"openrouter_api"}
vllm_url=${5:-"http://localhost:8000/v1"}
mcp_server_dir=${6:-"../mcp_servers/smithery_mcp_servers_0210"}
step="3.21"

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    exit 1
fi

echo "👻 Starting Virtual Agent Generation..."
echo "   - Agent Model: $model_path"
echo "   - Tool Model:  $virtual_model"
echo "   - Engine:      $engine"
echo "   - MCP Dir:     $mcp_server_dir"

# Build engine-specific args
engine_args="--engine ${engine}"
if [ "$engine" = "vllm_api" ]; then
    engine_args="${engine_args} --vllm_api_url ${vllm_url}"
fi

# Run Python Script
python completion_openai_agent.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    ${engine_args} \
    --step "${step}" \
    --agent "openai_agent" \
    --virtual_tools \
    --virtual_tool_model "${virtual_model}" \
    --mcp_server_dir "${mcp_server_dir}" \
    --max_workers 8 \
    --timeout 120 \
