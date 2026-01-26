#!/bin/bash

# Configuration
input_file=${1}
model_path=${2:-"openai/gpt-4o"}
virtual_model=${3:-"openai/gpt-4o"}
step="3.21"

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    exit 1
fi

echo "👻 Starting Virtual Agent Generation..."
echo "   - Agent Model: $model_path"
echo "   - Tool Model:  $virtual_model"

# Run Python Script
python completion_openai_agent.py \
    --input_file "${input_file}" \
    --model_path "${model_path}" \
    --engine "openrouter_api" \
    --step "${step}" \
    --agent "openai_agent" \
    --virtual_tools \
    --virtual_tool_model "${virtual_model}" \
    --max_workers 8 \
    --timeout 120 \