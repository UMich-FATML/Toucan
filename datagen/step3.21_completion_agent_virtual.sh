#!/bin/bash

set -euo pipefail

# Configuration
input_file=${1:-}
start_idx=${2:-}
batch_size=${3:-}
model_path=${4:-"Kimi-K2-Thinking"}
base_url=${5:-"http://localhost:8000/v1"}
api_key=${6:-"EMPTY"}
mcp_server_dir=${7:-"../mcp_servers/smithery_mcp_servers_0210"}
step="3.21"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_PID=""

if [ -z "${input_file}" ]; then
    echo "Error: input_file is required."
    exit 1
fi

if [ -z "${start_idx}" ] || [ -z "${batch_size}" ]; then
    echo "Usage: bash step3.21_completion_agent_virtual.sh <input_file> <start_idx> <batch_size> [model_path] [base_url] [api_key] [mcp_server_dir]"
    exit 1
fi

if [ ! -f "${input_file}" ]; then
    echo "Error: Input file does not exist: ${input_file}"
    exit 1
fi

if ! [[ "${start_idx}" =~ ^[0-9]+$ ]]; then
    echo "Error: start_idx must be a non-negative integer."
    exit 1
fi

if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: batch_size must be a positive integer."
    exit 1
fi

compute_expected_output_file() {
    python - "${input_file}" "${start_idx}" "${batch_size}" "${model_path}" "${SCRIPT_DIR}/model_configs.json" <<'PY'
import json
import sys

input_file, start_idx_arg, batch_size_arg, model_path, config_file = sys.argv[1:6]
start_idx = int(start_idx_arg)
batch_size = int(batch_size_arg)

def load_dataset_from_file(filename):
    if filename.endswith(".json"):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    if filename.endswith(".jsonl"):
        data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")

def get_model_short_name(model_path):
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path

def get_model_abbreviation(model_path, cfg_path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            model_configs = json.load(f)
        if model_path in model_configs:
            return model_configs[model_path]["abbreviation"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return get_model_short_name(model_path)

dataset = load_dataset_from_file(input_file)
if not isinstance(dataset, list):
    dataset = [dataset]

total_rows = len(dataset)
if start_idx >= total_rows:
    raise ValueError(
        f"--start_idx ({start_idx}) must be smaller than dataset size ({total_rows})."
    )

end_idx = min(start_idx + batch_size, total_rows)
base_name = input_file[: input_file.rfind(".")]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]
elif base_name.endswith("_prepared"):
    base_name = base_name[:-9]

model_abbreviation = get_model_abbreviation(model_path, config_file)
config_str = f"{model_abbreviation}_high_pfc"
saved_file = f"{base_name}_{config_str}_results_{start_idx}_{end_idx}.jsonl"
print(saved_file)
PY
}

is_all_error_output() {
    local output_file="$1"
    python - "$output_file" <<'PY'
import json
import sys

path = sys.argv[1]
total = 0
error_only = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = item.get("messages") or []
        if not messages:
            continue
        last_message = messages[-1]
        content = last_message.get("content") if isinstance(last_message, dict) else ""
        if isinstance(content, str) and content.startswith("[ERROR:"):
            error_only += 1

if total > 0 and error_only == total:
    print("all_error")
PY
}

resolve_vllm_model_path() {
    case "${model_path}" in
        "Kimi-K2-Thinking")
            echo "/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"
            ;;
        "Kimi-K2.5")
            echo "/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1"
            ;;
        *)
            echo "${model_path}"
            ;;
    esac
}

normalize_base_url() {
    local normalized="$1"
    normalized="${normalized%/}"
    normalized="${normalized%/chat/completions}"
    if [[ "${normalized}" != */v1 ]]; then
        normalized="${normalized}/v1"
    fi
    echo "${normalized}"
}

endpoint_is_healthy() {
    local normalized_url="$1"
    curl -s --max-time 10 "${normalized_url}/models" > /dev/null 2>&1
}

cleanup_vllm() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "🧹 Cleaning up vLLM server (PID: ${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}

ensure_base_url_ready() {
    local normalized_url
    normalized_url="$(normalize_base_url "${base_url}")"

    if endpoint_is_healthy "${normalized_url}"; then
        echo "✅ Found reachable model endpoint at ${normalized_url}"
        base_url="${normalized_url}"
        return
    fi

    if [[ "${normalized_url}" != "http://localhost:8000/v1" && "${normalized_url}" != "http://127.0.0.1:8000/v1" ]]; then
        echo "Error: base_url is unreachable: ${normalized_url}" >&2
        echo "Set a reachable --base_url or use localhost with local vLLM startup." >&2
        exit 1
    fi

    local vllm_model_path
    vllm_model_path="$(resolve_vllm_model_path)"
    echo "⚠️ Endpoint ${normalized_url} is unavailable. Starting local vLLM for ${vllm_model_path}..."

    if command -v enroot >/dev/null 2>&1; then
        VLLM_PID=$(bash "${SCRIPT_DIR}/run_vllm_image.sh" "${vllm_model_path}")
    else
        VLLM_PID=$(bash "${SCRIPT_DIR}/start_vllm.sh" "${vllm_model_path}")
    fi

    if [[ -z "${VLLM_PID}" ]]; then
        echo "Error: failed to start local vLLM server." >&2
        exit 1
    fi

    trap cleanup_vllm EXIT INT TERM

    if ! endpoint_is_healthy "${normalized_url}"; then
        echo "Error: local vLLM did not become healthy at ${normalized_url}" >&2
        exit 1
    fi

    echo "✅ Local vLLM ready at ${normalized_url} (PID: ${VLLM_PID})"
    base_url="${normalized_url}"
}

expected_output_file=$(compute_expected_output_file)
if [ -f "${expected_output_file}" ]; then
    if [[ "$(is_all_error_output "${expected_output_file}")" == "all_error" ]]; then
        echo "⚠️ Existing shard output is all errors. Regenerating: ${expected_output_file}"
    else
        echo "✅ Completed shard already exists. Skipping: ${expected_output_file}"
        exit 0
    fi
fi

ensure_base_url_ready

echo "👻 Starting Virtual Agent Generation..."
echo "   - Input:       $input_file"
echo "   - Range:       [$start_idx, $((start_idx + batch_size)))"
echo "   - Agent Model: $model_path"
echo "   - Base URL:    $base_url"
echo "   - MCP Dir:     $mcp_server_dir"
echo "   - Output:      $expected_output_file"

# Run Python Script
python completion_openai_agent.py \
    --input_file "${input_file}" \
    --start_idx "${start_idx}" \
    --batch_size "${batch_size}" \
    --model_path "${model_path}" \
    --base_url "${base_url}" \
    --api_key "${api_key}" \
    --step "${step}" \
    --agent "openai_agent" \
    --virtual_tools \
    --mcp_server_dir "${mcp_server_dir}" \
    --max_workers 80 \
    --timeout 900
