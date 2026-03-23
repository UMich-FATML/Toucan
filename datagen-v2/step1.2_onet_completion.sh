#!/bin/bash
#
# Step 1.2 (O*NET): Run completions for O*NET question generation.
#
# Usage: bash step1.2_onet_completion.sh <input_file> <start_idx> <batch_size> [step] [output_schema_file]
#
# This script ALWAYS starts a vLLM server via run_vllm_image.sh and cleans it up on exit.
# vLLM is started from a fixed local model path for Kimi-K2.5.
# model_name is fixed here and passed to structured_completions_endpoint.py as the served model identifier.

input_file=${1}
start_idx=${2}
batch_size=${3}
model_name="moonshotai/kimi-k2.5"
model_path="/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1"
step=${4:-"1.2_onet"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_schema_file="${SCRIPT_DIR}/prompts/genq_from_onet_tasks_output_schema.json"
schema_file=${5:-"${default_schema_file}"}
# Extra args (positional $6+) are forwarded to run_vllm_image.sh
shift $(( $# < 5 ? $# : 5 ))
vllm_extra_args=("$@")

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    echo "Usage: bash step1.2_onet_completion.sh <input_file> <start_idx> <batch_size> [step] [output_schema_file]"
    exit 1
fi
if [ -z "$start_idx" ] || [ -z "$batch_size" ]; then
    echo "Error: start_idx and batch_size are required."
    echo "Usage: bash step1.2_onet_completion.sh <input_file> <start_idx> <batch_size> [step] [output_schema_file]"
    exit 1
fi
if ! [[ "$start_idx" =~ ^[0-9]+$ ]]; then
    echo "Error: start_idx must be a non-negative integer."
    exit 1
fi
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: batch_size must be a positive integer."
    exit 1
fi
if [ ! -f "$schema_file" ]; then
    echo "Error: output schema file not found: $schema_file"
    exit 1
fi

activate_toucan_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo -e "${RED}[step1.2_onet] Error: conda is not available on PATH.${NC}" >&2
        echo "[step1.2_onet] Install/load conda and ensure the 'toucan' env exists." >&2
        exit 1
    fi

    if ! eval "$(conda shell.bash hook)"; then
        echo -e "${RED}[step1.2_onet] Error: failed to initialize conda shell hook.${NC}" >&2
        exit 1
    fi

    if ! conda activate toucan; then
        echo -e "${RED}[step1.2_onet] Error: failed to activate conda env 'toucan'.${NC}" >&2
        echo "[step1.2_onet] Create it and install dependencies before submitting the job." >&2
        exit 1
    fi

    echo -e "${GREEN}[step1.2_onet] Activated conda environment: toucan${NC}"
}

verify_python_deps() {
    local requirements_file="${SCRIPT_DIR}/../requirements.txt"
    if ! python - <<'PY'
import importlib.util
import sys

required = ["openai", "jsonschema", "tqdm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
    then
        echo -e "${RED}[step1.2_onet] Error: dependency preflight failed in env 'toucan'.${NC}" >&2
        if [ -f "$requirements_file" ]; then
            echo "[step1.2_onet] Install deps with: pip install -r $requirements_file" >&2
        else
            echo "[step1.2_onet] Install deps with: pip install openai jsonschema tqdm" >&2
        fi
        exit 1
    fi

    echo -e "${GREEN}[step1.2_onet] Python dependency preflight passed.${NC}"
}

get_total_rows() {
    case "$input_file" in
        *.jsonl)
            wc -l < "$input_file" | tr -d '[:space:]'
            ;;
        *.json)
            python - "$input_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data) if isinstance(data, list) else 1)
PY
            ;;
        *)
            echo "Error: Invalid input file format: $input_file" >&2
            return 1
            ;;
    esac
}

get_model_abbreviation() {
    python - "$1" "$2" <<'PY'
import json
import sys

model_path, config_file = sys.argv[1:3]

try:
    with open(config_file, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    abbreviation = model_configs.get(model_path, {}).get("abbreviation")
    if abbreviation:
        print(abbreviation)
    elif "/" in model_path:
        print(model_path.split("/")[-1])
    else:
        print(model_path)
except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
    if "/" in model_path:
        print(model_path.split("/")[-1])
    else:
        print(model_path)
PY
}

compute_expected_output_file() {
    local total_rows
    local requested_end_idx
    local end_idx
    local base_name
    local model_abbreviation
    local candidate_output_file

    base_name="${input_file%.*}"
    if [[ "$base_name" == *_4prepared ]]; then
        base_name="${base_name%_4prepared}"
    elif [[ "$base_name" == *_prepared ]]; then
        base_name="${base_name%_prepared}"
    fi

    model_abbreviation=$(get_model_abbreviation "$model_name" "${SCRIPT_DIR}/model_configs.json")
    requested_end_idx=$(( start_idx + batch_size ))
    candidate_output_file="${base_name}_${model_abbreviation}_results_${start_idx}_${requested_end_idx}.jsonl"
    if [[ -f "$candidate_output_file" ]]; then
        printf '%s\n' "$candidate_output_file"
        return 0
    fi

    # Only count rows when the optimistic full-batch shard path is missing.
    total_rows=$(get_total_rows)
    if ! [[ "$total_rows" =~ ^[0-9]+$ ]]; then
        echo "Error: Failed to determine total row count from $input_file" >&2
        return 1
    fi
    if (( total_rows == 0 )); then
        echo "Error: Input dataset is empty: $input_file" >&2
        return 1
    fi
    if (( start_idx >= total_rows )); then
        echo "Error: --start_idx ($start_idx) must be smaller than dataset size ($total_rows)." >&2
        return 1
    fi

    end_idx=$requested_end_idx
    if (( end_idx > total_rows )); then
        end_idx=$total_rows
    fi

    printf '%s_%s_results_%s_%s.jsonl\n' "$base_name" "$model_abbreviation" "$start_idx" "$end_idx"
}

# Fail fast on runtime environment issues before starting vLLM.
activate_toucan_env
expected_output_file=$(compute_expected_output_file)
if [ -f "$expected_output_file" ]; then
    echo -e "${GREEN}[step1.2_onet] Final output already exists. Skipping generation: ${expected_output_file}${NC}"
    exit 0
fi
echo "[step1.2_onet] Expected output file: ${expected_output_file}"

verify_python_deps

VLLM_PID=""

echo -e "${BLUE}[step1.2_onet] Starting vLLM server via run_vllm_image.sh...${NC}"
if ! VLLM_PID=$(bash "${SCRIPT_DIR}/run_vllm_image.sh" "$model_path" "${vllm_extra_args[@]}"); then
    echo "Error: Failed to start vLLM server."
    exit 1
fi
if [ -z "$VLLM_PID" ]; then
    echo "Error: Failed to start vLLM server."
    exit 1
fi
echo -e "${GREEN}[step1.2_onet] vLLM server started with PID: ${VLLM_PID}${NC}"

trap '
if [ -n "${VLLM_PID:-}" ]; then
    echo "[step1.2_onet] Cleaning up vLLM server (PID: ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
fi
' EXIT INT TERM

python structured_completions_endpoint.py \
    --input_file "${input_file}" \
    --start_idx "${start_idx}" \
    --batch_size "${batch_size}" \
    --model_name "${model_name}" \
    --output_schema_file "${schema_file}" \
    --step "${step}"
