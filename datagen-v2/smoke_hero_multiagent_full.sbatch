#!/bin/zsh
#SBATCH --job-name=hero-multiagent-3task
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=../logs/slurm/hero-multiagent-3task-%j.out
#SBATCH --error=../logs/slurm/hero-multiagent-3task-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

eval "$(conda shell.zsh hook)"
conda activate toucan

export SMOKE_BASE_URL="http://fs-mbz-gpu-962:8000/v1"
export SMOKE_API_KEY="EMPTY"
export SMOKE_MODEL="moonshotai/kimi-k2.5"
export SMOKE_NUM_TASKS=3
export SMOKE_MAX_PER_OCC=1540

STEP11="../data/hero_run_fseamus3tasks/hero_top9_3tasks_1540max_diversity_1773839500_prepared.jsonl"
WITHHELD_SCHEMA="prompts/genq_from_onet_tasks_withheld_output_schema.json"
MCP_SERVER_DIR="../mcp_servers/smithery_mcp_servers_0210"
MODEL_SUFFIX="${SMOKE_MODEL##*/}"

if [[ ! -f "${STEP11}" ]]; then
    echo "Error: step 1.1 artifact not found: ${STEP11}" >&2
    exit 1
fi

if [[ ! -f "${WITHHELD_SCHEMA}" ]]; then
    echo "Error: withheld schema not found: ${WITHHELD_SCHEMA}" >&2
    exit 1
fi

if [[ ! -f "completion_multiagent.py" ]]; then
    echo "Error: completion_multiagent.py not found in datagen-v2." >&2
    exit 1
fi

if [[ ! -f "prompts/user.md" || ! -f "prompts/student.md" || ! -f "prompts/virtual_toucan.md" ]]; then
    echo "Error: one or more step-3 prompt assets are missing from prompts/." >&2
    exit 1
fi

if [[ ! -d "${MCP_SERVER_DIR}" ]]; then
    echo "Error: MCP server directory not found: ${MCP_SERVER_DIR}" >&2
    exit 1
fi

if [[ ! -f "smithery_api_pool.json" ]]; then
    echo "Error: smithery_api_pool.json not found." >&2
    exit 1
fi

STEP12="${STEP11%_prepared.jsonl}_${MODEL_SUFFIX}_results.jsonl"
STEP12_CHECKPOINT_DIR="${STEP11%_prepared.jsonl}_${MODEL_SUFFIX}_results_checkpoints"
STEP11_ROWS_EXPECTED="$(wc -l < "${STEP11}" | tr -d '[:space:]')"

echo "[config] step11=${STEP11}"
echo "[config] smoke_model=${SMOKE_MODEL}"
echo "[config] smoke_base_url=${SMOKE_BASE_URL}"
echo "[config] step11_expected_rows=${STEP11_ROWS_EXPECTED}"

step_file_row_count() {
    command wc -l < "$1" | command tr -d '[:space:]'
}

step_file_exists_nonempty() {
    local target_file="$1"
    [[ -f "${target_file}" ]] || return 1
    [[ "$(step_file_row_count "${target_file}")" -gt 0 ]]
}

step_file_exists_with_expected_rows() {
    local target_file="$1"
    local expected_rows="$2"
    [[ -f "${target_file}" ]] || return 1
    [[ "$(step_file_row_count "${target_file}")" == "${expected_rows}" ]]
}

echo "[step1.2] target=${STEP12}"
if step_file_exists_with_expected_rows "${STEP12}" "${STEP11_ROWS_EXPECTED}"; then
    echo "[step1.2] canonical results already complete; skipping."
else
    if [[ -d "${STEP12_CHECKPOINT_DIR}" ]]; then
        echo "[step1.2] resuming from checkpoint dir ${STEP12_CHECKPOINT_DIR}"
    fi

    python structured_completions_endpoint.py \
        --input_file "${STEP11}" \
        --model_name "${SMOKE_MODEL}" \
        --base_url "${SMOKE_BASE_URL}" \
        --api_key "${SMOKE_API_KEY}" \
        --output_schema_file "${WITHHELD_SCHEMA}" \
        --step 1.2_hero_multiagent_smoke
fi

if ! step_file_exists_with_expected_rows "${STEP12}" "${STEP11_ROWS_EXPECTED}"; then
    echo "Error: step 1.2 did not produce the expected canonical output: ${STEP12}" >&2
    exit 1
fi

echo "[step1.2] row_count=$(step_file_row_count "${STEP12}")"

STEP13="$(dirname "${STEP12}")/processed/$(basename "${STEP12%_results.jsonl}")_3sanitized.jsonl"
echo "[step1.3] target=${STEP13}"
if step_file_exists_nonempty "${STEP13}"; then
    echo "[step1.3] processed output already complete; skipping."
else
    python step1.3_process_onet_completion.py --input_file "${STEP12}" --disable_sanitize
fi

if ! step_file_exists_nonempty "${STEP13}"; then
    echo "Error: step 1.3 did not produce the expected processed output: ${STEP13}" >&2
    exit 1
fi

STEP13_ROWS_ACTUAL="$(step_file_row_count "${STEP13}")"
echo "[step1.3] row_count=${STEP13_ROWS_ACTUAL}"

python - <<'PY' "${STEP13}"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]
has_withheld = sum(1 for item in items if item.get("withheld_info"))
print(f"[step1.3] Items with withheld_info: {has_withheld}/{len(items)}")
if has_withheld == 0:
    raise SystemExit("No withheld_info found after step 1.3.")
PY

STEP2="${STEP13%_3sanitized.jsonl}_validated_prepared.jsonl"
echo "[step2] target=${STEP2}"
if step_file_exists_nonempty "${STEP2}"; then
    echo "[step2] validated prepared output already complete; skipping."
else
    python step2_validate_and_convert.py --input_file "${STEP13}"
fi

if ! step_file_exists_nonempty "${STEP2}"; then
    echo "Error: step 2 did not produce the expected validated prepared output: ${STEP2}" >&2
    exit 1
fi

STEP2_ROWS_ACTUAL="$(step_file_row_count "${STEP2}")"
echo "[step2] row_count=${STEP2_ROWS_ACTUAL}"

STEP31M="${STEP2%_prepared.jsonl}_${MODEL_SUFFIX}_multiagent_pfc_results.jsonl"
echo "[step3.1m] target=${STEP31M}"
if step_file_exists_with_expected_rows "${STEP31M}" "${STEP2_ROWS_ACTUAL}"; then
    echo "[step3.1m] final multi-agent output already complete; skipping."
else
    python completion_multiagent.py \
        --input_file "${STEP2}" \
        --model_path "${SMOKE_MODEL}" \
        --base_url "${SMOKE_BASE_URL}" \
        --api_key "${SMOKE_API_KEY}" \
        --step 3.1m \
        --agent openai_agent \
        --virtual_tools \
        --user_max_turns 10 \
        --mcp_server_dir "${MCP_SERVER_DIR}" \
        --max_workers 80 \
        --timeout 1200
fi

if ! step_file_exists_with_expected_rows "${STEP31M}" "${STEP2_ROWS_ACTUAL}"; then
    echo "Error: step 3.1m did not produce the expected multi-agent output: ${STEP31M}" >&2
    exit 1
fi

echo "[step3.1m] row_count=$(step_file_row_count "${STEP31M}")"
echo "[step3.1m] error_count=$(rg -c '\\[ERROR:' "${STEP31M}")"

python - <<'PY' "${STEP31M}"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]
for item in items[:3]:
    msgs = item.get("messages", [])
    user_turns = sum(1 for m in msgs if m.get("role") == "user")
    asst_turns = sum(1 for m in msgs if m.get("role") == "assistant")
    fn_turns = sum(1 for m in msgs if m.get("role") == "function")
    pid = item.get("metadata", {}).get("prompt_id", "?")
    print(f"[step3.1m] {pid}: {len(msgs)} msgs (user={user_turns}, asst={asst_turns}, fn={fn_turns})")
PY

echo "[done] step12=${STEP12}"
echo "[done] step13=${STEP13}"
echo "[done] step2=${STEP2}"
echo "[done] step31m=${STEP31M}"
