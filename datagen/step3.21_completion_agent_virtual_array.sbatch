#!/bin/zsh
#SBATCH --job-name=step3.21-onet-array
#SBATCH --reservation=moe
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --array=0-9%200
#SBATCH --output=../logs/slurm/step3.21-onet-array-%A_%a.out
#SBATCH --error=../logs/slurm/step3.21-onet-array-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

# -----------------------------
# User config (edit these)
# -----------------------------
INPUT_FILE="../data/self_contained_two_onet_task_q_no_refs/processed/onet_tasks_2_tasks_all_1772474997_Kimi-K2-Thinking_validated_prepared.jsonl"
BATCH_SIZE="1000"

MODEL_NAME="Kimi-K2-Thinking"
BASE_URL="http://localhost:8000/v1"
API_KEY="EMPTY"
MCP_SERVER_DIR="../mcp_servers/smithery_mcp_servers_0210"

# Submit with:
#   sbatch step3.21_completion_agent_virtual_array.sbatch

if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: INPUT_FILE is empty. Set INPUT_FILE in this script." >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: INPUT_FILE not found: $INPUT_FILE" >&2
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: BATCH_SIZE must be a positive integer." >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set." >&2
    echo "Submit with: sbatch $0" >&2
    exit 1
fi

total_rows=$(wc -l < "$INPUT_FILE")
total_rows=$(echo "$total_rows" | tr -d '[:space:]')

if ! [[ "$total_rows" =~ ^[0-9]+$ ]]; then
    echo "Error: Failed to read total row count from $INPUT_FILE" >&2
    exit 1
fi

if (( total_rows == 0 )); then
    echo "Error: INPUT_FILE has 0 rows: $INPUT_FILE" >&2
    exit 1
fi

num_tasks=$(( (total_rows + BATCH_SIZE - 1) / BATCH_SIZE ))
task_id=$SLURM_ARRAY_TASK_ID

if ! [[ "$task_id" =~ ^[0-9]+$ ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID must be a non-negative integer." >&2
    exit 1
fi

if (( task_id >= num_tasks )); then
    echo "Task ${task_id} is out of range for num_tasks=${num_tasks}. Exiting." >&2
    exit 0
fi

start_idx=$(( task_id * BATCH_SIZE ))

echo "[step3.21_virtual_array] input_file=${INPUT_FILE}"
echo "[step3.21_virtual_array] total_rows=${total_rows}, batch_size=${BATCH_SIZE}, num_tasks=${num_tasks}"
echo "[step3.21_virtual_array] task_id=${task_id}, start_idx=${start_idx}"
echo "[step3.21_virtual_array] model=${MODEL_NAME}, base_url=${BASE_URL}"

bash step3.21_completion_agent_virtual.sh \
    "$INPUT_FILE" \
    "$start_idx" \
    "$BATCH_SIZE" \
    "$MODEL_NAME" \
    "$BASE_URL" \
    "$API_KEY" \
    "$MCP_SERVER_DIR"
