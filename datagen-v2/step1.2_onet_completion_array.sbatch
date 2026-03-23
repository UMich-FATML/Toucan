#!/bin/zsh
#SBATCH --job-name=step1.2-onet-array
#SBATCH --account=k2p
#SBATCH --reservation=moe
#SBATCH --qos=k2p
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --array=0-283%284
#SBATCH --output=../logs/slurm/step1.2-onet-array-%A_%a.out
#SBATCH --error=../logs/slurm/step1.2-onet-array-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

# -----------------------------
# User config (edit these)
# -----------------------------
INPUT_FILE="../data/three_onet_tasks_no_refs/onet_tasks_3_tasks_all_1773890762_prepared.jsonl"
# INPUT_FILE="../data/three_onet_tasks_self_contained_no_refs/onet_tasks_3_tasks_all_1773588590_prepared.jsonl"
BATCH_SIZE="10000"

STEP_TAG="1.2_onet"
OUTPUT_SCHEMA_FILE="./prompts/genq_from_onet_tasks_output_schema.json"

# Forwarded as positional args ($6+) to step1.2_onet_completion.sh.
# Example: VLLM_EXTRA_ARGS=(--port 8001 --max-model-len 32768)
VLLM_EXTRA_ARGS=()

# Submit with:
#   sbatch step1.2_onet_completion_array.sbatch

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

if [[ ! -f "$OUTPUT_SCHEMA_FILE" ]]; then
    echo "Error: OUTPUT_SCHEMA_FILE not found: $OUTPUT_SCHEMA_FILE" >&2
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

echo "[step1.2_onet_array] input_file=${INPUT_FILE}"
echo "[step1.2_onet_array] total_rows=${total_rows}, batch_size=${BATCH_SIZE}, num_tasks=${num_tasks}"
echo "[step1.2_onet_array] task_id=${task_id}, start_idx=${start_idx}"

bash step1.2_onet_completion.sh \
    "$INPUT_FILE" \
    "$start_idx" \
    "$BATCH_SIZE" \
    "$STEP_TAG" \
    "$OUTPUT_SCHEMA_FILE" \
    "${VLLM_EXTRA_ARGS[@]}"
