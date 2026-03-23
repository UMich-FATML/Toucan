#!/bin/bash
#SBATCH --job-name=step1.3-onet
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../logs/slurm/step1.3-onet-%j.out
#SBATCH --error=../logs/slurm/step1.3-onet-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

CONDA_SH="${CONDA_SH:-/mnt/weka/home/yuekai.sun/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-toucan}"
INPUT_FILE="${INPUT_FILE:-../data/three_onet_tasks_no_refs/onet_tasks_3_tasks_all_1773890762_kimi-k2.5_results.jsonl}"
OUTPUT_SCHEMA_FILE="${OUTPUT_SCHEMA_FILE:-./prompts/genq_from_onet_tasks_output_schema.json}"
DISABLE_SANITIZE="${DISABLE_SANITIZE:-1}"
DISABLE_PREPARE="${DISABLE_PREPARE:-0}"
ENABLE_TOOL_HINT="${ENABLE_TOOL_HINT:-0}"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Error: conda init script not found: $CONDA_SH" >&2
    exit 1
fi

input_dir=$(dirname "$INPUT_FILE")
if [[ ! -d "$input_dir" ]]; then
    echo "Error: input directory not found: $input_dir" >&2
    exit 1
fi

if [[ ! -f "$OUTPUT_SCHEMA_FILE" ]]; then
    echo "Error: OUTPUT_SCHEMA_FILE not found: $OUTPUT_SCHEMA_FILE" >&2
    exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cmd=(
    python
    -u
    step1.3_process_onet_completion.py
    --input_file "$INPUT_FILE"
    --output_schema_file "$OUTPUT_SCHEMA_FILE"
)

if [[ "$DISABLE_SANITIZE" == "1" ]]; then
    cmd+=(--disable_sanitize)
fi

if [[ "$DISABLE_PREPARE" == "1" ]]; then
    cmd+=(--disable_prepare)
fi

if [[ "$ENABLE_TOOL_HINT" == "1" ]]; then
    cmd+=(--enable_tool_hint)
fi

echo "[step1.3_onet] submit_dir=${SLURM_SUBMIT_DIR:-$PWD}"
echo "[step1.3_onet] input_file=${INPUT_FILE}"
echo "[step1.3_onet] conda_env=${CONDA_ENV}"
echo "[step1.3_onet] command=${cmd[*]}"
"${cmd[@]}"
