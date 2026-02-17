#!/bin/zsh
#SBATCH --job-name=vllm-kimi-k2
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=14-00:00:00
#SBATCH --output=../logs/slurm/vllm-kimi-k2-%j.out
#SBATCH --error=../logs/slurm/vllm-kimi-k2-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

# --- Parse arguments ---
MODE="docker"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            MODE="local"; shift ;;
        --docker)
            MODE="docker"; shift ;;
        *)
            echo "Unknown option: $1 (usage: $0 [--local|--docker])" >&2; exit 1 ;;
    esac
done

MODEL_PATH=~/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/

if [[ "$MODE" == "docker" ]]; then
    VLLM_PID=$(bash run_vllm_image.sh "$MODEL_PATH")
else
    VLLM_PID=$(bash start_vllm.sh "$MODEL_PATH")
fi
echo "vLLM server ready (PID: $VLLM_PID, mode: $MODE)"

# Keep the SLURM job alive until the server exits.
# Cannot use `wait` because vLLM was started in a subshell (start_vllm.sh),
# so its PID is not a direct child of this shell.
while kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep 15
done
