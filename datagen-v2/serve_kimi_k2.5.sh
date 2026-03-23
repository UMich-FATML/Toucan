#!/bin/zsh
#SBATCH --job-name=vllm_kimi-k2.5
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=14-00:00:00
#SBATCH --output=../logs/slurm/%j_vllm_kimi-k2.5.out
#SBATCH --error=../logs/slurm/%j_vllm_kimi-k2.5.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

MODEL_PATH=/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1

VLLM_PID=$(bash run_vllm_image.sh "$MODEL_PATH")
echo "vLLM server ready (PID: $VLLM_PID, mode: docker)"

# Keep the SLURM job alive until the server exits.
# Cannot use `wait` because vLLM was started by a child launcher script,
# so its PID is not a direct child of this shell.
while kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep 15
done
