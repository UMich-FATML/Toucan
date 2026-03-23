#!/bin/zsh
#SBATCH --job-name=vllm_glm-5
#SBATCH --partition=main
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=14-00:00:00
#SBATCH --output=../logs/slurm/%j_vllm_glm-5.out
#SBATCH --error=../logs/slurm/%j_vllm_glm-5.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p ../logs/slurm

MODEL_PATH=/mnt/weka/home/yuekai.sun/.cache/huggingface/hub/models--zai-org--GLM-5-FP8/snapshots/c0a80ae962efbd9228e8f046525672cda0370579
DOCKER_IMAGE=vllm/vllm-openai:glm5

VLLM_PID=$(bash run_vllm_image.sh "$MODEL_PATH" --image "$DOCKER_IMAGE")
echo "vLLM server ready (PID: $VLLM_PID, mode: docker)"

# Keep the SLURM job alive until the server exits.
# Cannot use `wait` because vLLM was started in a subshell (start_vllm.sh),
# so its PID is not a direct child of this shell.
while kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep 15
done
