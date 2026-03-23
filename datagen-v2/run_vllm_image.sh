#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'
PREFIX='[start_vllm_docker]'

log() { local color="$1"; shift; echo -e "${color}${PREFIX} $*${NC}" >&2; }
option_value() { [[ -n "${2-}" ]] || { echo "Missing value for $1" >&2; exit 1; }; printf '%s\n' "$2"; }
slugify() { local value="$1"; value="${value//\//-}"; value="${value//:/-}"; printf '%s\n' "$value"; }

default_log_file() {
    local job_name_slug
    job_name_slug="$(printf '%s\n' "${SLURM_JOB_NAME:-noname}" | sed 's/[^A-Za-z0-9._-]/_/g')"
    printf '../logs/vllm/%s_%s.log\n' "${SLURM_JOB_ID:-nojob}" "$job_name_slug"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$(option_value "$1" "${2-}")"; shift 2 ;;
            --log-file) log_file="$(option_value "$1" "${2-}")"; shift 2 ;;
            --image) image="$(option_value "$1" "${2-}")"; shift 2 ;;
            *) echo "Unknown option: $1" >&2; exit 1 ;;
        esac
    done
}

prepare_log_file() {
    local dir
    log_file="${log_file:-$(default_log_file)}"
    dir="$(dirname "$log_file")"
    mkdir -p "$dir"
    log_dir="$(cd "$dir" && pwd)"
    log_file="${log_dir}/$(basename "$log_file")"
}

validate_model_path() {
    if [[ "$model_path" == /* ]] && [[ ! -e "$model_path" ]]; then
        log "$RED" "Model path does not exist: ${model_path}"
        return 1
    fi
}

ensure_container() {
    local tmp_dir sqsh_file
    container_name="$(slugify "$image")"
    tmp_dir="${TMPDIR:-/tmp}"
    mkdir -p "$tmp_dir"
    sqsh_file="${tmp_dir}/$(slugify "${image#//}")-${SLURM_JOB_ID:-$$}.sqsh"

    if enroot list | grep -qx "$container_name"; then
        log "$BLUE" "Enroot container '$container_name' already exists, skipping import"
        return
    fi
    if [[ -f "$sqsh_file" ]]; then
        log "$BLUE" "Reusing existing sqsh file '$sqsh_file'"
    else
        log "$BLUE" "Importing docker://${image} ..."
        enroot import --output "$sqsh_file" "docker://${image}"
    fi
    log "$BLUE" "Creating enroot container '$container_name' ..."
    enroot create --name "$container_name" "$sqsh_file"
    rm -f "$sqsh_file"
}

build_model_args() {
    vllm_extra_args=()
    case "$model_path" in
        *Kimi-K2.5*)
            vllm_extra_args+=(--tensor-parallel-size 8 --mm-encoder-tp-mode data --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --enable-auto-tool-choice --trust-remote-code --served_model_name moonshotai/kimi-k2.5)
            log "$BLUE" "Applying Kimi-K2.5 configuration"
            ;;
        *GLM-5-FP8*)
            vllm_extra_args+=(--tensor-parallel-size 8 --gpu-memory-utilization 0.9 --speculative-config.method mtp --speculative-config.num_speculative_tokens 1 --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --served-model-name z-ai/glm-5)
            log "$BLUE" "Applying GLM-5-FP8 configuration"
            ;;
    esac
}

prepare_model_mounts() {
    local hf_cache_real
    HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
    mkdir -p "$HF_CACHE_DIR"
    enroot_opts=(--rw --mount "$HF_CACHE_DIR":/root/.cache/huggingface --mount "$log_dir":"$log_dir")

    if [[ -d "$model_path" ]]; then
        model_path="$(cd "$model_path" && pwd)"
        hf_cache_real="$(cd "$HF_CACHE_DIR" && pwd)"
        if [[ "$model_path" == "$hf_cache_real"/* ]]; then
            model_path="/root/.cache/huggingface${model_path#"$hf_cache_real"}"
        else
            enroot_opts+=(--mount "$model_path":"$model_path")
        fi
    fi

    if [[ -n "${HF_TOKEN:-}" ]]; then
        enroot_opts+=(--env "HF_TOKEN=$HF_TOKEN")
    fi
    if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
        enroot_opts+=(--env "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND")
    fi
}

wait_until_ready() {
    local retry_count=0 max_retries=200
    log "$BLUE" "Waiting for vLLM server to be ready..."
    while (( retry_count < max_retries )); do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log "$GREEN" "vLLM server is ready!"
            echo "$VLLM_PID"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log "$RED" "vLLM process exited before becoming ready."
            [[ -f "$log_file" ]] && { log "$RED" "Last 40 log lines from ${log_file}:"; tail -n 40 "$log_file" >&2 || true; }
            return 1
        fi
        retry_count=$((retry_count + 1))
        log "$BLUE" "Waiting for vLLM server to start... (${retry_count}/${max_retries})"
        sleep 15
    done
    log "$RED" "Failed to start vLLM server after $max_retries attempts."
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

model_path="${1:?Usage: bash run_vllm_image.sh <model_path> [options]}"
shift
port=8000
log_file=""
image="vllm/vllm-openai:latest"

parse_args "$@"
prepare_log_file
validate_model_path
ensure_container
build_model_args
prepare_model_mounts

log "$BLUE" "Starting vLLM server for ${model_path} (image: ${image})..."
log "$BLUE" "Log file: ${log_file}"
enroot start \
    "${enroot_opts[@]}" \
    -- "$container_name" \
    "$model_path" \
    --port "$port" \
    --host 0.0.0.0 \
    "${vllm_extra_args[@]}" > "$log_file" 2>&1 &
VLLM_PID=$!
log "$BLUE" "vLLM server started with PID: $VLLM_PID"
wait_until_ready
