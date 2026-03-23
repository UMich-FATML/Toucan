#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ../logs/slurm

CONDA_SH="${CONDA_SH:-/mnt/weka/home/yuekai.sun/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-toucan}"
STEP13_INPUT="${STEP13_INPUT:-../data/three_onet_tasks_no_refs/onet_tasks_3_tasks_all_1773890762_kimi-k2.5_results.jsonl}"
STEP13_SANITIZED="${STEP13_SANITIZED:-../data/three_onet_tasks_no_refs/processed/onet_tasks_3_tasks_all_1773890762_kimi-k2.5_3sanitized.jsonl}"
STEP13_PREPARED="${STEP13_PREPARED:-../data/three_onet_tasks_no_refs/processed/onet_tasks_3_tasks_all_1773890762_kimi-k2.5_4prepared.jsonl}"
STEP2_INPUT="${STEP2_INPUT:-$STEP13_SANITIZED}"
STEP2_PREPARED="${STEP2_PREPARED:-../data/three_onet_tasks_no_refs/processed/onet_tasks_3_tasks_all_1773890762_kimi-k2.5_validated_prepared.jsonl}"
STEP2_ANSWER_KEY="${STEP2_ANSWER_KEY:-../data/three_onet_tasks_no_refs/processed/onet_tasks_3_tasks_all_1773890762_kimi-k2.5_answer_key.jsonl}"
STEP2_LOG="${STEP2_LOG:-../logs/slurm/step2_validate_and_convert_1773890762.log}"
MONITOR_LOG="${MONITOR_LOG:-../logs/slurm/step1.3_step2_monitor_1773890762.log}"
PID_FILE="${PID_FILE:-../logs/slurm/step1.3_step2_monitor.pid}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
STEP13_TIME_LIMIT="${STEP13_TIME_LIMIT:-24:00:00}"
STEP13_CPUS="${STEP13_CPUS:-4}"
STEP13_MEM_GB="${STEP13_MEM_GB:-32}"

CURRENT_JOB_ID="${1:-}"
if [[ -z "$CURRENT_JOB_ID" ]]; then
    echo "Usage: $0 <current_step1.3_job_id>" >&2
    exit 1
fi

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp)] $*" | tee -a "$MONITOR_LOG"
}

cleanup() {
    rm -f "$PID_FILE"
}

trap cleanup EXIT

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Watcher already running with PID $existing_pid" >&2
        exit 1
    fi
fi

echo "$$" > "$PID_FILE"

job_state() {
    local job_id="$1"
    local state
    state="$(squeue -h -j "$job_id" -o "%T" | head -n 1 || true)"
    if [[ -n "$state" ]]; then
        echo "$state"
        return
    fi

    state="$(sacct -j "$job_id" --format=State -n -P 2>/dev/null | awk -F'|' 'NR==1 {print $1}' || true)"
    state="${state%% *}"
    if [[ -n "$state" ]]; then
        echo "$state"
    else
        echo "UNKNOWN"
    fi
}

merged_size_gib() {
    if [[ -f "$STEP13_INPUT" ]]; then
        python - <<'PY' "$STEP13_INPUT"
from pathlib import Path
import sys
p = Path(sys.argv[1])
print(f"{p.stat().st_size / 1024 / 1024 / 1024:.3f}")
PY
    else
        echo "missing"
    fi
}

report_progress() {
    local state="$1"
    local size_gib
    size_gib="$(merged_size_gib)"
    log "step1.3 job=${CURRENT_JOB_ID} state=${state} merged_size_gib=${size_gib} sanitized_exists=$([[ -s "$STEP13_SANITIZED" ]] && echo yes || echo no)"
}

prepare_restart() {
    if [[ ! -d "$(dirname "$STEP13_SANITIZED")" ]] || [[ ! -s "$STEP13_SANITIZED" ]]; then
        if [[ -f "$STEP13_INPUT" ]]; then
            log "Removing partial merged input before restart: $STEP13_INPUT"
            rm -f "$STEP13_INPUT"
        fi
    fi
}

choose_restart_resources() {
    local state="$1"
    local next_mem="$STEP13_MEM_GB"
    local next_time="$STEP13_TIME_LIMIT"

    case "$state" in
        OUT_OF_MEMORY)
            if (( next_mem < 64 )); then
                next_mem=64
            elif (( next_mem < 96 )); then
                next_mem=96
            else
                next_mem=128
            fi
            ;;
        TIMEOUT)
            next_time="48:00:00"
            ;;
        FAILED)
            if (( next_mem < 64 )); then
                next_mem=64
            fi
            ;;
    esac

    echo "$next_mem|$next_time"
}

submit_step13() {
    local mem_gb="$1"
    local time_limit="$2"
    local new_job

    new_job="$(
        sbatch \
            --parsable \
            --time="$time_limit" \
            --cpus-per-task="$STEP13_CPUS" \
            --mem="${mem_gb}G" \
            --export=ALL,INPUT_FILE="$STEP13_INPUT",DISABLE_SANITIZE=1,CONDA_SH="$CONDA_SH",CONDA_ENV="$CONDA_ENV" \
            step1.3_process_onet_completion.sbatch
    )"

    STEP13_MEM_GB="$mem_gb"
    STEP13_TIME_LIMIT="$time_limit"
    CURRENT_JOB_ID="$new_job"
    log "Submitted replacement step1.3 job=${CURRENT_JOB_ID} mem=${STEP13_MEM_GB}G time_limit=${STEP13_TIME_LIMIT}"
}

run_step2() {
    if [[ -s "$STEP2_PREPARED" && -s "$STEP2_ANSWER_KEY" ]]; then
        log "step2 outputs already present; skipping step2 run"
        return
    fi

    if [[ ! -s "$STEP2_INPUT" ]]; then
        log "step2 input missing: $STEP2_INPUT"
        exit 1
    fi

    if [[ ! -f "$CONDA_SH" ]]; then
        log "conda init script missing: $CONDA_SH"
        exit 1
    fi

    log "Starting step2 validation and conversion"
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"

    PYTHONUNBUFFERED=1 python -u step2_validate_and_convert.py --input_file "$STEP2_INPUT" \
        2>&1 | tee -a "$STEP2_LOG"
    local step2_status="${PIPESTATUS[0]}"
    if (( step2_status != 0 )); then
        log "step2 failed with exit_code=${step2_status}"
        exit "$step2_status"
    fi

    if [[ ! -s "$STEP2_PREPARED" ]]; then
        log "step2 completed but expected prepared output is missing: $STEP2_PREPARED"
        exit 1
    fi

    log "step2 completed successfully prepared_output=$STEP2_PREPARED answer_key=$STEP2_ANSWER_KEY"
}

restart_count=0
poll_count=0
log "Watcher started for step1.3 job=${CURRENT_JOB_ID}"

while true; do
    state="$(job_state "$CURRENT_JOB_ID")"
    poll_count=$((poll_count + 1))

    case "$state" in
        PENDING|RUNNING|CONFIGURING|COMPLETING)
            report_progress "$state"
            ;;
        COMPLETED)
            log "step1.3 job=${CURRENT_JOB_ID} completed"
            if [[ ! -s "$STEP13_SANITIZED" ]]; then
                log "step1.3 completed without expected sanitized output: $STEP13_SANITIZED"
                prepare_restart
                resources="$(choose_restart_resources FAILED)"
                IFS='|' read -r next_mem next_time <<< "$resources"
                restart_count=$((restart_count + 1))
                if (( restart_count > MAX_RESTARTS )); then
                    log "Exceeded max restarts while recovering missing sanitized output"
                    exit 1
                fi
                submit_step13 "$next_mem" "$next_time"
            else
                break
            fi
            ;;
        FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|CANCELLED)
            log "step1.3 job=${CURRENT_JOB_ID} ended with state=${state}"
            restart_count=$((restart_count + 1))
            if (( restart_count > MAX_RESTARTS )); then
                log "Exceeded max restarts=${MAX_RESTARTS}; stopping watcher"
                exit 1
            fi
            prepare_restart
            resources="$(choose_restart_resources "$state")"
            IFS='|' read -r next_mem next_time <<< "$resources"
            submit_step13 "$next_mem" "$next_time"
            ;;
        *)
            log "step1.3 job=${CURRENT_JOB_ID} has unexpected state=${state}"
            ;;
    esac

    sleep "$POLL_INTERVAL"
done

run_step2
log "Watcher finished successfully"
