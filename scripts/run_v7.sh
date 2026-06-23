#!/bin/bash
# V7 training: hybrid_v7 reward (V5 + r_o over-emit) + length-control.
# Retry-with-resume loop: each crash resumes from latest checkpoint.
# Slack notify every 50 steps via NOTIFY_EVERY=10 (logging_steps=5 -> 50).

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
SFT=/home/tio/Documents/Starjob/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800
RUNS=/home/tio/Documents/Starjob/grpo_jssp/runs
RUN_NAME=full_lora_hybrid_lc_over_n2000_v7
RUN_DIR=$RUNS/$RUN_NAME
TRAIN_LOG=$RUN_DIR/training.log
MAX_ATTEMPTS=5

mkdir -p "$RUN_DIR"

slog() {
    echo "=== $(date) [v7] $* ==="
    "$NOTIFY" grpo-v7 "$*" 2>/dev/null || true
}

slog "V7 launcher start (max ${MAX_ATTEMPTS} attempts, retry-with-resume)"

# Wipe Triton cache before launch (driver unchanged but good hygiene)
slog "wiping Triton/Inductor caches"
rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    # Skip if already done
    if [[ -d "$RUN_DIR/final_adapter" ]]; then
        slog "final_adapter exists, training complete"
        break
    fi

    # Find latest checkpoint for resume
    LATEST_CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [[ -n "$LATEST_CKPT" ]]; then
        RESUME_ARG="--resume-from $LATEST_CKPT"
        slog "attempt ${attempt}/${MAX_ATTEMPTS} resume from $(basename $LATEST_CKPT)"
    else
        RESUME_ARG=""
        slog "attempt ${attempt}/${MAX_ATTEMPTS} fresh start"
    fi

    # Start hw monitor + step-rate-limited notify watcher (every 50 steps)
    bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$RUN_DIR/hw_monitor.log" &
    HW_PID=$!
    NOTIFY_EVERY=10 bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$TRAIN_LOG" "grpo-v7" &
    NW_PID=$!

    python -u -m grpo_jssp.run train \
        --sft-checkpoint "$SFT" \
        --reward-mode hybrid_v7 --length-control \
        --max-records 2000 --max-steps 500 \
        --kl-coef 0.05 --grad-accum 4 --temperature 0.7 \
        --save-every 25 \
        --run-name "$RUN_NAME" \
        $RESUME_ARG \
        >> "$TRAIN_LOG" 2>&1
    EC=$?

    # Stop watchers
    kill "$HW_PID" "$NW_PID" 2>/dev/null
    wait 2>/dev/null

    if [[ $EC -eq 0 ]]; then
        slog "attempt ${attempt} OK exit=0"
        break
    fi

    slog "attempt ${attempt} CRASHED exit=$EC"
    if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
        slog "cooldown 60s then resume from latest checkpoint"
        sleep 60
    fi
done

if [[ -d "$RUN_DIR/final_adapter" ]]; then
    slog "V7 TRAINING DONE — final_adapter saved"

    # Eval V7 (SM + OOD) using existing eval runner
    slog "starting V7 eval"
    python -u -m grpo_jssp._run_eval \
        --adapter "$RUN_DIR/final_adapter" \
        --out-prefix "$RUNS/../eval_results/$RUN_NAME" \
        >> "$RUN_DIR/eval.log" 2>&1
    EVAL_EC=$?
    slog "V7 EVAL done exit=$EVAL_EC"
else
    slog "V7 FAILED after ${MAX_ATTEMPTS} attempts — no final_adapter"
    exit 1
fi
