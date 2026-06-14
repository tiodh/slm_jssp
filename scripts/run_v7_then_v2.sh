#!/bin/bash
# Chain: resume V7 (350->500) then attempt V2 (700->2000).
# V7 is healthy and just needs ~150 more steps.
# V2 is known-collapsed at step 700+; this is a "coba lanjut" attempt — if it
# stays at parseable=0/4 for 50 consecutive steps, abandon.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
SFT=/home/tio/Documents/Starjob/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800
RUNS=/home/tio/Documents/Starjob/grpo_jssp/runs
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
SUMMARY=$RUNS/v7_then_v2_summary.log
MAX_ATTEMPTS=50

mkdir -p "$EVAL_DIR"

slog() {
    echo "=== $(date) [v7+v2] $* ===" | tee -a "$SUMMARY"
    "$NOTIFY" grpo-chain "$*" 2>/dev/null || true
}

declare -a JOBS=(
    "v7|full_lora_hybrid_lc_over_n2000_v7|--reward-mode hybrid_v7 --length-control --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7 --save-every 25"
    "v2|full_lora_stratified_v2_2000_v2|--reward-mode stratified_v2 --max-steps 2000 --kl-coef 0.10 --grad-accum 1 --temperature 0.8 --save-every 50"
)

slog "START V7-resume + V2-retry chain. max_attempts=$MAX_ATTEMPTS each."

for job in "${JOBS[@]}"; do
    IFS='|' read -r VER RUN_NAME TRAIN_ARGS <<< "$job"
    RUN_DIR=$RUNS/$RUN_NAME
    TRAIN_LOG=$RUN_DIR/training.log
    EVAL_LOG=$RUN_DIR/eval.log
    mkdir -p "$RUN_DIR"

    if [ -d "$RUN_DIR/final_adapter" ] && [ -e "$RUN_DIR/final_adapter/adapter_model.safetensors" ]; then
        slog "[$VER] SKIP — final_adapter already exists"
        continue
    fi

    rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null

    slog "[$VER] start train (up to $MAX_ATTEMPTS attempts, auto-resume from latest checkpoint)"

    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        if [ -d "$RUN_DIR/final_adapter" ]; then
            slog "[$VER] final_adapter saved, done"
            break
        fi

        LATEST_CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [[ -n "$LATEST_CKPT" ]]; then
            RESUME_ARG="--resume-from $LATEST_CKPT"
            slog "[$VER] attempt $attempt/$MAX_ATTEMPTS resume from $(basename $LATEST_CKPT)"
        else
            RESUME_ARG=""
            slog "[$VER] attempt $attempt/$MAX_ATTEMPTS fresh"
        fi

        bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$RUN_DIR/hw_monitor.log" &
        HW_PID=$!
        NOTIFY_EVERY=10 bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$TRAIN_LOG" "grpo-$VER" &
        NW_PID=$!

        python -u -m grpo_jssp.run train \
            --sft-checkpoint "$SFT" \
            $TRAIN_ARGS \
            --run-name "$RUN_NAME" \
            $RESUME_ARG \
            >> "$TRAIN_LOG" 2>&1
        EC=$?

        kill "$HW_PID" "$NW_PID" 2>/dev/null
        wait 2>/dev/null

        if [[ $EC -eq 0 ]]; then
            slog "[$VER] attempt $attempt OK exit=0"
            break
        fi

        slog "[$VER] attempt $attempt CRASHED exit=$EC"
        [[ $attempt -lt $MAX_ATTEMPTS ]] && { slog "[$VER] cooldown 60s..."; sleep 60; }
    done

    if [ -d "$RUN_DIR/final_adapter" ]; then
        slog "[$VER] training done — starting eval"
        python -u -m grpo_jssp._run_eval \
            --adapter "$RUN_DIR/final_adapter" \
            --out-prefix "$EVAL_DIR/$RUN_NAME" \
            >> "$EVAL_LOG" 2>&1
        EVAL_EC=$?
        if [ $EVAL_EC -eq 0 ] && [ -f "$EVAL_DIR/${RUN_NAME}_ood.json" ]; then
            FEAS=$(python3 -c "import json; d=json.load(open('$EVAL_DIR/${RUN_NAME}_ood.json')); print(d['summary']['n_feasible'])" 2>/dev/null || echo "?")
            slog "[$VER] EVAL OOD: ${FEAS}/18 strict feasible"
        else
            slog "[$VER] EVAL FAILED exit=$EVAL_EC"
        fi
    else
        slog "[$VER] TRAIN FAILED after $MAX_ATTEMPTS attempts — no final_adapter"
    fi
done

slog "CHAIN DONE"
