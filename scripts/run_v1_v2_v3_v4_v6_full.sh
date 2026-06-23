#!/bin/bash
# Train V1, V2, V3, V4, V6 to completion with retry-resume.
# V5 skipped (final_adapter already exists). V7 separate.

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
SUMMARY=$RUNS/v1_v2_v3_v4_v6_summary.log
MAX_ATTEMPTS=50

mkdir -p "$EVAL_DIR"

slog() {
    echo "=== $(date) [v1-v6] $* ===" | tee -a "$SUMMARY"
    "$NOTIFY" grpo-v1v6 "$*" 2>/dev/null || true
}

# JOBS: ver | run_name | train_args (max-steps comes from args)
declare -a JOBS=(
    "v1|full_lora_stratified_2000_v1|--reward-mode stratified --max-steps 2000 --kl-coef 0.04 --grad-accum 1 --temperature 0.8 --save-every 50"
    # V2 SKIPPED — checkpoint trail at runs/full_lora_stratified_v2_2000_v2/ (ckpt 50-700) will be eval'd separately
    # "v2|full_lora_stratified_v2_2000_v2|--reward-mode stratified_v2 --max-steps 2000 --kl-coef 0.10 --grad-accum 1 --temperature 0.8 --save-every 50"
    "v3|full_lora_stratified_n2000_v3|--reward-mode stratified --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7 --save-every 25"
    "v4|full_lora_hybrid_n2000_v4|--reward-mode hybrid --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7 --save-every 25"
    "v6|full_lora_stratified_lc_n2000_v6|--reward-mode stratified --length-control --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7 --save-every 25"
)

slog "START full V1+V2+V3+V4+V6 (V5 skipped, V7 done). max_attempts=$MAX_ATTEMPTS each."

for job in "${JOBS[@]}"; do
    IFS='|' read -r VER RUN_NAME TRAIN_ARGS <<< "$job"
    RUN_DIR=$RUNS/$RUN_NAME
    TRAIN_LOG=$RUN_DIR/training.log
    EVAL_LOG=$RUN_DIR/eval.log
    mkdir -p "$RUN_DIR"

    if [ -d "$RUN_DIR/final_adapter" ] && [ ! -L "$RUN_DIR/final_adapter" -o -e "$RUN_DIR/final_adapter" ]; then
        slog "[$VER] SKIP — final_adapter already exists"
        continue
    fi

    # Wipe Triton cache before each version (safety against driver edge cases)
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

        # hw monitor + notify watcher
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

slog "ALL DONE"

# Final summary
echo "" | tee -a "$SUMMARY"
echo "================================================" | tee -a "$SUMMARY"
echo "Final strict OOD feasibility:" | tee -a "$SUMMARY"
echo "================================================" | tee -a "$SUMMARY"
for job in "${JOBS[@]}"; do
    IFS='|' read -r VER RUN_NAME _ <<< "$job"
    F="$EVAL_DIR/${RUN_NAME}_ood.json"
    if [ -s "$F" ]; then
        python3 -c "
import json
d = json.load(open('$F'))
s = d['summary']
print(f'  $VER  ({\"$RUN_NAME\"[-30:]}):  {s[\"n_feasible\"]}/{s[\"n\"]}  mean_gap={s[\"mean_gap_to_bks\"]*100:.2f}%')
" | tee -a "$SUMMARY"
    fi
done
