#!/bin/bash
# Eval all 10 V4-LoRA checkpoints on OOD + SM for fair comparison vs V4-rsLoRA.
# Output: eval_results/full_lora_hybrid_n2000_v4_ckpt{N}_{ood,sm}.json

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CC=/usr/bin/clang-18
export CXX=/usr/bin/clang++-18
export TRITON_CC=/usr/bin/clang-18
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

NOTIFY=~/.local/bin/notify
RUN_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs/full_lora_hybrid_n2000_v4
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
LOG_DIR=$RUN_DIR/eval_logs
SUMMARY=$RUN_DIR/eval_all_ckpts_summary.log

mkdir -p "$LOG_DIR"

slog() {
    echo "=== $(date) [eval-lora-all] $* ===" | tee -a "$SUMMARY"
    "$NOTIFY" grpo-v4-lora-eval "$*" 2>/dev/null || true
}

CKPTS=(50 100 150 200 250 300 350 400 450 500)
TOTAL=${#CKPTS[@]}

slog "START eval $TOTAL V4-LoRA checkpoints (OOD + SM each)"

for i in "${!CKPTS[@]}"; do
    N=${CKPTS[$i]}
    IDX=$((i+1))
    CKPT_DIR=$RUN_DIR/checkpoint-$N
    PREFIX="full_lora_hybrid_n2000_v4_ckpt${N}"
    OOD_OUT="$EVAL_DIR/${PREFIX}_ood.json"
    SM_OUT="$EVAL_DIR/${PREFIX}_sm.json"
    EVAL_LOG="$LOG_DIR/eval_ckpt${N}.log"

    if [ -f "$OOD_OUT" ] && [ -f "$SM_OUT" ]; then
        slog "[$IDX/$TOTAL] SKIP ckpt-$N (both ood+sm exist)"
        continue
    fi

    if [ ! -d "$CKPT_DIR" ]; then
        slog "[$IDX/$TOTAL] MISSING ckpt-$N"
        continue
    fi

    # Wipe Triton/torchinductor cache between checkpoints to reduce SIGSEGV risk
    rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null

    slog "[$IDX/$TOTAL] start ckpt-$N"

    MAX_RETRY=2
    for r in $(seq 1 $MAX_RETRY); do
        python -u -m grpo_jssp._run_eval \
            --adapter "$CKPT_DIR" \
            --out-prefix "$EVAL_DIR/$PREFIX" \
            >> "$EVAL_LOG" 2>&1
        EC=$?
        if [ $EC -eq 0 ] && [ -f "$OOD_OUT" ] && [ -f "$SM_OUT" ]; then
            break
        fi
        slog "[$IDX/$TOTAL] ckpt-$N retry $r/$MAX_RETRY (exit=$EC)"
        rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
        sleep 30
    done

    if [ $EC -eq 0 ] && [ -f "$OOD_OUT" ]; then
        FEAS=$(python3 -c "import json; d=json.load(open('$OOD_OUT'))['summary']; print(d['n_feasible'])" 2>/dev/null || echo "?")
        GAP=$(python3 -c "import json; d=json.load(open('$OOD_OUT'))['summary']; print(round(d['mean_gap_to_bks']*100, 2))" 2>/dev/null || echo "?")
        slog "[$IDX/$TOTAL] OK ckpt-$N: OOD ${FEAS}/18 feasible, gap ${GAP}%"
    else
        slog "[$IDX/$TOTAL] FAIL ckpt-$N final exit=$EC"
    fi
done

slog "ALL DONE — see $EVAL_DIR/full_lora_hybrid_n2000_v4_ckpt*"
