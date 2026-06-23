#!/bin/bash
# Eval all 9 saved checkpoints of V4-rsLoRA on OOD + SM.
# Output: eval_results/full_rslora_hybrid_n2000_v4_ckpt{N}_{ood,sm}.json

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
RUN_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs/full_rslora_hybrid_n2000_v4
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
LOG_DIR=$RUN_DIR/eval_logs
SUMMARY=$RUN_DIR/eval_all_ckpts_summary.log

mkdir -p "$LOG_DIR"

slog() {
    echo "=== $(date) [eval-all] $* ===" | tee -a "$SUMMARY"
    "$NOTIFY" grpo-v4-rslora-eval "$*" 2>/dev/null || true
}

CKPTS=(50 100 150 200 250 300 350 400 450)
TOTAL=${#CKPTS[@]}

slog "START eval $TOTAL checkpoints (OOD + SM each)"

for i in "${!CKPTS[@]}"; do
    N=${CKPTS[$i]}
    IDX=$((i+1))
    CKPT_DIR=$RUN_DIR/checkpoint-$N
    PREFIX="full_rslora_hybrid_n2000_v4_ckpt${N}"
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

    slog "[$IDX/$TOTAL] start ckpt-$N"

    python -u -m grpo_jssp._run_eval \
        --adapter "$CKPT_DIR" \
        --out-prefix "$EVAL_DIR/$PREFIX" \
        >> "$EVAL_LOG" 2>&1
    EC=$?

    if [ $EC -eq 0 ] && [ -f "$OOD_OUT" ]; then
        FEAS=$(python3 -c "import json; d=json.load(open('$OOD_OUT')); print(d['summary']['n_feasible'])" 2>/dev/null || echo "?")
        GAP=$(python3 -c "import json; d=json.load(open('$OOD_OUT')); print(round(d['summary']['mean_gap_to_bks']*100, 2))" 2>/dev/null || echo "?")
        slog "[$IDX/$TOTAL] OK ckpt-$N: OOD ${FEAS}/18 feasible, gap ${GAP}%"
    else
        slog "[$IDX/$TOTAL] FAIL ckpt-$N exit=$EC"
    fi
done

slog "ALL DONE — see $EVAL_DIR/full_rslora_hybrid_n2000_v4_ckpt*"
