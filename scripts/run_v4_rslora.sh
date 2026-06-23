#!/bin/bash
# GRPO V4 (hybrid reward) on rsLoRA SFT base.
# Mirror of full_lora_hybrid_n2000_v4 hyperparameters but pointed at the
# rsLoRA SFT checkpoint (use_rslora=true) instead of LoRA SFT.
# Purpose: validate that V4 hybrid reward generalizes to rsLoRA base.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# Workaround for GCC 13.3 internal compiler error (cfgcleanup.cc:580) that
# Triton JIT compilation hits for rsLoRA's α/√r scaled kernel patterns.
# Force Triton and any inductor C++ codegen to use clang-18 instead of gcc-13.
# torch.compile re-enabled — only the underlying C compiler changes.
export CC=/usr/bin/clang-18
export CXX=/usr/bin/clang++-18
export TRITON_CC=/usr/bin/clang-18

# Bypass glibc malloc — V4-rsLoRA hits intermittent "double free or corruption
# (out)" / SIGSEGV in the libcuda 580 + Unsloth rsLoRA path. jemalloc has
# different free-list bookkeeping that tolerates this access pattern.
# No slowdown observed; common workaround in ML repos.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Diagnostic: force synchronous CUDA so crash stack trace points to the actual
# offending kernel launch (otherwise async queue masks the origin). ~20% slowdown
# acceptable for diagnosis. Remove once root cause confirmed.
export CUDA_LAUNCH_BLOCKING=1

NOTIFY=~/.local/bin/notify
SFT=/home/tio/Documents/Starjob/output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800
RUNS=/home/tio/Documents/Starjob/grpo_jssp/runs
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
RUN_NAME=full_rslora_hybrid_n2000_v4
RUN_DIR=$RUNS/$RUN_NAME
TRAIN_LOG=$RUN_DIR/training.log
EVAL_LOG=$RUN_DIR/eval.log
SUMMARY=$RUNS/v4_rslora_summary.log
MAX_ATTEMPTS=50

mkdir -p "$RUN_DIR" "$EVAL_DIR"

slog() {
    echo "=== $(date) [v4-rslora] $* ===" | tee -a "$SUMMARY"
    "$NOTIFY" grpo-v4-rslora "$*" 2>/dev/null || true
}

# Sanity: SFT base must be rsLoRA
if ! grep -q '"use_rslora": true' "$SFT/adapter_config.json" 2>/dev/null; then
    slog "ABORT: SFT base at $SFT is not rsLoRA (use_rslora != true)"
    exit 1
fi
slog "SFT base verified rsLoRA: $SFT"

if [ -d "$RUN_DIR/final_adapter" ] && [ -e "$RUN_DIR/final_adapter/adapter_model.safetensors" ]; then
    slog "SKIP — final_adapter already exists, running eval only"
else
    rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
    slog "START V4 hybrid on rsLoRA SFT (max $MAX_ATTEMPTS attempts, auto-resume)"

    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        if [ -d "$RUN_DIR/final_adapter" ]; then
            slog "final_adapter saved, done"
            break
        fi

        LATEST_CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [[ -n "$LATEST_CKPT" ]]; then
            RESUME_ARG="--resume-from $LATEST_CKPT"
            slog "attempt $attempt/$MAX_ATTEMPTS resume from $(basename $LATEST_CKPT)"
        else
            RESUME_ARG=""
            slog "attempt $attempt/$MAX_ATTEMPTS fresh"
        fi

        bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$RUN_DIR/hw_monitor.log" &
        HW_PID=$!
        NOTIFY_EVERY=50 bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$TRAIN_LOG" "grpo-v4-rslora" &
        NW_PID=$!

        CUDA_LAUNCH_BLOCKING=1 python -u -m grpo_jssp.run train \
            --sft-checkpoint "$SFT" \
            --reward-mode hybrid \
            --max-records 2000 \
            --max-steps 500 \
            --kl-coef 0.05 \
            --grad-accum 4 \
            --temperature 0.7 \
            --save-every 50 \
            --run-name "$RUN_NAME" \
            $RESUME_ARG \
            >> "$TRAIN_LOG" 2>&1
        EC=$?

        kill "$HW_PID" "$NW_PID" 2>/dev/null
        wait 2>/dev/null

        if [[ $EC -eq 0 ]]; then
            slog "attempt $attempt OK exit=0"
            break
        fi

        slog "attempt $attempt CRASHED exit=$EC"
        [[ $attempt -lt $MAX_ATTEMPTS ]] && { slog "cooldown 60s..."; sleep 60; }
    done
fi

if [ -d "$RUN_DIR/final_adapter" ]; then
    slog "starting eval (OOD + SM)"
    python -u -m grpo_jssp._run_eval \
        --adapter "$RUN_DIR/final_adapter" \
        --out-prefix "$EVAL_DIR/$RUN_NAME" \
        >> "$EVAL_LOG" 2>&1
    EVAL_EC=$?
    if [ $EVAL_EC -eq 0 ] && [ -f "$EVAL_DIR/${RUN_NAME}_ood.json" ]; then
        FEAS=$(python3 -c "import json; d=json.load(open('$EVAL_DIR/${RUN_NAME}_ood.json')); print(d['summary']['n_feasible'])" 2>/dev/null || echo "?")
        GAP=$(python3 -c "import json; d=json.load(open('$EVAL_DIR/${RUN_NAME}_ood.json')); print(round(d['summary']['mean_gap_to_bks']*100, 2))" 2>/dev/null || echo "?")
        slog "EVAL DONE: OOD ${FEAS}/18 feasible, mean_gap ${GAP}%"
    else
        slog "EVAL FAILED exit=$EVAL_EC"
    fi
else
    slog "TRAIN FAILED after $MAX_ATTEMPTS attempts — no final_adapter"
fi

slog "ALL DONE"
