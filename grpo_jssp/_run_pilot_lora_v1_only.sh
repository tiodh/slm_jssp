#!/bin/bash
# Retry V1 from the V1-V6 LoRA pilot. First attempt crashed with SIGSEGV at
# step 63/100 (preserved at pilot_lora_v1_stratified_n100_attempt1_partial/).
# Reward was healthy throughout the partial run, so the crash was a torch/CUDA
# allocator flake, not a config issue. Same hyperparams as attempt 1:
#   stratified reward, KL=0.04, grad_accum=1, T=0.8, 100 steps (400 prompts).

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
RUNS_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs
SFT_CKPT=/home/tio/Documents/Starjob/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800

if [ ! -d "$SFT_CKPT" ]; then
  echo "ERROR: LoRA SFT checkpoint not found: $SFT_CKPT" >&2
  exit 1
fi

LABEL="grpo-pilot-lora-v1-rerun"
RUN_NAME="pilot_lora_v1_stratified_n100"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
LOG="$RUN_DIR/training.log"
SUMMARY_LOG="$RUNS_DIR/pilot_lora_v1_to_v6_summary.log"

mkdir -p "$RUN_DIR"

bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$RUN_DIR/hw_monitor.log" &
HW_PID=$!
bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$LOG" "$LABEL" &
NW_PID=$!

echo "=== $(date) [$LABEL] start steps=100 run_dir=$RUN_DIR (hw=$HW_PID, nw=$NW_PID) ===" \
  | tee -a "$LOG" "$SUMMARY_LOG"
"$NOTIFY" "$LABEL" "V1 rerun start" >/dev/null 2>&1 || true

set +e
python -u -m grpo_jssp.run train \
  --sft-checkpoint "$SFT_CKPT" \
  --max-steps 100 \
  --save-every 25 \
  --run-name "$RUN_NAME" \
  --reward-mode stratified \
  --kl-coef 0.04 \
  --grad-accum 1 \
  --temperature 0.8 \
  >> "$LOG" 2>&1
EC=$?
set -e

echo "=== $(date) [$LABEL] end exit=$EC ===" | tee -a "$LOG" "$SUMMARY_LOG"
"$NOTIFY" "$LABEL" "V1 rerun end exit=$EC" >/dev/null 2>&1 || true
kill "$HW_PID" 2>/dev/null || true
kill "$NW_PID" 2>/dev/null || true
wait 2>/dev/null || true
exit $EC
