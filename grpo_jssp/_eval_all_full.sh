#!/bin/bash
# Eval 3 pre-collapse checkpoints from full runs.
set -e
cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
mkdir -p "$EVAL_DIR"

run_eval() {
  local label="$1"
  local adapter="$2"
  local out_prefix="$3"
  echo "=== $(date) [$label] start ==="
  "$NOTIFY" "$label" "eval start" >/dev/null 2>&1 || true
  set +e
  python -u -m grpo_jssp._run_eval --adapter "$adapter" --out-prefix "$out_prefix"
  local ec=$?
  set -e
  echo "=== $(date) [$label] end exit=$ec ==="
  "$NOTIFY" "$label" "eval end exit=$ec" >/dev/null 2>&1 || true
  return $ec
}

run_eval "eval-v1-ckpt600" \
  "grpo_jssp/runs/full_stratified_2000/checkpoint-600" \
  "${EVAL_DIR}/full_v1_ckpt600"

run_eval "eval-v2-ckpt200" \
  "grpo_jssp/runs/full_stratified_2000_v2/checkpoint-200" \
  "${EVAL_DIR}/full_v2_ckpt200"

run_eval "eval-v3-ckpt200" \
  "grpo_jssp/runs/full_stratified_n2000_v3/checkpoint-200" \
  "${EVAL_DIR}/full_v3_ckpt200"

"$NOTIFY" grpo "all 3 full ckpt evals DONE" >/dev/null 2>&1 || true
echo "=== $(date) ALL DONE ==="
