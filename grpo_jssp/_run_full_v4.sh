#!/bin/bash
# Full V4: reward-mode hybrid (V4 Hybrid P-GRPO reward) + V3 hyperparams
# (K=4, T=0.7, KL=0.05, grad_accum=4, max_grad_norm=1.0). Only the reward
# function changes vs V3 -- single-variable experiment.
# N=2000, max_steps=500.
set -e

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# Pre-flight: config.py must be on the V4 hybrid reward.
python -c "
from grpo_jssp.config import REWARD_MODE
import sys
if REWARD_MODE != 'hybrid':
    print(f'ERROR: REWARD_MODE={REWARD_MODE!r}, expected hybrid', file=sys.stderr)
    sys.exit(1)
print(f'OK: REWARD_MODE={REWARD_MODE}')
" || { echo "Aborting: config.py REWARD_MODE must be 'hybrid'."; exit 1; }

NOTIFY=~/.local/bin/notify
RUN_NAME="full_hybrid_n2000_v4"
RUNS_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
RUN_DIR="$RUNS_DIR/$RUN_NAME"
TRAIN_LOG="$RUN_DIR/training.log"
EVAL_LOG="$RUN_DIR/eval.log"
mkdir -p "$RUN_DIR" "$EVAL_DIR"

run_phase() {
  local label="$1"
  local cmd="$2"
  local log="$3"

  bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$RUN_DIR/hw_monitor.log" &
  local hw_pid=$!
  bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$log" "$label" &
  local nw_pid=$!

  echo "=== $(date) [$label] start (hw=$hw_pid, nw=$nw_pid) ===" | tee -a "$log"
  "$NOTIFY" "$label" "phase start" >/dev/null 2>&1 || true

  set +e
  bash -c "$cmd" >> "$log" 2>&1
  local ec=$?
  set -e

  echo "=== $(date) [$label] end exit=$ec ===" | tee -a "$log"
  "$NOTIFY" "$label" "phase end exit=$ec" >/dev/null 2>&1 || true
  kill "$hw_pid" 2>/dev/null || true
  kill "$nw_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  return $ec
}

"$NOTIFY" grpo "full V4 HYBRID starting: N=2000, max_steps=500" >/dev/null 2>&1 || true

run_phase "grpo-v4-hybrid-train" \
  "python -u -m grpo_jssp.run train --reward-mode hybrid --max-records 2000 --max-steps 500 --run-name ${RUN_NAME}" \
  "$TRAIN_LOG"

run_phase "grpo-v4-hybrid-eval" \
  "python -u -m grpo_jssp._run_eval --adapter ${RUN_DIR}/final_adapter --out-prefix ${EVAL_DIR}/${RUN_NAME}" \
  "$EVAL_LOG"

"$NOTIFY" grpo "full V4 HYBRID ALL DONE" >/dev/null 2>&1 || true
echo "=== $(date) ALL DONE ==="
