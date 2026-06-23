#!/bin/bash
# Full V4: stratified mode + EQUAL WEIGHTS (all 0.2) + V3 hyperparams.
# Requires config.py WEIGHTS to be set to equal-0.2 before launch.
# Pre-flight check below verifies this.
set -e

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# Pre-flight: ensure config.py has equal weights
python -c "
from grpo_jssp.config import WEIGHTS
import sys
expected = 0.2
vals = list(WEIGHTS.values())
if not all(abs(v - expected) < 1e-6 for v in vals):
    print(f'ERROR: WEIGHTS not equal-0.2: {WEIGHTS}', file=sys.stderr)
    sys.exit(1)
print(f'OK: WEIGHTS = {WEIGHTS}')
" || { echo "Aborting: config.py WEIGHTS must be set to all 0.2 first."; exit 1; }

NOTIFY=~/.local/bin/notify
RUN_NAME="full_stratified_n2000_v4_equal"
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

"$NOTIFY" grpo "full V4 EQUAL-WEIGHTS starting: N=2000, max_steps=500" >/dev/null 2>&1 || true

run_phase "grpo-v4-equal-train" \
  "python -u -m grpo_jssp.run train --reward-mode stratified --max-records 2000 --max-steps 500 --run-name ${RUN_NAME}" \
  "$TRAIN_LOG"

run_phase "grpo-v4-equal-eval" \
  "python -u -m grpo_jssp._run_eval --adapter ${RUN_DIR}/final_adapter --out-prefix ${EVAL_DIR}/${RUN_NAME}" \
  "$EVAL_LOG"

"$NOTIFY" grpo "full V4 EQUAL-WEIGHTS ALL DONE" >/dev/null 2>&1 || true
echo "=== $(date) ALL DONE ==="
