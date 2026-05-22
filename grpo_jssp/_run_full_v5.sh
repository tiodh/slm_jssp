#!/bin/bash
# Full V5: hybrid reward (V4) + length control (advantage masking) + grad_accum=4.
# N=2000, max_steps=500 -> 2000 prompts = 1 epoch (same data budget as V4).
# Single-variable test vs V4: only the advantage-masking length control is added.
set -e

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# Pre-flight: config must be hybrid reward + grad_accum=4.
python -c "
from grpo_jssp.config import REWARD_MODE, GRAD_ACCUM_STEPS
import sys
if REWARD_MODE != 'hybrid' or GRAD_ACCUM_STEPS != 4:
    print(f'ERROR: REWARD_MODE={REWARD_MODE!r} GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} '
          f'(expected hybrid / 4)', file=sys.stderr)
    sys.exit(1)
print(f'OK: REWARD_MODE={REWARD_MODE} GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}')
" || { echo "Aborting: config.py must be hybrid + GRAD_ACCUM_STEPS=4."; exit 1; }

NOTIFY=~/.local/bin/notify
RUN_NAME="full_hybrid_lc_n2000_v5"
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

"$NOTIFY" grpo "full V5 starting: hybrid + length-control, N=2000, max_steps=500" >/dev/null 2>&1 || true

run_phase "grpo-v5-train" \
  "python -u -m grpo_jssp.run train --reward-mode hybrid --length-control --max-records 2000 --max-steps 500 --run-name ${RUN_NAME}" \
  "$TRAIN_LOG"

run_phase "grpo-v5-eval" \
  "python -u -m grpo_jssp._run_eval --adapter ${RUN_DIR}/final_adapter --out-prefix ${EVAL_DIR}/${RUN_NAME}" \
  "$EVAL_LOG"

"$NOTIFY" grpo "full V5 ALL DONE" >/dev/null 2>&1 || true
echo "=== $(date) ALL DONE ==="
