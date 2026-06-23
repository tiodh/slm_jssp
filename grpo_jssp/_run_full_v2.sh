#!/bin/bash
# Full GRPO run v2 with notify + HW monitoring + new reward shaping.
# Usage: ./_run_full_v2.sh [stratified|uniform|both]
set -e

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
MODE="${1:-stratified}"
RUNS_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
mkdir -p "$EVAL_DIR"

run_phase() {
  local label="$1"
  local cmd="$2"
  local log="$3"
  local run_dir
  run_dir=$(dirname "$log")
  mkdir -p "$run_dir"

  # Start HW monitor + notify watcher
  bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$run_dir/hw_monitor.log" &
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

  # Stop watchers
  kill "$hw_pid" 2>/dev/null || true
  kill "$nw_pid" 2>/dev/null || true
  wait 2>/dev/null || true

  return $ec
}

train_and_eval() {
  local mode="$1"
  local run_name="full_${mode}_2000_v2"
  local run_dir="$RUNS_DIR/$run_name"
  local train_log="$run_dir/training.log"
  local eval_log="$run_dir/eval.log"

  run_phase "grpo-${mode}-train" \
    "python -u -m grpo_jssp.run train --reward-mode ${mode} --max-steps 2000 --run-name ${run_name}" \
    "$train_log"

  run_phase "grpo-${mode}-eval" \
    "python -u -m grpo_jssp._run_eval --adapter ${run_dir}/final_adapter --out-prefix ${EVAL_DIR}/${run_name}" \
    "$eval_log"
}

"$NOTIFY" grpo "full run v2 starting: mode=$MODE" >/dev/null 2>&1 || true

case "$MODE" in
  stratified) train_and_eval stratified ;;
  uniform)    train_and_eval uniform ;;
  both)       train_and_eval stratified ; train_and_eval uniform ;;
  *) echo "unknown mode: $MODE (use stratified|uniform|both)" >&2 ; exit 2 ;;
esac

"$NOTIFY" grpo "full run v2 ALL DONE" >/dev/null 2>&1 || true
echo "=== $(date) ALL DONE ==="
