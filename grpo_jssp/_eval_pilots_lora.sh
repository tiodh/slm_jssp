#!/bin/bash
# Eval all 6 pilot LoRA adapters (V1-V6) on OOD + StarJob SM.
# v2: skip if output exists, 60s cooldown between phases, retry-up-to-3 on crash.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
RUNS_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
SUMMARY_LOG="$RUNS_DIR/pilot_lora_evals_summary.log"
mkdir -p "$EVAL_DIR"

eval_with_retry() {
  local label="$1"
  local adapter="$2"
  local out_prefix="$3"
  local log_file="$4"

  # Skip if both SM and OOD outputs already exist with non-zero size
  local sm_out="${out_prefix}_sm.json"
  local ood_out="${out_prefix}_ood.json"
  if [ -s "$sm_out" ] && [ -s "$ood_out" ]; then
    echo "=== $(date) [$label] SKIP (already done: SM=$(stat -c%s $sm_out)B, OOD=$(stat -c%s $ood_out)B) ===" | tee -a "$SUMMARY_LOG"
    return 0
  fi

  local max_attempts=3
  for attempt in $(seq 1 $max_attempts); do
    echo "=== $(date) [$label] attempt ${attempt}/${max_attempts} start ===" | tee -a "$SUMMARY_LOG"
    "$NOTIFY" "$label" "eval start (attempt $attempt)" >/dev/null 2>&1 || true
    python -u -m grpo_jssp._run_eval --adapter "$adapter" --out-prefix "$out_prefix" > "$log_file" 2>&1
    local ec=$?
    echo "=== $(date) [$label] attempt ${attempt} end exit=$ec ===" | tee -a "$SUMMARY_LOG"
    if [ $ec -eq 0 ]; then
      "$NOTIFY" "$label" "eval done exit=0" >/dev/null 2>&1 || true
      return 0
    fi
    "$NOTIFY" "$label" "eval crashed exit=$ec (attempt $attempt)" >/dev/null 2>&1 || true
    if [ $attempt -lt $max_attempts ]; then
      echo "    cooldown 60s before retry..." | tee -a "$SUMMARY_LOG"
      sleep 60
    fi
  done
  echo "=== $(date) [$label] FAILED after ${max_attempts} attempts ===" | tee -a "$SUMMARY_LOG"
  "$NOTIFY" "$label" "eval FAILED after $max_attempts attempts" >/dev/null 2>&1 || true
  return 1
}

echo "=== $(date) PILOT LoRA EVAL ALL START (v2: skip/cooldown/retry) ===" | tee -a "$SUMMARY_LOG"
"$NOTIFY" grpo "pilot LoRA eval v2 start" >/dev/null 2>&1 || true

declare -A ADAPTERS=(
  ["v1"]="$RUNS_DIR/pilot_lora_v1_stratified_n100/final_adapter"
  ["v2"]="$RUNS_DIR/pilot_lora_v2_stratified_v2_n100/final_adapter"
  ["v3"]="$RUNS_DIR/pilot_lora_v3_stratified_n25/final_adapter"
  ["v4"]="$RUNS_DIR/pilot_lora_v4_hybrid_n25/final_adapter"
  ["v5"]="$RUNS_DIR/pilot_lora_v5_hybrid_lc_n25/final_adapter"
  ["v6"]="$RUNS_DIR/pilot_lora_v6_stratified_lc_n25/final_adapter"
)

FAILED=()
for v in v1 v2 v3 v4 v5 v6; do
  adapter="${ADAPTERS[$v]}"
  out_prefix="$EVAL_DIR/pilot_lora_${v}"
  log_file="$(dirname $adapter)/../eval.log"
  eval_with_retry "eval-pilot-lora-${v}" "$adapter" "$out_prefix" "$log_file" || FAILED+=("$v")
  # Cooldown between phases — let driver release CUDA context cleanly
  echo "    inter-phase cooldown 60s..." | tee -a "$SUMMARY_LOG"
  sleep 60
done

echo "=== $(date) PILOT LoRA EVAL ALL DONE (failed: ${FAILED[*]:-none}) ===" | tee -a "$SUMMARY_LOG"
"$NOTIFY" grpo "pilot LoRA eval v2 DONE (failed: ${FAILED[*]:-none})" >/dev/null 2>&1 || true
exit 0
