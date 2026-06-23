#!/bin/bash
# Full GRPO training V1-V6 on LoRA SFT base.
# v2: NO set -e (single phase crash does not abort pipeline), 60s cooldown
#     between phases, retry-up-to-2 on training crash, skip-if-final-adapter-exists.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

NOTIFY=~/.local/bin/notify
RUNS_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs
EVAL_DIR=/home/tio/Documents/Starjob/grpo_jssp/eval_results
SFT_CKPT=/home/tio/Documents/Starjob/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800
SUMMARY_LOG="$RUNS_DIR/full_lora_all_summary.log"
mkdir -p "$EVAL_DIR"

if [ ! -d "$SFT_CKPT" ]; then
  echo "ERROR: LoRA SFT checkpoint not found: $SFT_CKPT" >&2
  exit 1
fi

run_phase_with_retry() {
  local label="$1"; local cmd="$2"; local log="$3"; local max_attempts="${4:-2}"
  for attempt in $(seq 1 $max_attempts); do
    bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$(dirname $log)/hw_monitor.log" &
    local hw_pid=$!
    bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$log" "$label" &
    local nw_pid=$!
    echo "=== $(date) [$label] attempt ${attempt}/${max_attempts} start (hw=$hw_pid, nw=$nw_pid) ===" | tee -a "$log" "$SUMMARY_LOG"
    "$NOTIFY" "$label" "phase start (attempt $attempt)" >/dev/null 2>&1 || true
    bash -c "$cmd" >> "$log" 2>&1
    local ec=$?
    echo "=== $(date) [$label] attempt ${attempt} end exit=$ec ===" | tee -a "$log" "$SUMMARY_LOG"
    kill "$hw_pid" 2>/dev/null || true
    kill "$nw_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    if [ $ec -eq 0 ]; then
      "$NOTIFY" "$label" "phase done exit=0" >/dev/null 2>&1 || true
      return 0
    fi
    "$NOTIFY" "$label" "phase crashed exit=$ec (attempt $attempt)" >/dev/null 2>&1 || true
    if [ $attempt -lt $max_attempts ]; then
      echo "    cooldown 60s before retry..." | tee -a "$SUMMARY_LOG"
      sleep 60
    fi
  done
  echo "=== $(date) [$label] FAILED after ${max_attempts} attempts ===" | tee -a "$SUMMARY_LOG"
  "$NOTIFY" "$label" "FAILED after $max_attempts attempts" >/dev/null 2>&1 || true
  return 1
}

train_and_eval() {
  local ver="$1"
  local run_name="$2"
  local train_args="$3"
  local run_dir="$RUNS_DIR/$run_name"
  local train_log="$run_dir/training.log"
  local eval_log="$run_dir/eval.log"
  mkdir -p "$run_dir"

  # Skip training if final_adapter already exists
  if [ -d "$run_dir/final_adapter" ]; then
    echo "=== $(date) [grpo-lora-${ver}-train] SKIP (final_adapter exists) ===" | tee -a "$SUMMARY_LOG"
  else
    run_phase_with_retry "grpo-lora-${ver}-train" \
      "python -u -m grpo_jssp.run train --sft-checkpoint ${SFT_CKPT} ${train_args} --run-name ${run_name}" \
      "$train_log" 2
    if [ $? -ne 0 ]; then
      echo "WARN: ${ver} train failed after retries, skipping eval"
      return 1
    fi
  fi

  # Skip eval if both outputs exist
  local sm_out="$EVAL_DIR/${run_name}_sm.json"
  local ood_out="$EVAL_DIR/${run_name}_ood.json"
  if [ -s "$sm_out" ] && [ -s "$ood_out" ]; then
    echo "=== $(date) [grpo-lora-${ver}-eval] SKIP (already done) ===" | tee -a "$SUMMARY_LOG"
  else
    run_phase_with_retry "grpo-lora-${ver}-eval" \
      "python -u -m grpo_jssp._run_eval --adapter ${run_dir}/final_adapter --out-prefix ${EVAL_DIR}/${run_name}" \
      "$eval_log" 3
  fi
  return 0
}

echo "=== $(date) FULL LoRA ALL START (v2: skip/cooldown/retry) ===" | tee -a "$SUMMARY_LOG"
"$NOTIFY" grpo "full LoRA v2 start (V1-V6)" >/dev/null 2>&1 || true

FAILED=()

# V1: stratified, KL=0.04, ga=1, T=0.8, max_steps=2000
# RESUME: 2026-06-05 freeze killed V1 at step 700; resume from checkpoint-700.
# Remove --resume-from after V1 finishes if re-running the pipeline from scratch.
V1_RESUME="$RUNS_DIR/full_lora_stratified_2000_v1/checkpoint-700"
if [ -d "$V1_RESUME" ] && [ ! -d "$RUNS_DIR/full_lora_stratified_2000_v1/final_adapter" ]; then
  V1_RESUME_ARG="--resume-from $V1_RESUME"
else
  V1_RESUME_ARG=""
fi
train_and_eval "v1" "full_lora_stratified_2000_v1" \
  "--reward-mode stratified --max-steps 2000 --kl-coef 0.04 --grad-accum 1 --temperature 0.8 $V1_RESUME_ARG" || FAILED+=("v1")
sleep 60

# V2: stratified_v2, KL=0.10, ga=1, T=0.8, max_steps=2000
train_and_eval "v2" "full_lora_stratified_v2_2000_v2" \
  "--reward-mode stratified_v2 --max-steps 2000 --kl-coef 0.10 --grad-accum 1 --temperature 0.8" || FAILED+=("v2")
sleep 60

# V3: stratified, KL=0.05, ga=4, T=0.7, max_records=2000, max_steps=500
train_and_eval "v3" "full_lora_stratified_n2000_v3" \
  "--reward-mode stratified --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7" || FAILED+=("v3")
sleep 60

# V4: hybrid, KL=0.05, ga=4, T=0.7, max_records=2000, max_steps=500
train_and_eval "v4" "full_lora_hybrid_n2000_v4" \
  "--reward-mode hybrid --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7" || FAILED+=("v4")
sleep 60

# V5: hybrid + length-control
train_and_eval "v5" "full_lora_hybrid_lc_n2000_v5" \
  "--reward-mode hybrid --length-control --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7" || FAILED+=("v5")
sleep 60

# V6: stratified + length-control
train_and_eval "v6" "full_lora_stratified_lc_n2000_v6" \
  "--reward-mode stratified --length-control --max-records 2000 --max-steps 500 --kl-coef 0.05 --grad-accum 4 --temperature 0.7" || FAILED+=("v6")

echo "=== $(date) FULL LoRA ALL DONE (failed: ${FAILED[*]:-none}) ===" | tee -a "$SUMMARY_LOG"
"$NOTIFY" grpo "full LoRA v2 DONE (failed: ${FAILED[*]:-none})" >/dev/null 2>&1 || true
exit 0
