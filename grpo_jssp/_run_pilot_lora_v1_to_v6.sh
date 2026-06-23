#!/bin/bash
# LLaMA LoRA pilot: 6 GRPO reward-shape variants (V1..V6), equal-data-budget.
# Replicates per-version hyperparameters from EXPERIMENT_NOTES.md but swaps the
# SFT adapter from rsLoRA -> LoRA (output_alpha32_r32_seq8192_b1_ga8_ep1/ck-9800).
#
# Equal-data-budget pilot (~400 prompts per version):
#   V1/V2: grad_accum=1, 100 optimizer steps -> 400 prompts seen
#   V3-V6: grad_accum=4, 25 optimizer steps  -> 400 prompts seen
#
# Sequential: each phase tears down its Python process so GPU memory is freed
# between runs. A failing phase (e.g. expected V2 collapse) does NOT abort the
# remaining variants.

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

PILOT_LABEL="pilot-lora-v1-to-v6"
SUMMARY_LOG="$RUNS_DIR/pilot_lora_v1_to_v6_summary.log"
mkdir -p "$RUNS_DIR"
echo "=== $(date) PILOT START: $PILOT_LABEL ===" | tee "$SUMMARY_LOG"

run_phase() {
  local label="$1"; shift
  local run_name="$1"; shift
  local max_steps="$1"; shift
  local save_every="$1"; shift
  local run_dir="$RUNS_DIR/$run_name"
  local log="$run_dir/training.log"
  mkdir -p "$run_dir"

  bash /home/tio/Documents/Starjob/grpo_jssp/_monitor_hw.sh "$run_dir/hw_monitor.log" &
  local hw_pid=$!
  bash /home/tio/Documents/Starjob/grpo_jssp/_notify_watcher.sh "$log" "$label" &
  local nw_pid=$!

  echo "=== $(date) [$label] start steps=$max_steps run_dir=$run_dir (hw=$hw_pid, nw=$nw_pid) ===" \
    | tee -a "$log" "$SUMMARY_LOG"
  "$NOTIFY" "$label" "phase start steps=$max_steps" >/dev/null 2>&1 || true

  set +e
  python -u -m grpo_jssp.run train \
    --sft-checkpoint "$SFT_CKPT" \
    --max-steps "$max_steps" \
    --save-every "$save_every" \
    --run-name "$run_name" \
    "$@" \
    >> "$log" 2>&1
  local ec=$?
  set -e

  echo "=== $(date) [$label] end exit=$ec ===" | tee -a "$log" "$SUMMARY_LOG"
  "$NOTIFY" "$label" "phase end exit=$ec" >/dev/null 2>&1 || true
  kill "$hw_pid" 2>/dev/null || true
  kill "$nw_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  # do NOT propagate ec -- expected collapses (V2) should not abort the sweep
  return 0
}

"$NOTIFY" grpo "$PILOT_LABEL starting: SFT=LoRA, equal-data-budget ~400 prompts/version" \
  >/dev/null 2>&1 || true

# V1: stratified, KL=0.04, grad_accum=1, T=0.8, 100 steps (400 prompts)
run_phase "grpo-pilot-lora-v1" "pilot_lora_v1_stratified_n100" 100 25 \
  --reward-mode stratified \
  --kl-coef 0.04 \
  --grad-accum 1 \
  --temperature 0.8

# V2: stratified_v2 (V1 + lp + eos), KL=0.10, grad_accum=1, T=0.8, 100 steps
run_phase "grpo-pilot-lora-v2" "pilot_lora_v2_stratified_v2_n100" 100 25 \
  --reward-mode stratified_v2 \
  --kl-coef 0.10 \
  --grad-accum 1 \
  --temperature 0.8 \
  --lp-alpha 0.10 \
  --eos-beta 0.05

# V3: stratified, KL=0.05, grad_accum=4, T=0.7, 25 steps (400 prompts)
run_phase "grpo-pilot-lora-v3" "pilot_lora_v3_stratified_n25" 25 5 \
  --reward-mode stratified \
  --kl-coef 0.05 \
  --grad-accum 4 \
  --temperature 0.7

# V4: hybrid, KL=0.05, grad_accum=4, T=0.7, 25 steps
run_phase "grpo-pilot-lora-v4" "pilot_lora_v4_hybrid_n25" 25 5 \
  --reward-mode hybrid \
  --kl-coef 0.05 \
  --grad-accum 4 \
  --temperature 0.7

# V5: hybrid + length control, KL=0.05, grad_accum=4, T=0.7, 25 steps
run_phase "grpo-pilot-lora-v5" "pilot_lora_v5_hybrid_lc_n25" 25 5 \
  --reward-mode hybrid \
  --length-control \
  --kl-coef 0.05 \
  --grad-accum 4 \
  --temperature 0.7

# V6: stratified + length control, KL=0.05, grad_accum=4, T=0.7, 25 steps
run_phase "grpo-pilot-lora-v6" "pilot_lora_v6_stratified_lc_n25" 25 5 \
  --reward-mode stratified \
  --length-control \
  --kl-coef 0.05 \
  --grad-accum 4 \
  --temperature 0.7

echo "=== $(date) ALL 6 PILOT PHASES DONE ===" | tee -a "$SUMMARY_LOG"
"$NOTIFY" grpo "$PILOT_LABEL ALL DONE" >/dev/null 2>&1 || true
