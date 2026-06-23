#!/bin/bash
# Canary V1 fresh — sanity-check GPU + Triton cache + driver before full pipeline.
# Usage: bash run_canary.sh [max_steps]
#   default max_steps = 20 (~7 min)

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

STEPS=${1:-20}
RUN_NAME=canary_v1_steps${STEPS}_$(date +%H%M)

echo "=== $(date) CANARY start steps=$STEPS run_name=$RUN_NAME ==="
python -u -m grpo_jssp.run train \
  --sft-checkpoint /home/tio/Documents/Starjob/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800 \
  --reward-mode stratified --max-steps "$STEPS" --kl-coef 0.04 --grad-accum 1 --temperature 0.8 \
  --run-name "$RUN_NAME"
ec=$?
echo "=== $(date) CANARY end exit=$ec steps=$STEPS ==="
exit $ec
