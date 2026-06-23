#!/bin/bash
# Run full stratified, eval, then full uniform, eval.
# Background runner — chained so each phase starts when prior succeeds.
set -e

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

LOG=/home/tio/Documents/Starjob/grpo_jssp/runs/full_runs.log

echo "=== $(date) [stratified] training start ===" | tee -a "$LOG"
python -u -m grpo_jssp.run train \
    --reward-mode stratified \
    --max-steps 2000 \
    --run-name full_stratified_2000 2>&1 | tee -a "$LOG"
echo "=== $(date) [stratified] training done ===" | tee -a "$LOG"

echo "=== $(date) [stratified] eval start ===" | tee -a "$LOG"
python -u -m grpo_jssp._run_eval \
    --adapter /home/tio/Documents/Starjob/grpo_jssp/runs/full_stratified_2000/final_adapter \
    --out-prefix /home/tio/Documents/Starjob/grpo_jssp/eval_results/full_stratified_2000 2>&1 | tee -a "$LOG"
echo "=== $(date) [stratified] eval done ===" | tee -a "$LOG"

echo "=== $(date) [uniform] training start ===" | tee -a "$LOG"
python -u -m grpo_jssp.run train \
    --reward-mode uniform \
    --max-steps 2000 \
    --run-name full_uniform_2000 2>&1 | tee -a "$LOG"
echo "=== $(date) [uniform] training done ===" | tee -a "$LOG"

echo "=== $(date) [uniform] eval start ===" | tee -a "$LOG"
python -u -m grpo_jssp._run_eval \
    --adapter /home/tio/Documents/Starjob/grpo_jssp/runs/full_uniform_2000/final_adapter \
    --out-prefix /home/tio/Documents/Starjob/grpo_jssp/eval_results/full_uniform_2000 2>&1 | tee -a "$LOG"
echo "=== $(date) [uniform] eval done ===" | tee -a "$LOG"

echo "=== $(date) ALL DONE ===" | tee -a "$LOG"
