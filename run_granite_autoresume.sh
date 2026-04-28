#!/bin/bash
# Auto-resume wrapper for Granite training
# Keeps restarting on crash until training completes successfully

source /home/tio/Documents/Starjob/venv/bin/activate
export LD_LIBRARY_PATH=/home/tio/Documents/Starjob/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export WANDB_MODE=offline
export WANDB__DISABLE_STATS=true

MAX_RETRIES=20
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))
    echo "=========================================="
    echo "[AUTO-RESUME] Attempt $RETRY / $MAX_RETRIES"
    echo "[AUTO-RESUME] $(date)"
    echo "=========================================="

    python train_granite_8b.py --gradient_accumulation_steps 8
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[AUTO-RESUME] Training completed successfully!"
        break
    else
        echo "[AUTO-RESUME] Crashed with exit code $EXIT_CODE. Waiting 10s before resuming..."
        sleep 10
    fi
done

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "[AUTO-RESUME] Exceeded max retries ($MAX_RETRIES). Giving up."
fi
