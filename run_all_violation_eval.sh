#!/bin/bash
# Re-evaluasi 4 model x 2 metode dengan validator 5-kategori.
# Starjob SM (200 samples) + OOD FT+LA (18 instances), no TA.
set -e
cd /home/tio/Documents/Starjob
source venv/bin/activate
export LD_LIBRARY_PATH="venv/lib/python3.12/site-packages/nvidia/cu13/lib:venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

NUM_SAMPLES=200
LOGDIR=logs_violation_eval
mkdir -p $LOGDIR

run() {
    local label=$1
    local script=$2
    shift 2
    local args="$@"
    echo "===== [$(date +'%H:%M:%S')] $label ====="
    python $script $args 2>&1 | tee $LOGDIR/${label}.log
    echo "===== [$(date +'%H:%M:%S')] $label DONE ====="
}

# Starjob SM (8 runs)
for M in llama granite ministral qwen2; do
    run "starjob_lora_${M}"   eval_lora.py    --model $M --num_samples $NUM_SAMPLES
    run "starjob_rslora_${M}" eval_rslora.py  --model $M --num_samples $NUM_SAMPLES
done

# OOD FT+LA only (8 runs, --skip-tai)
for M in llama granite ministral qwen2; do
    run "ood_lora_${M}"   eval_benchmarks.py        --model $M --skip-tai
    run "ood_rslora_${M}" eval_rslora_benchmarks.py --model $M --skip-tai
done

echo "ALL DONE @ $(date)"
