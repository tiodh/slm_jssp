#!/bin/bash
set -e
cd /home/tio/Documents/Starjob
source venv/bin/activate
export LD_LIBRARY_PATH=/home/tio/Documents/Starjob/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

for M in llama ministral qwen2 granite; do
    echo "===== $M ====="
    python3 eval_benchmarks.py --model $M > eval_bench_${M}.log 2>&1
    echo "$M done."
done
echo "ALL DONE"
