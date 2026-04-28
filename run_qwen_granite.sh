#!/bin/bash
# Wait for ministral python to finish, then run qwen2 + granite with --skip-tai
cd /home/tio/Documents/Starjob
source venv/bin/activate
export LD_LIBRARY_PATH=/home/tio/Documents/Starjob/venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# Wait for running ministral eval to finish
while pgrep -f "eval_benchmarks.py --model ministral" >/dev/null; do
    sleep 30
done
echo "Ministral finished at $(date)"

for M in qwen2 granite; do
    echo "===== $M (skip TAI) ====="
    python3 eval_benchmarks.py --model $M --skip-tai > eval_bench_${M}.log 2>&1
    echo "$M done at $(date)"
done
echo "ALL DONE"
