#!/bin/bash
# GPT-5 only OOD eval, 2 reasoning_effort variants. Minimal + low.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source ~/.openai_env

SUMMARY=/home/tio/Documents/Starjob/openai_gpt5_summary.log
echo "=== $(date) START GPT-5 only OOD eval ===" | tee "$SUMMARY"

declare -a JOBS=(
    "_minimal|gpt-5|1.50|--max-tokens 8000 --reasoning-effort minimal"
    "_low|gpt-5|2.00|--max-tokens 10000 --reasoning-effort low"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r SUFFIX MODEL BUDGET EXTRA <<< "$job"
    OUT="metrics_openai_${MODEL}${SUFFIX}_benchmarks.json"
    LOG="/tmp/openai_eval_${MODEL}${SUFFIX}.log"

    if [ -s "$OUT" ]; then
        echo "--- SKIP $MODEL$SUFFIX (exists) ---" | tee -a "$SUMMARY"
        continue
    fi

    echo | tee -a "$SUMMARY"
    echo "=== $(date) [$MODEL$SUFFIX] start budget=\$$BUDGET ===" | tee -a "$SUMMARY"

    python eval_openai_benchmarks.py \
        --model "$MODEL" --budget "$BUDGET" --out "$OUT" $EXTRA \
        2>&1 | tee "$LOG"
    EC=${PIPESTATUS[0]}
    echo "=== $(date) [$MODEL$SUFFIX] exit=$EC ===" | tee -a "$SUMMARY"

    if [ -s "$OUT" ]; then
        python3 -c "
import json
with open('$OUT') as f: d = json.load(f)
print(f\"[$MODEL$SUFFIX] cost=\${d['total_cost_usd']:.3f}  feasible={sum(1 for r in d['results'] if r['feasible'])}/{d['n_instances_evaluated']}  parseable={sum(1 for r in d['results'] if r['ops_emitted']>0)}/{d['n_instances_evaluated']}\")
" 2>&1 | tee -a "$SUMMARY"
    fi
done

echo | tee -a "$SUMMARY"
echo "=== $(date) GPT-5 ALL DONE ===" | tee -a "$SUMMARY"
