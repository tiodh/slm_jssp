#!/bin/bash
# Run OOD eval for ALL OpenAI models w/ multiple reasoning_effort variants.
# Sorted cheap -> expensive; per-model budget cap.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source ~/.openai_env

SUMMARY=/home/tio/Documents/Starjob/openai_all_summary.log
echo "=== $(date) START all-models OpenAI OOD eval (tiered C) ===" | tee "$SUMMARY"

# Format: SUFFIX | MODEL | BUDGET | EXTRA_ARGS
# SUFFIX appended to output filename so multiple reasoning runs don't collide.
# Empty suffix = no suffix.
declare -a JOBS=(
    "|gpt-4o-mini|0.10|--max-tokens 4000"
    "|gpt-4.1-mini|0.30|--max-tokens 4000"
    "|gpt-4.1|0.40|--max-tokens 4000"
    "_minimal|gpt-5-mini|0.50|--max-tokens 6000 --reasoning-effort minimal"
    "_minimal|gpt-5|1.50|--max-tokens 8000 --reasoning-effort minimal"
    "_low|gpt-5|2.00|--max-tokens 10000 --reasoning-effort low"
    "_medium|o3-mini|1.50|--max-tokens 15000 --reasoning-effort medium"
    "_high|o3-mini|3.00|--max-tokens 25000 --reasoning-effort high"
    "_medium|o4-mini|1.50|--max-tokens 15000 --reasoning-effort medium"
    "_medium|o3|10.00|--max-tokens 20000 --reasoning-effort medium"
    "_high|o3|25.00|--max-tokens 35000 --reasoning-effort high"
)

SKIP_MODELS=("gpt-4o")  # already done

for job in "${JOBS[@]}"; do
    IFS='|' read -r SUFFIX MODEL BUDGET EXTRA <<< "$job"

    if [[ " ${SKIP_MODELS[*]} " == *" $MODEL "* ]]; then
        echo "--- SKIP $MODEL$SUFFIX (model in SKIP list) ---" | tee -a "$SUMMARY"
        continue
    fi

    OUT="metrics_openai_${MODEL}${SUFFIX}_benchmarks.json"
    LOG="/tmp/openai_eval_${MODEL}${SUFFIX}.log"

    if [ -s "$OUT" ]; then
        echo "--- SKIP $MODEL$SUFFIX (result file already exists: $OUT) ---" | tee -a "$SUMMARY"
        continue
    fi

    echo | tee -a "$SUMMARY"
    echo "=== $(date) [$MODEL$SUFFIX] start, budget=\$$BUDGET, args:$EXTRA ===" | tee -a "$SUMMARY"

    python eval_openai_benchmarks.py \
        --model "$MODEL" \
        --budget "$BUDGET" \
        --out "$OUT" \
        $EXTRA \
        2>&1 | tee "$LOG"
    EC=${PIPESTATUS[0]}

    echo "=== $(date) [$MODEL$SUFFIX] exit=$EC ===" | tee -a "$SUMMARY"

    if [ -s "$OUT" ]; then
        python3 -c "
import json
with open('$OUT') as f: d = json.load(f)
m = d['model'].replace('openai/', '')
re_eff = d.get('reasoning_effort') or '-'
cost = d.get('total_cost_usd', 0)
n = d.get('n_instances_evaluated', 0)
feas = sum(1 for r in d['results'] if r['feasible'])
parse = sum(1 for r in d['results'] if r['ops_emitted'] > 0)
print(f'[$MODEL$SUFFIX] cost=\${cost:.3f}  feasible={feas}/{n}  parseable={parse}/{n}  reasoning={re_eff}')
" 2>&1 | tee -a "$SUMMARY"
    fi
done

echo | tee -a "$SUMMARY"
echo "=== $(date) ALL DONE ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "================================================" | tee -a "$SUMMARY"
echo "Aggregate comparison:" | tee -a "$SUMMARY"
echo "================================================" | tee -a "$SUMMARY"
for f in metrics_openai_*_benchmarks.json; do
    [ ! -s "$f" ] && continue
    python3 -c "
import json
with open('$f') as fp: d = json.load(fp)
m = d['model'].replace('openai/', '')
re_eff = d.get('reasoning_effort') or '-'
cost = d.get('total_cost_usd', 0)
n = d.get('n_instances_evaluated', 0)
feas = sum(1 for r in d['results'] if r['feasible'])
parse = sum(1 for r in d['results'] if r['ops_emitted'] > 0)
feas_recs = [r for r in d['results'] if r['feasible'] and r.get('gap_pct') is not None]
mg = (sum(r['gap_pct'] for r in feas_recs) / len(feas_recs)) if feas_recs else None
print(f'  {m:18s} reason={re_eff:8s}  cost=\${cost:6.3f}  feasible={feas:>2}/{n}  parseable={parse:>2}/{n}  feas_mean_gap={mg:+.2f}%' if mg is not None else f'  {m:18s} reason={re_eff:8s}  cost=\${cost:6.3f}  feasible={feas:>2}/{n}  parseable={parse:>2}/{n}  feas_mean_gap=NA')
" 2>&1 | tee -a "$SUMMARY"
done
