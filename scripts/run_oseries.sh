#!/bin/bash
# o-series OOD eval: o3-mini medium, o3-mini high, o3 medium.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source ~/.openai_env

SUMMARY=/home/tio/Documents/Starjob/openai_oseries_summary.log
echo "=== $(date) START o-series OOD eval ===" | tee "$SUMMARY"

declare -a JOBS=(
    "_medium|o3-mini|1.50|--max-tokens 15000 --reasoning-effort medium"
    "_high|o3-mini|4.00|--max-tokens 25000 --reasoning-effort high"
    "_medium|o3|12.00|--max-tokens 20000 --reasoning-effort medium"
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
echo "=== $(date) o-series ALL DONE ===" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"
echo "================================================" | tee -a "$SUMMARY"
echo "Final comparison (incl. previously-done models):" | tee -a "$SUMMARY"
echo "================================================" | tee -a "$SUMMARY"
for f in metrics_openai_*_benchmarks.json; do
    [ ! -s "$f" ] && continue
    [[ "$f" == *.BROKEN* ]] && continue
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
mg_s = f'{mg:+.2f}%' if mg is not None else 'NA'
print(f'  {m:18s} reason={re_eff:8s}  cost=\${cost:6.3f}  feas={feas:>2}/{n}  parse={parse:>2}/{n}  feas_mean_gap={mg_s}')
" 2>&1 | tee -a "$SUMMARY"
done
