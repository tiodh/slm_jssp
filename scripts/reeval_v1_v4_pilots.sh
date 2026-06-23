#!/bin/bash
# Re-eval V1-V4 pilot adapters with post-patch strict checker.
# Each eval ~3-5 min for 18 OOD instances.

cd /home/tio/Documents/Starjob
source venv-grpo/bin/activate
source /home/tio/Documents/Starjob/grpo_jssp/_cuda_env.sh

SUMMARY=/home/tio/Documents/Starjob/v1_v4_reeval_summary.log
echo "=== $(date) START V1-V4 pilot re-eval (strict checker) ===" | tee "$SUMMARY"

declare -a JOBS=(
    "v1_pilot|pilot_lora_v1_stratified_n100"
    "v2_pilot|pilot_lora_v2_stratified_v2_n100"
    "v3_pilot|pilot_lora_v3_stratified_n25"
    "v4_pilot|pilot_lora_v4_hybrid_n25"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r LABEL RUN_DIR <<< "$job"
    ADAPTER="grpo_jssp/runs/$RUN_DIR/final_adapter"
    OUT_PREFIX="grpo_jssp/eval_results/strict_${LABEL}"

    if [[ -f "${OUT_PREFIX}_ood.json" ]]; then
        echo "--- SKIP $LABEL (exists) ---" | tee -a "$SUMMARY"
        continue
    fi

    if [[ ! -d "$ADAPTER" && ! -L "$ADAPTER" ]]; then
        echo "--- SKIP $LABEL ($ADAPTER not found) ---" | tee -a "$SUMMARY"
        continue
    fi

    echo | tee -a "$SUMMARY"
    echo "=== $(date) [$LABEL] start ===" | tee -a "$SUMMARY"
    echo "  adapter: $ADAPTER" | tee -a "$SUMMARY"

    # OOD only (skip SM to save time)
    python -u -c "
from pathlib import Path
import json
import sys
sys.path.insert(0, '/home/tio/Documents/Starjob')
import unsloth  # noqa
from grpo_jssp.evaluate import eval_ood
o = eval_ood(Path('$ADAPTER'), out_path=Path('${OUT_PREFIX}_ood.json'))
print(json.dumps(o['summary'], indent=2))
" 2>&1 | tee -a "$SUMMARY"

    if [[ -f "${OUT_PREFIX}_ood.json" ]]; then
        python3 -c "
import json
with open('${OUT_PREFIX}_ood.json') as f: d = json.load(f)
s = d['summary']
print(f'[$LABEL] strict_feas={s[\"n_feasible\"]}/18  mean_gap={s[\"mean_gap_to_bks\"]*100:.2f}%')
" 2>&1 | tee -a "$SUMMARY"
    fi
done

echo | tee -a "$SUMMARY"
echo "=== $(date) ALL DONE ===" | tee -a "$SUMMARY"
