#!/bin/bash
# Resume chain from Stage 2 (canary 200) — Stage 1 already passed.
# Wipes Triton/Inductor caches before each stage to avoid back-to-back races
# (Stage 2 failed first attempt with double-free 2s after prior canary exited).

NOTIFY=~/.local/bin/notify
KILL=/tmp/STOP_GRPO
RUNS=/home/tio/Documents/Starjob/grpo_jssp/runs
TS=$(date +%Y%m%d_%H%M)
LOG=$RUNS/grpo_chain_s23_$TS.log

mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1

slog() {
    echo "=== $(date) [chain-s23] $* ==="
    "$NOTIFY" grpo-chain "$*" 2>/dev/null || true
}

check_kill() {
    if [[ -f "$KILL" ]]; then
        slog "ABORTED via kill switch ($KILL)"
        exit 99
    fi
}

slog "resumed chain start (PID=$$, stage 2+3 only)"
echo "Kill-switch: touch $KILL"

# Wipe caches + pause for GPU memory release
slog "wipe stale caches + 15s pause for GPU settle"
rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
sleep 15

# ---------- Stage 2: canary 200 ----------
slog "stage 2 RETRY: canary 200 step (~75 min)"
bash /home/tio/Documents/Starjob/scripts/run_canary.sh 200
EC=$?
if [[ $EC -ne 0 ]]; then
    slog "STAGE 2 FAIL exit=$EC — stopping. Try Plan B: downgrade kernel."
    exit 2
fi
slog "STAGE 2 OK — driver survives sustained load"
check_kill

# Wipe again before Stage 3 (longer chain, want fresh state)
slog "wipe caches + 20s pause before full pipeline"
rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
sleep 20

# ---------- Stage 3: full V1-V6 pipeline ----------
slog "stage 3: full V1-V6 pipeline (~30h, V5 SKIP)"
if [[ -d $RUNS/full_lora_stratified_2000_v1 ]] && [[ ! -d $RUNS/full_lora_stratified_2000_v1/final_adapter ]]; then
    BACKUP=$RUNS/full_lora_stratified_2000_v1_partial_$TS
    echo "Renaming partial V1 dir: $BACKUP"
    mv $RUNS/full_lora_stratified_2000_v1 "$BACKUP"
fi
bash /home/tio/Documents/Starjob/grpo_jssp/_pilot_evals_then_full_lora.sh
EC=$?
if [[ $EC -ne 0 ]]; then
    slog "STAGE 3 DONE with errors exit=$EC — see full_lora_all_summary.log"
else
    slog "STAGE 3 DONE OK"
fi
exit $EC
