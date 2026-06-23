#!/bin/bash
# Three-stage auto-escalation chain:
#   1. wait/verify canary 20 step (existing in tmux 'canary' or launch fresh)
#   2. canary 200 step (~75 min) — proves driver survives sustained load
#   3. full V1-V6 pipeline (~30h, V5 skip from May 23 final_adapter)
#
# Stops at first failure. Notifies Slack at every transition.
# Kill-switch: `touch /tmp/STOP_GRPO` halts the chain at next checkpoint.
#
# NOTE: chain does NOT auto-resume across reboots. If kernel watchdog fires
# during a long stage, user must re-launch on return.

NOTIFY=~/.local/bin/notify
KILL=/tmp/STOP_GRPO
RUNS=/home/tio/Documents/Starjob/grpo_jssp/runs
TS=$(date +%Y%m%d_%H%M)
CHAIN_LOG=$RUNS/grpo_auto_chain_$TS.log

mkdir -p "$RUNS"
exec > >(tee -a "$CHAIN_LOG") 2>&1

slog() {
    echo "=== $(date) [chain] $* ==="
    "$NOTIFY" grpo-chain "$*" 2>/dev/null || true
}

check_kill() {
    if [[ -f "$KILL" ]]; then
        slog "ABORTED via kill switch ($KILL)"
        exit 99
    fi
}

slog "chain start (PID=$$), log=$CHAIN_LOG"
echo "Kill-switch: touch $KILL to halt at next stage boundary."

# ---------- Stage 1: canary 20 (poll existing or launch fresh) ----------
slog "stage 1: canary 20 step"

CAN20=$(ls -t $RUNS/canary_v1_steps20_*_top.log 2>/dev/null | head -1)
if [[ -z "$CAN20" ]] || grep -q "CANARY end exit=" "$CAN20" 2>/dev/null; then
    # No active canary OR last one already finished — only launch if NO good result exists
    LAST_GOOD=$(grep -l "CANARY end exit=0" $RUNS/canary_v1_steps20_*_top.log 2>/dev/null | tail -1)
    if [[ -z "$LAST_GOOD" ]]; then
        slog "no in-progress canary 20 and no recent success — launching"
        bash /home/tio/Documents/Starjob/scripts/run_canary.sh 20 > /tmp/canary20_chain_$TS.log 2>&1 &
        sleep 6
        CAN20=$(ls -t $RUNS/canary_v1_steps20_*_top.log 2>/dev/null | head -1)
    else
        echo "Reusing recent successful canary 20: $LAST_GOOD"
        CAN20=$LAST_GOOD
    fi
fi

echo "Polling: $CAN20"
WAIT_START=$(date +%s)
while ! grep -q "CANARY end exit=" "$CAN20" 2>/dev/null; do
    check_kill
    sleep 30
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if (( ELAPSED > 1800 )); then
        slog "STAGE 1 TIMEOUT (30 min, expected ~7 min) — stopping"
        exit 1
    fi
done

EC=$(grep -oE "CANARY end exit=[0-9]+" "$CAN20" | tail -1 | awk -F= '{print $2}')
if [[ "$EC" != "0" ]]; then
    slog "STAGE 1 FAIL exit=$EC log=$CAN20 — stopping"
    tail -10 "$CAN20" 2>/dev/null
    exit 1
fi
slog "STAGE 1 OK"
check_kill

# ---------- Stage 2: canary 200 ----------
slog "stage 2: canary 200 step (~75 min, threshold > 24-min crash point)"
bash /home/tio/Documents/Starjob/scripts/run_canary.sh 200
EC=$?
if [[ $EC -ne 0 ]]; then
    slog "STAGE 2 FAIL exit=$EC — stopping. Plan B: downgrade kernel to 6.16."
    exit 2
fi
slog "STAGE 2 OK — driver survived sustained load"
check_kill

# ---------- Stage 3: full V1-V6 pipeline ----------
slog "stage 3: full V1-V6 pipeline (~30h, V5 SKIP)"
# Rename any partial V1 dir from prior crashed attempt
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
    slog "STAGE 3 DONE OK — full pipeline complete"
fi
exit $EC
