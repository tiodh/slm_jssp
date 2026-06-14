#!/bin/bash
# Auto-resume watchdog for V7+V2 chain session.
# Mirrors auto_resume_grpo.sh but targets the V7+V2 chain tmux session.

NOTIFY=~/.local/bin/notify
TMUX_BIN=/usr/bin/tmux
SESSION=grpo_v7_v2
SCRIPT=/home/tio/Documents/Starjob/scripts/run_v7_then_v2.sh
LOG=/home/tio/Documents/Starjob/grpo_jssp/runs/auto_resume_v7_v2_watchdog.log

mkdir -p "$(dirname $LOG)"
exec >>"$LOG" 2>&1

date_iso() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(date_iso)] $*"; }

if $TMUX_BIN has-session -t "$SESSION" 2>/dev/null; then
    log "session '$SESSION' already alive, nothing to do"
    exit 0
fi

# Skip if both V7 final_adapter AND V2 final_adapter exist (chain truly done)
V7_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs/full_lora_hybrid_lc_over_n2000_v7
V2_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs/full_lora_stratified_v2_2000_v2
if [ -f "$V7_DIR/final_adapter/adapter_model.safetensors" ] && [ -f "$V2_DIR/final_adapter/adapter_model.safetensors" ]; then
    log "both V7 and V2 final_adapter present, chain complete, watchdog standing down"
    exit 0
fi

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$DRIVER" ]; then
    log "ERROR: nvidia-smi failed, skipping launch"
    "$NOTIFY" grpo-v7v2-watchdog "auto-resume aborted: GPU not detected" 2>/dev/null || true
    exit 1
fi
log "GPU driver: $DRIVER"

rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
log "wiped Triton cache"

$TMUX_BIN new-session -d -s "$SESSION" "bash '$SCRIPT'"
RC=$?
if [ $RC -eq 0 ]; then
    log "launched tmux '$SESSION'"
    "$NOTIFY" grpo-v7v2-watchdog "auto-resume launched grpo_v7_v2 session (driver $DRIVER)" 2>/dev/null || true
else
    log "ERROR: tmux new-session failed exit=$RC"
fi
exit $RC
