#!/bin/bash
# Auto-resume watchdog for V4-rsLoRA experiment.
# Mirrors auto_resume_v7_v2.sh.

NOTIFY=~/.local/bin/notify
TMUX_BIN=/usr/bin/tmux
SESSION=grpo_v4_rslora
SCRIPT=/home/tio/Documents/Starjob/scripts/run_v4_rslora.sh
LOG=/home/tio/Documents/Starjob/grpo_jssp/runs/auto_resume_v4_rslora_watchdog.log
RUN_DIR=/home/tio/Documents/Starjob/grpo_jssp/runs/full_rslora_hybrid_n2000_v4

mkdir -p "$(dirname $LOG)"
exec >>"$LOG" 2>&1

date_iso() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(date_iso)] $*"; }

if $TMUX_BIN has-session -t "$SESSION" 2>/dev/null; then
    log "session '$SESSION' already alive, nothing to do"
    exit 0
fi

# Stand down if final_adapter + eval result both exist
if [ -f "$RUN_DIR/final_adapter/adapter_model.safetensors" ] && \
   [ -f "/home/tio/Documents/Starjob/grpo_jssp/eval_results/full_rslora_hybrid_n2000_v4_ood.json" ]; then
    log "training + eval already complete, watchdog standing down"
    exit 0
fi

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$DRIVER" ]; then
    log "ERROR: nvidia-smi failed, skipping launch"
    "$NOTIFY" grpo-v4-rslora-watchdog "auto-resume aborted: GPU not detected" 2>/dev/null || true
    exit 1
fi
log "GPU driver: $DRIVER"

rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
log "wiped Triton cache"

$TMUX_BIN new-session -d -s "$SESSION" "bash '$SCRIPT'"
RC=$?
if [ $RC -eq 0 ]; then
    log "launched tmux '$SESSION'"
    "$NOTIFY" grpo-v4-rslora-watchdog "auto-resume launched grpo_v4_rslora (driver $DRIVER)" 2>/dev/null || true
else
    log "ERROR: tmux new-session failed exit=$RC"
fi
exit $RC
