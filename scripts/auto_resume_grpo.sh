#!/bin/bash
# Auto-resume watchdog: launches the V1-V6 training pipeline if it's not already running.
# Run on every boot via cron @reboot, OR manually after a crash.
#
# The main script is idempotent:
#   - skips versions with existing final_adapter
#   - resumes each version from latest checkpoint
#   - up to 5 process-level retries per version
# So this wrapper just ensures the supervisor process exists.

NOTIFY=~/.local/bin/notify
TMUX_BIN=/usr/bin/tmux
SESSION=grpo_full
SCRIPT=/home/tio/Documents/Starjob/scripts/run_v1_v2_v3_v4_v6_full.sh
LOG=/home/tio/Documents/Starjob/grpo_jssp/runs/auto_resume_watchdog.log

mkdir -p "$(dirname $LOG)"
exec >>"$LOG" 2>&1

date_iso() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(date_iso)] $*"; }

# If session already exists, do nothing
if $TMUX_BIN has-session -t "$SESSION" 2>/dev/null; then
    log "session '$SESSION' already alive, nothing to do"
    exit 0
fi

# Pre-flight: GPU & driver sanity
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$DRIVER" ]; then
    log "ERROR: nvidia-smi failed, GPU not available; skipping launch"
    "$NOTIFY" grpo-watchdog "auto-resume aborted: GPU not detected" 2>/dev/null || true
    exit 1
fi
log "GPU driver: $DRIVER"

# Wipe stale Triton cache (driver may have changed across reboot)
rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache 2>/dev/null
log "wiped Triton cache"

# Launch in new tmux session
$TMUX_BIN new-session -d -s "$SESSION" "bash '$SCRIPT'"
RC=$?
if [ $RC -eq 0 ]; then
    log "launched tmux '$SESSION'"
    "$NOTIFY" grpo-watchdog "auto-resume launched grpo_full session (driver $DRIVER)" 2>/dev/null || true
else
    log "ERROR: tmux new-session failed exit=$RC"
    "$NOTIFY" grpo-watchdog "auto-resume FAILED to launch tmux exit=$RC" 2>/dev/null || true
fi
exit $RC
