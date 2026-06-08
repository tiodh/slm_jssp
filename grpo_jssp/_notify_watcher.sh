#!/bin/bash
# Tail a GRPO training log and send notify for each step metric, errors, and completion.
# Usage: ./_notify_watcher.sh <training.log> <label>
# Stop by killing this process.

LOG="$1"
LABEL="${2:-grpo}"
NOTIFY=~/.local/bin/notify
NOTIFY_EVERY="${NOTIFY_EVERY:-1}"  # send 1-in-N metric dicts (1 = every dict, 10 = every 10th = every 50 steps when logging_steps=5)

if [ -z "$LOG" ]; then
  echo "usage: $0 <log_path> [label]" >&2
  exit 1
fi

# wait until log file exists (up to 60s)
for i in $(seq 1 60); do
  [ -f "$LOG" ] && break
  sleep 1
done

# tail -F follows even if file is rotated; tr '\r' '\n' splits tqdm-mixed lines
tail -F "$LOG" 2>/dev/null | tr '\r' '\n' | while IFS= read -r line; do
  [ -z "$line" ] && continue

  # tqdm prints "NNN/MMM [elapsed<remaining]" on its own line (tr split it off
  # the 'loss': dict), so track the latest step seen and carry it forward.
  s=$(printf '%s' "$line" | grep -oP '[0-9]+/[0-9]+(?= \[)' | head -1)
  [ -n "$s" ] && last_step="$s"

  # metric dict line: contains 'loss': prefix (unique to TRL step dicts)
  if [[ "$line" == *"'loss':"* ]]; then
    step="${last_step:-?}"
    reward=$(printf '%s' "$line"   | grep -oP "'reward':\s*\K[\-0-9.e]+"             | head -1)
    rstd=$(printf '%s' "$line"     | grep -oP "'reward_std':\s*\K[\-0-9.e]+"         | head -1)
    clen=$(printf '%s' "$line"     | grep -oP "'completion_length':\s*\K[\-0-9.e]+"  | head -1)
    grad=$(printf '%s' "$line"     | grep -oP "'grad_norm':\s*\K[\-0-9.e]+"          | head -1)
    kl=$(printf '%s' "$line"       | grep -oP "'kl':\s*\K[\-0-9.e]+"                 | head -1)

    # gate by step modulo when NOTIFY_EVERY > 1; otherwise notify on every dict
    step_num="${step%%/*}"
    if [[ "$NOTIFY_EVERY" -le 1 ]] || ( [[ -n "$step_num" ]] && (( step_num % NOTIFY_EVERY == 0 )) ); then
        "$NOTIFY" "$LABEL" "step=${step:-?} r=${reward} std=${rstd} clen=${clen} grad=${grad} kl=${kl}" >/dev/null 2>&1
    fi

    # inline collapse detection: reward_std==0 AND completion_length>=4090
    if [[ "$rstd" == "0.0" || "$rstd" == "0" ]]; then
      clen_int=${clen%%.*}
      if [[ -n "$clen_int" && "$clen_int" -ge 4090 ]] 2>/dev/null; then
        "$NOTIFY" warn "collapse signal @step=${step:-?}: reward_std=0 clen=${clen}" >/dev/null 2>&1
      fi
    fi
    continue
  fi

  # phase markers + errors
  case "$line" in
    *"reward_mode="*)         "$NOTIFY" "$LABEL" "training started" >/dev/null 2>&1 ;;
    *"saved final adapter"*)  "$NOTIFY" "$LABEL" "adapter saved" >/dev/null 2>&1 ;;
    *"[eval] DONE"*)          "$NOTIFY" "$LABEL" "eval done" >/dev/null 2>&1 ;;
    *"feasibility_rate"*)
      rate=$(printf '%s' "$line" | grep -oP '[0-9.]+' | head -1)
      "$NOTIFY" "$LABEL" "feasibility_rate=${rate}" >/dev/null 2>&1 ;;
    *Traceback*|*"Killed"*|*"OutOfMemory"*|*"FAILED"*|*"CUDA error"*)
      "$NOTIFY" warn "training error: ${line:0:160}" >/dev/null 2>&1 ;;
  esac
done
