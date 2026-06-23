#!/usr/bin/env bash
#
# Setup kernel watchdog auto-reboot to prevent silent hangs.
#
# Why: 2026-06-05 18:46 freeze — NVIDIA driver (OOT_MODULE) triggered NULL
# pointer deref in scheduler. Kernel did NOT panic; instead RCU stalled and
# cascaded into hard lockup over 6 minutes (18:46:28 → 18:52:40). Machine
# was unreachable until hard-reset ~18 hours later.
#
# With panic_on_oops=1, the same bug would have panicked at 18:46:28 and
# auto-rebooted 10s later — total downtime ~1 minute instead of 18 hours.
#
# Idempotent: safe to re-run.

set -euo pipefail

DROPIN=/etc/sysctl.d/99-kernel-watchdog.conf

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo "$0" "$@"
fi

echo "=== Current values ==="
sysctl kernel.panic_on_oops kernel.softlockup_panic kernel.hung_task_panic \
       kernel.panic kernel.nmi_watchdog

echo
echo "=== Writing $DROPIN ==="
cat > "$DROPIN" <<'EOF'
# Auto-reboot on kernel fault — avoid silent multi-hour hangs.
# Tuned for ML workstation that runs unattended GPU training.
kernel.panic_on_oops    = 1   # any kernel oops → panic (catches OOT-module bugs early)
kernel.softlockup_panic = 1   # CPU stuck >22s in kernel mode → panic
kernel.hung_task_panic  = 1   # D-state task >120s → panic
kernel.panic            = 10  # wait 10s after panic, then reboot
kernel.nmi_watchdog     = 1   # hardware NMI-based stuck-CPU detector (usually default)
EOF
chmod 0644 "$DROPIN"
echo "Wrote:"
cat "$DROPIN" | sed 's/^/  /'

echo
echo "=== Applying ==="
sysctl --system | grep -E "kernel\.(panic|softlockup_panic|hung_task_panic|nmi_watchdog)" || true

echo
echo "=== Verifying ==="
sysctl kernel.panic_on_oops kernel.softlockup_panic kernel.hung_task_panic \
       kernel.panic kernel.nmi_watchdog

echo
echo "Done. Settings will persist across reboot."
echo "To revert: sudo rm $DROPIN && sudo sysctl --system"
