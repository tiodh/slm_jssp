#!/usr/bin/env bash
#
# Swap NVIDIA driver 580 (current, unstable) → 575-server (Production Branch).
#
# Why: 580.126.09 caused kernel scheduler NULL deref on 2026-06-05 18:46 KST
# and 2026-05-16 09:31 KST. 575-server is NVIDIA's Production Branch — longer
# support window, fewer regressions, intended for data-center / sustained
# compute workloads.
#
# Requirements: console access (display manager will be stopped). Reboot
# is REQUIRED after this script. If you're SSH'ing in, the SSH session
# survives the driver swap but you must reboot manually.
#
# Safe to abort: until "apt remove" runs, nothing is destructive.

set -euo pipefail

TARGET_PKG="nvidia-driver-575-server"
TARGET_PKG_OPEN="nvidia-driver-575-server-open"
FALLBACK_PKG="nvidia-driver-535"  # last-resort: older LTS

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo "$0" "$@"
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "Pre-flight checks"

# 1. Apt cache fresh?
log "Updating apt cache"
apt update

# 2. Pick target package — prefer proprietary, fall back to open, then 535.
PICK=""
for cand in "$TARGET_PKG" "$TARGET_PKG_OPEN" "$FALLBACK_PKG"; do
    if apt-cache show "$cand" >/dev/null 2>&1; then
        PICK="$cand"
        echo "Found candidate: $cand"
        break
    else
        echo "Not available: $cand"
    fi
done
if [[ -z "$PICK" ]]; then
    echo "ERROR: no candidate driver available in apt. Add graphics-drivers PPA?"
    exit 1
fi
echo "→ Will install: $PICK"

# 3. Show what's currently installed for rollback reference.
log "Current NVIDIA packages (snapshot for rollback)"
dpkg -l | awk '/^ii.*nvidia-driver-[0-9]/ {print $2, $3}' | tee /tmp/nvidia-rollback-$(date +%Y%m%d-%H%M%S).txt

# 4. Confirm with user before destructive ops.
echo
echo "About to:"
echo "  - Stop display manager (gdm/lightdm/sddm — whichever is active)"
echo "  - Disable nvidia persistence mode"
echo "  - apt remove --autoremove '^nvidia-driver-580.*' '^libnvidia-.*-580.*'"
echo "  - apt install $PICK"
echo
read -rp "Proceed? [y/N] " ans
[[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 0; }

# 5. Stop display manager (whichever is active).
log "Stopping display manager"
for dm in gdm gdm3 lightdm sddm; do
    if systemctl is-active --quiet "$dm" 2>/dev/null; then
        echo "Stopping $dm"
        systemctl stop "$dm" || true
    fi
done

# 6. Kill any nvidia-smi / cuda processes.
log "Disabling persistence mode"
nvidia-smi -pm 0 2>/dev/null || true

# 7. Remove 580 driver.
log "Removing nvidia-driver-580 packages"
apt remove --autoremove -y '^nvidia-driver-580.*' '^libnvidia-.*-580.*' || {
    echo "ERROR: remove failed. System is in inconsistent state."
    echo "Try: sudo apt install -f"
    exit 1
}

# 8. Install target.
log "Installing $PICK"
apt install -y "$PICK"

# 9. Re-add persistence service hint (optional — most users prefer manual).
log "Driver swap complete"
dpkg -l | awk '/^ii.*nvidia-driver-[0-9]/ {print $2, $3}'

cat <<'EOF'

NEXT STEPS:
  1. Reboot:        sudo reboot
  2. After reboot:  nvidia-smi --query-gpu=driver_version,name --format=csv
                    sudo nvidia-smi -pm 1                      # re-enable persistence
  3. Verify torch:  python -c "import torch; print(torch.cuda.is_available())"

If GPU does not appear after reboot:
  - dmesg | grep -i nvidia
  - sudo dkms status
  - Rollback: sudo apt install nvidia-driver-580-open

EOF
