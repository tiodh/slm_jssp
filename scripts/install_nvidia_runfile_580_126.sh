#!/usr/bin/env bash
#
# Install NVIDIA driver 580.126.09 from the .run installer.
#
# Why: apt removed all nvidia-580 packages (status "rc" = config-only).
# Reinstalling via apt would land us on 580.159.03 again, which kernel-panics.
# The .run installer lets us pin to the exact last-known-good version,
# bypassing distro repos entirely.
#
# Uses --dkms so the module rebuilds automatically on future kernel updates
# (kernel 6.17.0-20-generic is current; -22 and -35 are queued — DKMS will
# rebuild for any of them).
#
# IMPORTANT: this MUST be run from a console (Ctrl+Alt+F3) OR with display
# manager stopped. Running with X/Wayland active will abort.

set -uo pipefail

RUNFILE=/home/tio/Documents/Starjob/NVIDIA-Linux-x86_64-580.126.09.run
LOG=/var/log/nvidia-installer.log

# Auto-wrap inside tmux so SSH disconnect cannot kill the install.
# Previous attempt died at 61% when SSH dropped during kernel module build.
if [[ -z "${TMUX:-}" ]] && [[ "${1:-}" != "--inner" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed. Install with: sudo apt install -y tmux"
        echo "Or run from a physical console where SSH disconnect is not a concern."
        exit 1
    fi
    SESS="nvinstall_$(date +%H%M%S)"
    echo "============================================================"
    echo "Launching inside tmux session: $SESS"
    echo "  Attach (from another shell):  tmux attach -t $SESS"
    echo "  Detach while attached:        Ctrl+B then D"
    echo "  Install survives SSH drops while detached."
    echo "============================================================"
    sleep 2
    exec tmux new-session -s "$SESS" "bash '$0' --inner"
fi

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo "$0" --inner
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "Pre-flight"

if [[ ! -f "$RUNFILE" ]]; then
    echo "ERROR: $RUNFILE not found."
    exit 1
fi

echo "Runfile size:    $(du -h $RUNFILE | awk '{print $1}')"
echo "Kernel:          $(uname -r)"
echo "GCC:             $(gcc --version | head -1)"
echo "Headers dir:     $(ls -d /lib/modules/$(uname -r)/build 2>/dev/null || echo MISSING)"

# Required prerequisites
log "Verifying build prerequisites"
MISSING_PKGS=()
for pkg in build-essential dkms "linux-headers-$(uname -r)" pkg-config libglvnd-dev; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "Missing: ${MISSING_PKGS[*]}"
    echo "Installing now..."
    apt update
    apt install -y "${MISSING_PKGS[@]}" || { echo "ERROR: prereq install failed"; exit 2; }
else
    echo "All prerequisites present."
fi

log "Checking display manager state"
DM_TO_RESTART=""
for dm in gdm gdm3 lightdm sddm; do
    if systemctl is-active --quiet "$dm" 2>/dev/null; then
        echo "  ACTIVE: $dm — will stop before install, restart after"
        DM_TO_RESTART="$dm"
    fi
done

if [[ -n "$DM_TO_RESTART" ]]; then
    echo
    echo "WARNING: display manager is running. Stopping it will log you out of GUI."
    echo "         This SSH session will survive. After install, GUI restarts automatically."
    echo
fi

log "Plan"
echo "  Target driver:   580.126.09"
echo "  Installer flags: --dkms --no-questions --ui=none --no-x-check"
echo "  Log will go to:  $LOG"
echo "  Restart DM:      ${DM_TO_RESTART:-none}"
echo
read -rp "Proceed? [y/N] " ans
[[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 0; }

# Truncate the empty existing log so progress is visible
: > "$LOG"

if [[ -n "$DM_TO_RESTART" ]]; then
    log "Stopping display manager: $DM_TO_RESTART"
    systemctl stop "$DM_TO_RESTART" || true
    sleep 2
fi

# Just in case any residual nvidia process pinned the driver from prior boot
log "Unloading any residual nvidia modules"
for mod in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
    if lsmod | grep -q "^$mod "; then
        rmmod "$mod" 2>/dev/null && echo "  unloaded: $mod" || echo "  could not unload (probably not loaded): $mod"
    fi
done

# Clean up partial state from previously aborted install (e.g. SSH disconnect mid-build).
log "Cleaning up any partial state from previous aborted .run install"
PARTIAL_SIGNS=0
if ls -d /usr/src/nvidia-* >/dev/null 2>&1; then
    echo "  Found /usr/src/nvidia-* source dirs from previous attempt"
    PARTIAL_SIGNS=1
fi
if dkms status 2>/dev/null | grep -qi nvidia; then
    echo "  Found nvidia entries in dkms"
    PARTIAL_SIGNS=1
fi
if [[ -d /var/lib/dkms/nvidia ]]; then
    echo "  Found /var/lib/dkms/nvidia"
    PARTIAL_SIGNS=1
fi

if [[ $PARTIAL_SIGNS -eq 1 ]]; then
    echo "  Running '$RUNFILE --uninstall' to clear partial state..."
    bash "$RUNFILE" --uninstall --silent 2>&1 | tail -10 || \
        echo "  (uninstall returned non-zero — may be fine if partial state was minimal)"
    # Also clean any orphaned dkms entries
    for entry in $(dkms status 2>/dev/null | awk -F'[,/ ]' '/nvidia/ {print $1"/"$2}' | sort -u); do
        echo "  dkms remove: $entry"
        dkms remove "$entry" --all 2>/dev/null || true
    done
else
    echo "  No partial state detected — proceeding clean."
fi

log "Running .run installer (non-interactive, with DKMS)"
# --dkms          register with dkms so kernel updates auto-rebuild
# --no-questions  accept all defaults
# --ui=none       no curses UI, plain stdout
# --no-x-check    don't sanity-check X (we already stopped DM)
# --no-nouveau-check  skip nouveau check (it should already be blacklisted)
bash "$RUNFILE" --dkms --no-questions --ui=none --no-x-check --no-nouveau-check
INSTALL_EC=$?

log "Installer exit code: $INSTALL_EC"
echo "Last 30 lines of $LOG:"
tail -30 "$LOG" 2>/dev/null

if [[ $INSTALL_EC -ne 0 ]]; then
    echo
    echo "ERROR: installer failed. Possible causes:"
    echo "  - nouveau still loaded (lsmod | grep nouveau)"
    echo "  - kernel headers mismatch (compare uname -r vs ls /lib/modules)"
    echo "  - dkms build failure (look at $LOG)"
    echo
    if [[ -n "$DM_TO_RESTART" ]]; then
        echo "Restarting $DM_TO_RESTART anyway so you can use the GUI..."
        systemctl start "$DM_TO_RESTART" 2>/dev/null || true
    fi
    exit 3
fi

log "Verifying install"
echo "DKMS status:"
dkms status 2>/dev/null | grep -i nvidia || echo "  (no dkms entries — investigate)"
echo
echo "Module files in /lib/modules/$(uname -r):"
find /lib/modules/$(uname -r) -name "nvidia*.ko*" 2>/dev/null | head -10
echo
echo "Loading nvidia module..."
if modprobe nvidia 2>&1; then
    echo "  OK: nvidia module loaded"
else
    echo "  WARN: modprobe failed — may need reboot"
fi
echo
echo "nvidia-smi output (or modprobe failure):"
nvidia-smi --query-gpu=driver_version,name --format=csv 2>&1

if [[ -n "$DM_TO_RESTART" ]]; then
    log "Restarting display manager: $DM_TO_RESTART"
    systemctl start "$DM_TO_RESTART" 2>/dev/null || true
fi

cat <<'EOF'

================================================================
NEXT STEPS:

  1. Reboot (safest, ensures clean module load):
       sudo reboot

  2. After reboot, confirm:
       nvidia-smi --query-gpu=driver_version,name --format=csv
       # Expect: 580.126.09, NVIDIA GeForce RTX 4090

  3. Wipe Triton/Inductor caches (driver changed):
       rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache

  4. Re-enable persistence mode:
       sudo nvidia-smi -pm 1

  5. Block apt from re-installing 580.159 ever again:
       sudo apt-mark hold nvidia-driver-580 || true
       # (apt-mark will fail since the package isn't installed via apt anymore;
       # that's actually fine — there is nothing for apt to upgrade.)

  6. Verify PyTorch:
       cd /home/tio/Documents/Starjob
       source venv-grpo/bin/activate
       python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  7. Run a canary (20 steps) before the full pipeline.

To uninstall this driver later:
  sudo bash NVIDIA-Linux-x86_64-580.126.09.run --uninstall

================================================================
EOF
