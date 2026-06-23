#!/usr/bin/env bash
#
# Install nvidia-driver-580-open from apt — uses Ubuntu's PREBUILT kernel
# module (no source compile = no risk of kernel panic during install).
#
# Why: .run installer for 580.126.09 panics the kernel during module build
# against 6.17 headers. Apt's -open variant ships precompiled .ko matched to
# the running kernel (linux-modules-nvidia-580-open-6.17.0-20-generic).
#
# Tradeoff: only 580.159.03 is available in apt now. If sustained training
# kernel-panics again with this version, plan B is to downgrade the KERNEL
# to where 580.126.09 worked (May 23 V5 success).

set -uo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo "$0" "$@"
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "Pre-flight"
echo "Kernel:            $(uname -r)"
echo "Current nvidia:    $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo NONE)"
echo "Residual rc pkgs:  $(dpkg -l 2>/dev/null | grep -c '^rc.*nvidia')"

log "Apt update"
apt update

log "Purging any residual rc nvidia packages"
RC_PKGS=$(dpkg -l 2>/dev/null | awk '/^rc.*nvidia/ {print $2}' | tr '\n' ' ')
if [[ -n "$RC_PKGS" ]]; then
    echo "Purging: $RC_PKGS"
    apt purge -y $RC_PKGS || true
else
    echo "No rc packages to purge."
fi

log "Install nvidia-driver-580-open (with prebuilt module)"
apt install -y nvidia-driver-580-open
EC=$?

if [[ $EC -ne 0 ]]; then
    echo "ERROR: apt install failed, ec=$EC"
    exit $EC
fi

log "Verify install"
echo "Installed nvidia packages:"
dpkg -l 2>/dev/null | awk '/^ii.*nvidia/ {printf "  %-45s %s\n", $2, $3}'
echo
echo "Prebuilt module file:"
find /lib/modules/$(uname -r) -name "nvidia*.ko*" 2>/dev/null | head -10

log "Loading nvidia module (no reboot needed if successful)"
if modprobe nvidia 2>&1; then
    echo "  OK: nvidia module loaded without reboot"
    sleep 2
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv 2>&1 | head -5
else
    echo "  WARN: modprobe failed — reboot required"
fi

cat <<'EOF'

================================================================
INSTALL DONE. NEXT STEPS:

  1. Reboot (recommended for clean module load):
       sudo reboot

  2. After reboot, verify:
       nvidia-smi
       # Expect: driver_version = 580.159.03

  3. Wipe Triton cache (driver loaded fresh):
       rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache

  4. Re-enable persistence mode:
       sudo nvidia-smi -pm 1

  5. Run a SHORT canary (20 steps) — confirms basic GPU function:
       cd /home/tio/Documents/Starjob
       bash /tmp/run_canary.sh  # if file from before still exists

  6. If 20-step canary passes, run a LONGER canary (200 steps):
       Same canary script but edit max-steps=200. This proves the
       driver is stable for ~70+ minutes of sustained GRPO load —
       longer than the 24-min duration that crashed before.

  7. If 200-step canary passes, relaunch full pipeline.

If kernel panic happens AGAIN during training:
  - That confirms 580.159.03 is fundamentally bad for our workload.
  - Plan B: downgrade kernel to 6.16 (or whatever the May 23 success
    was running), then install 580.126.09 via .run on the older
    kernel (where the source compiles cleanly).
================================================================
EOF
