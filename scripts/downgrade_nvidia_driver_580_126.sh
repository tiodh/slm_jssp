#!/usr/bin/env bash
#
# Downgrade NVIDIA driver 580.159.03 -> 580.126.09 (last known stable for our workload).
#
# Context: 580.159.03 kernel-panicked twice under prolonged RTX 4090 GRPO
# training (2026-06-05 18:46 KST, 2026-06-07 ~11:30 KST). Driver 580.126.09
# ran V5 to completion on 2026-05-23.
#
# Strategy: stay on 580 branch (simpler than 575 swap), pin to old patch,
# apt-mark hold all 580 packages so unattended-upgrades cannot move us forward.
#
# Reboot is REQUIRED after this script.

set -uo pipefail   # NO -e: we want to continue on non-fatal errors so partial
                    # state can be inspected.

TARGET_VER="580.126.09"

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo "$0" "$@"
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "Pre-flight"
echo "Kernel: $(uname -r)"
echo "nvidia module loaded? $(lsmod | grep -c '^nvidia ') (0 = not loaded)"
echo
echo "Currently installed nvidia-580 packages:"
dpkg -l 2>/dev/null | awk '/^ii.*(nvidia|libnvidia)-.*-580/ {printf "  %-45s %s\n", $2, $3}'

log "Updating apt cache"
apt update

log "Searching apt for $TARGET_VER"
AVAIL=$(apt-cache madison nvidia-driver-580 2>/dev/null | grep -F "$TARGET_VER" | head -1)
if [[ -z "$AVAIL" ]]; then
    echo "ERROR: $TARGET_VER not available in apt cache for nvidia-driver-580."
    echo
    echo "Available versions in apt:"
    apt-cache madison nvidia-driver-580 2>/dev/null | head -10
    echo
    echo "Fallback options if older patch is gone from apt:"
    echo "  A) Download .run installer:"
    echo "     wget https://download.nvidia.com/XFree86/Linux-x86_64/$TARGET_VER/NVIDIA-Linux-x86_64-$TARGET_VER.run"
    echo "     sudo apt purge -y '^nvidia-.*-580' '^libnvidia-.*-580'"
    echo "     sudo bash NVIDIA-Linux-x86_64-$TARGET_VER.run --dkms"
    echo "  B) Swap to nvidia-driver-575-server (production branch)."
    echo
    echo "No changes made. Exiting."
    exit 2
fi

FULL_VER=$(echo "$AVAIL" | awk '{print $3}')
echo "Resolved package version: nvidia-driver-580 = $FULL_VER"

log "Discovering installed 580 packages"
mapfile -t INSTALLED < <(dpkg-query -W -f='${binary:Package}\n' 2>/dev/null \
    | grep -E '^(nvidia|libnvidia|xserver-xorg-video-nvidia)-.*-580(-[a-z]+)?$' \
    | sort -u)

if [[ ${#INSTALLED[@]} -eq 0 ]]; then
    echo "ERROR: no nvidia-580 packages installed. Did the previous failed swap remove them?"
    echo "Run: dpkg -l | grep nvidia"
    exit 3
fi

echo "Found ${#INSTALLED[@]} packages to downgrade:"
INSTALL_LIST=()
for pkg in "${INSTALLED[@]}"; do
    echo "  $pkg"
    INSTALL_LIST+=("$pkg=$FULL_VER")
done

log "Snapshot rollback info"
SNAP=/tmp/nvidia-downgrade-rollback-$(date +%Y%m%d-%H%M%S).txt
dpkg -l 2>/dev/null | awk '/^ii.*(nvidia|libnvidia)/ {print $2, $3}' | tee "$SNAP"
echo
echo "Rollback snapshot: $SNAP"

log "Plan summary"
echo "  Target version:  $FULL_VER"
echo "  Packages:        ${#INSTALLED[@]}"
echo "  Will hold after: yes (apt-mark hold)"
echo "  Reboot needed:   yes"
echo
read -rp "Proceed? [y/N] " ans
[[ "${ans,,}" == "y" ]] || { echo "Aborted, no changes."; exit 0; }

log "Stop display manager (if running)"
for dm in gdm gdm3 lightdm sddm; do
    if systemctl is-active --quiet "$dm" 2>/dev/null; then
        echo "  stop $dm"
        systemctl stop "$dm" || true
    fi
done

log "Unhold any existing nvidia holds (in case)"
apt-mark unhold "${INSTALLED[@]}" 2>/dev/null || true

log "apt install --allow-downgrades"
if ! apt install -y --allow-downgrades "${INSTALL_LIST[@]}"; then
    echo
    echo "ERROR: install failed. State may be inconsistent."
    echo "  Diagnose with:  sudo apt install -f"
    echo "  Rollback list:  $SNAP"
    echo
    echo "If apt complains about unmet deps, often:"
    echo "  - linux-headers missing -> apt install linux-headers-\$(uname -r)"
    echo "  - dkms can't build      -> apt install --reinstall nvidia-dkms-580=$FULL_VER"
    exit 4
fi

log "Pin packages to prevent auto-upgrade"
apt-mark hold "${INSTALLED[@]}" 2>/dev/null
echo "Held:"
apt-mark showhold | grep -i nvidia

log "Verify dkms built"
dkms status 2>/dev/null | grep -i nvidia || echo "(no dkms entries — may use prebuilt module instead)"

log "Final installed versions"
dpkg -l 2>/dev/null | awk '/^ii.*(nvidia|libnvidia)-.*-580/ {printf "  %-45s %s\n", $2, $3}'

cat <<EOF

================================================================
NEXT STEPS — DO THESE IN ORDER:

  1. Reboot:
       sudo reboot

  2. After reboot, verify:
       nvidia-smi --query-gpu=driver_version,name --format=csv
       # Should print: 580.126.09, NVIDIA GeForce RTX 4090

  3. Wipe Triton/Inductor caches (driver version changed):
       rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache

  4. Re-enable persistence mode (reduces driver init churn):
       sudo nvidia-smi -pm 1

  5. Verify PyTorch sees the GPU:
       cd /home/tio/Documents/Starjob
       source venv-grpo/bin/activate
       python -c "import torch; print('cuda=', torch.cuda.is_available(), 'name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

  6. Run canary before full pipeline:
       cd /home/tio/Documents/Starjob
       bash /tmp/run_canary.sh    # (still in /tmp from previous canary)

  7. If canary passes, relaunch the full pipeline.

To later allow upgrades again (NOT recommended unless NVIDIA ships a fix):
  sudo apt-mark unhold \$(apt-mark showhold | grep -i nvidia)

Rollback file (kept for reference): $SNAP
================================================================
EOF
