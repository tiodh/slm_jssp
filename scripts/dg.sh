#!/bin/bash
# Downgrade nvidia-driver-580-open: 580.159.03 -> 580.126.09
# v2: pin ALL deps explicitly (graphics-drivers PPA was polluting version resolution).
#
# USAGE: bash /home/tio/Documents/Starjob/scripts/dg.sh

set -u

if [[ $EUID -ne 0 ]]; then
    echo "needs sudo, re-running..."
    exec sudo "$0" "$@"
fi

LOG=/tmp/dg-install.log
echo "===== started $(date) =====" | tee "$LOG"

V="580.126.09-0ubuntu0.24.04.2"
KV="6.17.0-20.20~24.04.1"

echo "STEP 1: stop display manager"
systemctl stop gdm3 2>/dev/null || systemctl stop gdm 2>/dev/null || true
sleep 2

echo "STEP 2: unhold any held nvidia packages"
HELD=$(apt-mark showhold 2>/dev/null | grep -iE "nvidia|libnvidia" || true)
if [[ -n "$HELD" ]]; then
    apt-mark unhold $HELD
else
    echo "  no holds"
fi

echo "STEP 3: apt install from snapshot 20260415 with ALL packages version-pinned"
echo "  (otherwise graphics-drivers PPA pulls in 580.159 for libnvidia-*)"
echo "  Output -> $LOG"
echo

apt install \
    --update \
    --snapshot 20260415T000000Z \
    --allow-downgrades \
    -y \
    nvidia-driver-580-open=$V \
    linux-modules-nvidia-580-open-6.17.0-20-generic=$KV \
    nvidia-dkms-580-open=$V \
    nvidia-kernel-source-580-open=$V \
    nvidia-kernel-common-580=$V \
    libnvidia-compute-580=$V \
    libnvidia-extra-580=$V \
    libnvidia-common-580=$V \
    libnvidia-cfg1-580=$V \
    libnvidia-decode-580=$V \
    libnvidia-encode-580=$V \
    libnvidia-fbc1-580=$V \
    libnvidia-gl-580=$V \
    nvidia-utils-580=$V \
    nvidia-compute-utils-580=$V \
    xserver-xorg-video-nvidia-580=$V \
    2>&1 | tee -a "$LOG"

EC=${PIPESTATUS[0]}
chmod a+r "$LOG"

echo
echo "===== apt install exit=$EC ====="

if [[ $EC -ne 0 ]]; then
    echo "FAILED. Full log: $LOG"
    echo
    echo "If error mentions 'held broken packages' or 'unmet dependencies':"
    echo "  Check if graphics-drivers PPA is still active:"
    echo "    grep -l graphics-drivers /etc/apt/sources.list.d/*"
    echo "  If yes, disable temporarily:"
    echo "    sudo mv /etc/apt/sources.list.d/graphics-drivers*.list{,.disabled}"
    echo "    sudo mv /etc/apt/sources.list.d/graphics-drivers*.sources{,.disabled} 2>/dev/null"
    echo "    bash $0  # re-run"
    exit $EC
fi

echo
echo "STEP 4: verify versions"
VER=$(dpkg -s nvidia-driver-580-open 2>/dev/null | awk -F': ' '/^Version:/{print $2}')
echo "  nvidia-driver-580-open = $VER"

if [[ "$VER" == "$V" ]]; then
    echo "  SUCCESS — version downgraded"
else
    echo "  UNEXPECTED version (wanted $V)"
    echo "  Check $LOG for what apt actually did"
    exit 3
fi

echo
echo "STEP 5: hold all nvidia packages to prevent auto-upgrade"
TO_HOLD=$(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null \
    | awk '/^.i / && /nvidia|libnvidia/ {print $2}')
apt-mark hold $TO_HOLD
echo "  Held packages:"
apt-mark showhold | grep -iE "nvidia|libnvidia" | sed 's/^/    /'

echo
echo "===== DONE $(date) ====="
cat <<EOF

NEXT STEPS:
  1. Reboot:
       sudo reboot
  2. After reboot, verify:
       nvidia-smi --query-gpu=driver_version --format=csv,noheader
       # expect: 580.126.09
  3. Wipe Triton cache:
       rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache
  4. Re-enable persistence:
       sudo nvidia-smi -pm 1
EOF
