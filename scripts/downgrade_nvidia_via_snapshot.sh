#!/usr/bin/env bash
#
# Downgrade nvidia-driver-580-open from 580.159.03 -> 580.126.09 via the
# Ubuntu Snapshot Service. Apr 15 archive is the youngest snapshot that
# still has 580.126.09 (May 1 jumped to 580.142, May 20 jumped to 580.159).
#
# Strategy: install the META + kernel-module packages from the snapshot
# and let apt resolve all the libnvidia/nvidia-firmware deps. Then re-hold
# the full installed set so unattended-upgrades can't bump it back.

set -uo pipefail

SNAPSHOT="20260415T000000Z"   # Ubuntu archive state with 580.126.09

# Auto-wrap in tmux so SSH drop can't kill the install
if [[ -z "${TMUX:-}" ]] && [[ "${1:-}" != "--inner" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux missing — proceeding without isolation"
    else
        SESS="nvdg_$(date +%H%M%S)"
        echo "Launching inside tmux session: $SESS"
        echo "  Reattach if disconnected:  tmux attach -t $SESS"
        echo "  Detach while attached:     Ctrl+B then D"
        sleep 2
        exec tmux new-session -s "$SESS" "bash '$0' --inner"
    fi
fi

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo "$0" --inner
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "Pre-flight"
echo "Current driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo NONE)"
echo "Target:          580.126.09 (via snapshot $SNAPSHOT)"
echo "Kernel:          $(uname -r)"

# Discover the metapackage variant the user has (open vs proprietary vs server)
META=""
for cand in nvidia-driver-580-open nvidia-driver-580 nvidia-driver-580-server; do
    if dpkg -s "$cand" >/dev/null 2>&1; then
        META="$cand"
        break
    fi
done
if [[ -z "$META" ]]; then
    echo "ERROR: no nvidia-driver-580 metapackage installed"
    exit 1
fi
echo "Metapackage:     $META"

KMOD_PKG="linux-modules-nvidia-580-open-$(uname -r)"
echo "Kernel module:   $KMOD_PKG"

# Currently held nvidia packages — must unhold before downgrade
mapfile -t HELD < <(apt-mark showhold 2>/dev/null | grep -iE "nvidia|libnvidia" || true)
echo "Currently held:  ${#HELD[@]} packages"
for h in "${HELD[@]}"; do echo "  $h"; done

log "Snapshot of current state"
SNAP=/tmp/nvidia-rollback-$(date +%Y%m%d-%H%M%S).txt
dpkg-query -W -f='${db:Status-Abbrev} ${Package}=${Version}\n' \
    | awk '/^.i / && /nvidia|libnvidia/ {print $2}' \
    | tee "$SNAP"
echo "Saved: $SNAP"

log "Plan"
echo "  - apt-mark unhold ${#HELD[@]} held packages"
echo "  - apt install --update --snapshot $SNAPSHOT --allow-downgrades $META $KMOD_PKG"
echo "    (apt resolves all libnvidia-*-580 and nvidia-firmware-* deps automatically)"
echo "  - apt-mark hold all installed nvidia/libnvidia packages afterwards"
echo "  - Reboot required after this script"
echo
read -rp "Proceed? [y/N] " ans
[[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 0; }

log "Stop display manager"
for dm in gdm gdm3 lightdm sddm; do
    if systemctl is-active --quiet "$dm" 2>/dev/null; then
        echo "  stop $dm"
        systemctl stop "$dm" || true
    fi
done

if [[ ${#HELD[@]} -gt 0 ]]; then
    log "Unhold ${#HELD[@]} packages"
    apt-mark unhold "${HELD[@]}"
fi

log "apt install META + kernel module from snapshot $SNAPSHOT"
META_VER="580.126.09-0ubuntu0.24.04.2"
KMOD_VER="6.17.0-20.20~24.04.1"
echo "(this will downgrade ~15 nvidia packages, ~300 MB download)"
echo "Pinning explicit versions: $META=$META_VER and $KMOD_PKG=$KMOD_VER"
echo "Output also captured to /tmp/nvidia-downgrade-install.log"
echo

APT_LOG=/tmp/nvidia-downgrade-install.log
chmod a+r "$SNAP" 2>/dev/null || true

set -o pipefail
if ! apt install --update --snapshot "$SNAPSHOT" --allow-downgrades -y \
        "${META}=${META_VER}" "${KMOD_PKG}=${KMOD_VER}" 2>&1 | tee "$APT_LOG"; then
    chmod a+r "$APT_LOG" 2>/dev/null || true
    echo
    echo "ERROR: snapshot install failed."
    echo "Full log: $APT_LOG (readable without sudo)"
    echo "Rollback snapshot saved: $SNAP"
    echo
    echo "Most-likely apt error reasons:"
    echo "  - Package version mismatch (snapshot vs current dep) -> sudo apt install -f"
    echo "  - Still-held package somewhere -> sudo apt-mark unhold (list)"
    echo "  - Snapshot 410 Gone for that date -> try 20260410T000000Z"
    echo
    echo "Recovery to re-pin current state:"
    echo "  sudo apt-mark hold \$(cat $SNAP)"
    exit 2
fi
chmod a+r "$APT_LOG" 2>/dev/null || true

log "Verify new versions"
dpkg-query -W -f='${db:Status-Abbrev} ${Package} ${Version}\n' \
    | awk '/^.i / && /nvidia.*580|libnvidia.*580/ {printf "  %-50s %s\n", $2, $3}'

log "Re-hold ALL installed nvidia/libnvidia packages"
mapfile -t TO_HOLD < <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null \
    | awk '/^.i / {print $2}' \
    | grep -E "^(nvidia|libnvidia|xserver-xorg-video-nvidia)-" \
    | sort -u)
if [[ ${#TO_HOLD[@]} -gt 0 ]]; then
    apt-mark hold "${TO_HOLD[@]}"
fi
echo "Held ${#TO_HOLD[@]} packages:"
apt-mark showhold | grep -iE "nvidia|libnvidia"

cat <<EOF

================================================================
DOWNGRADE DONE. NEXT STEPS:

  1. Reboot:
       sudo reboot

  2. After reboot, verify:
       nvidia-smi --query-gpu=driver_version --format=csv,noheader
       # Expect: 580.126.09

  3. Wipe Triton caches (driver version changed):
       rm -rf ~/.triton/cache /tmp/torchinductor_* ~/.nv/ComputeCache

  4. Re-enable persistence:
       sudo nvidia-smi -pm 1

  5. Sanity canary (20 step) before V7 relaunch:
       bash /home/tio/Documents/Starjob/scripts/run_canary.sh 20

Rollback snapshot: $SNAP
================================================================
EOF
