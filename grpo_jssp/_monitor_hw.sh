#!/bin/bash
# Continuous HW monitor: temps + GPU every 30s. Run alongside training.
# Stops when LOG file path's directory contains a STOP file, or via kill.

OUT="$1"
if [ -z "$OUT" ]; then
  echo "usage: $0 <output_log_path>" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "ts cpu_pkg_C cpu_max_C coolant_C pump_rpm fan_rpm gpu_temp_C gpu_pwr_W gpu_mem_MiB gpu_util_pct" > "$OUT"

while true; do
  ts=$(date +%H:%M:%S)
  cpu_pkg=$(sensors coretemp-isa-0000 2>/dev/null | grep "Package id 0" | grep -oP '\+\K[0-9.]+' | head -1)
  cpu_max=$(sensors coretemp-isa-0000 2>/dev/null | grep -E "^Core" | grep -oP '\+\K[0-9.]+' | grep -v "^80$\|^100$" | sort -n | tail -1)
  coolant=$(sensors kraken2023-hid-3-5 2>/dev/null | grep Coolant | grep -oP '\+\K[0-9.]+' | head -1)
  pump=$(sensors kraken2023-hid-3-5 2>/dev/null | grep "Pump speed" | grep -oP '\d+(?= RPM)' | head -1)
  fan=$(sensors kraken2023-hid-3-5 2>/dev/null | grep "Fan speed" | grep -oP '\d+(?= RPM)' | head -1)
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')
  echo "$ts ${cpu_pkg:-NA} ${cpu_max:-NA} ${coolant:-NA} ${pump:-NA} ${fan:-NA} ${gpu:-NA NA NA NA}" >> "$OUT"
  sleep 30
done
