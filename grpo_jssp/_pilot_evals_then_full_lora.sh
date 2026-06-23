#!/bin/bash
# Master: eval 6 pilot LoRA adapters, then train+eval 6 full LoRA versions.
# v2: NO set -e — let inner scripts handle their own retry/skip logic.

NOTIFY=~/.local/bin/notify
"$NOTIFY" grpo "MASTER v2 START: pilot evals -> full LoRA all" >/dev/null 2>&1 || true
echo "=== $(date) MASTER v2 START ==="

bash /home/tio/Documents/Starjob/grpo_jssp/_eval_pilots_lora.sh
ec1=$?
echo "=== pilot evals exit=$ec1 ==="

bash /home/tio/Documents/Starjob/grpo_jssp/_full_lora_all.sh
ec2=$?
echo "=== full LoRA all exit=$ec2 ==="

"$NOTIFY" grpo "MASTER v2 DONE: pilot_evals=$ec1 full_lora=$ec2" >/dev/null 2>&1 || true
echo "=== $(date) MASTER v2 DONE pilot_evals=$ec1 full_lora=$ec2 ==="
