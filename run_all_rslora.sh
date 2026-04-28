#!/bin/bash
# Fine-tune all models using rsLoRA
# Models: LLaMA 3.1 8B -> Granite 3.2 8B -> Ministral 8B -> Qwen2 7B

set -e
export WANDB_DISABLED=true
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "============================================"
echo "  [1/4] Training LLaMA 3.1 8B with rsLoRA"
echo "============================================"
python train_llama_3.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1

echo "============================================"
echo "  [2/4] Training Granite 3.2 8B with rsLoRA"
echo "============================================"
python train_granite_8b.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1

echo "============================================"
echo "  [3/4] Training Ministral 8B with rsLoRA"
echo "============================================"
python train_ministral_8b.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1

echo "============================================"
echo "  [4/4] Training Qwen2 7B with rsLoRA"
echo "============================================"
python train_qwen2_7b.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1

echo "============================================"
echo "  All rsLoRA training completed!"
echo "============================================"
