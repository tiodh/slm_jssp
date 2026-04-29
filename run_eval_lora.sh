#!/bin/bash
# Evaluate all LoRA models on small+medium JSSP instances
# Mirrors run_eval_rslora.sh exactly so LoRA vs rsLoRA results are head-to-head comparable.
set -e
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

NUM_SAMPLES=200

echo "============================================"
echo "  [1/4] Evaluating LLaMA 3.1 8B LoRA"
echo "============================================"
python eval_lora.py --model llama --num_samples $NUM_SAMPLES 2>&1 | tee eval_lora_llama.log

echo "============================================"
echo "  [2/4] Evaluating Granite 3.2 8B LoRA"
echo "============================================"
python eval_lora.py --model granite --num_samples $NUM_SAMPLES 2>&1 | tee eval_lora_granite.log

echo "============================================"
echo "  [3/4] Evaluating Ministral 8B LoRA"
echo "============================================"
python eval_lora.py --model ministral --num_samples $NUM_SAMPLES 2>&1 | tee eval_lora_ministral.log

echo "============================================"
echo "  [4/4] Evaluating Qwen2 7B LoRA"
echo "============================================"
python eval_lora.py --model qwen2 --num_samples $NUM_SAMPLES 2>&1 | tee eval_lora_qwen2.log

echo "============================================"
echo "  All LoRA evaluations completed!"
echo "============================================"
