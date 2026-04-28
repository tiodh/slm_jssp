#!/bin/bash
# Evaluate all rsLoRA models on small+medium JSSP instances
set -e
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

NUM_SAMPLES=200

echo "============================================"
echo "  [1/4] Evaluating LLaMA 3.1 8B rsLoRA"
echo "============================================"
python eval_rslora.py --model llama --num_samples $NUM_SAMPLES 2>&1 | tee eval_rslora_llama.log

echo "============================================"
echo "  [2/4] Evaluating Granite 3.2 8B rsLoRA"
echo "============================================"
python eval_rslora.py --model granite --num_samples $NUM_SAMPLES 2>&1 | tee eval_rslora_granite.log

echo "============================================"
echo "  [3/4] Evaluating Ministral 8B rsLoRA"
echo "============================================"
python eval_rslora.py --model ministral --num_samples $NUM_SAMPLES 2>&1 | tee eval_rslora_ministral.log

echo "============================================"
echo "  [4/4] Evaluating Qwen2 7B rsLoRA"
echo "============================================"
python eval_rslora.py --model qwen2 --num_samples $NUM_SAMPLES 2>&1 | tee eval_rslora_qwen2.log

echo "============================================"
echo "  All rsLoRA evaluations completed!"
echo "============================================"
