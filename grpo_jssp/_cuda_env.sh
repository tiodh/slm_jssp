#!/bin/bash
# Source this file before any Unsloth/torch invocation.
# Mitigation for libcuda.so 580.126.09 intermittent crashes:
#  - expandable_segments allocator: reduces fragmentation across reinits
#  - CUDA_MODULE_LOADING=LAZY: defer kernel compilation, lower init failure
#  - CUDA_VISIBLE_DEVICES=0: pin to GPU 0 (single device)
#  - Limit threads to avoid CPU oversubscription that has correlated with crashes
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
