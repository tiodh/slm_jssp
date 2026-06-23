#!/usr/bin/env bash
# Download pretrained base models (4-bit quantized) from Hugging Face Hub.
# Run once before training. Models are cached at ~/.cache/huggingface/hub/.
#
# Usage:
#   bash download_models.sh           # download all 4 models (~4-5 GB each)
#   bash download_models.sh llama     # download only llama
#   bash download_models.sh qwen2 granite

set -euo pipefail

ALL_MODELS=(llama qwen2 granite ministral)

declare -A MODEL_ID=(
  [llama]="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
  [qwen2]="unsloth/Qwen2-7B-Instruct-bnb-4bit"
  [granite]="unsloth/granite-3.2-8b-instruct-bnb-4bit"
  [ministral]="mistralai/Ministral-8B-Instruct-2410"
)

TARGETS=("${@:-${ALL_MODELS[@]}}")

for key in "${TARGETS[@]}"; do
  if [[ -z "${MODEL_ID[$key]+_}" ]]; then
    echo "Unknown model key: $key. Valid: ${ALL_MODELS[*]}" >&2
    exit 1
  fi
  repo="${MODEL_ID[$key]}"
  echo "==> Downloading $key  ($repo)"
  python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$repo", ignore_patterns=["*.pt", "original/"])
print("Done: $repo")
PYEOF
done

echo ""
echo "All requested models are cached at ~/.cache/huggingface/hub/"
echo "You can now run:  python main.py train --model <key>"
