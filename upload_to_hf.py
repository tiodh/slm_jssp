"""Upload all 8 fine-tuned LoRA/rsLoRA adapters to Hugging Face.

Usage:
    HF_TOKEN=hf_xxx python upload_to_hf.py [--user tiodh] [--dry-run]

Each adapter is uploaded as its own model repo with an auto-generated model
card containing the eval metrics from comparison_lora_vs_rslora.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

EXCLUDE_PATTERNS = [
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
    "README.md",  # auto-generated trainer README; we write our own
]

ADAPTERS = [
    {
        "key":        "llama_lora",
        "model":      "llama",
        "method":     "lora",
        "repo":       "llama3.1-8b-jssp-lora",
        "base":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "base_pretty":"Llama-3.1-8B-Instruct",
        "ckpt":       "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406",
    },
    {
        "key":        "llama_rslora",
        "model":      "llama",
        "method":     "rslora",
        "repo":       "llama3.1-8b-jssp-rslora",
        "base":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "base_pretty":"Llama-3.1-8B-Instruct",
        "ckpt":       "output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230",
    },
    {
        "key":        "granite_lora",
        "model":      "granite",
        "method":     "lora",
        "repo":       "granite3.2-8b-jssp-lora",
        "base":       "ibm-granite/granite-3.2-8b-instruct",
        "base_pretty":"Granite-3.2-8B-Instruct",
        "ckpt":       "output_granite8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406",
    },
    {
        "key":        "granite_rslora",
        "model":      "granite",
        "method":     "rslora",
        "repo":       "granite3.2-8b-jssp-rslora",
        "base":       "ibm-granite/granite-3.2-8b-instruct",
        "base_pretty":"Granite-3.2-8B-Instruct",
        "ckpt":       "output_granite8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230",
    },
    {
        "key":        "ministral_lora",
        "model":      "ministral",
        "method":     "lora",
        "repo":       "ministral-8b-jssp-lora",
        "base":       "mistralai/Ministral-8B-Instruct-2410",
        "base_pretty":"Ministral-8B-Instruct-2410",
        "ckpt":       "output_ministral8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406",
    },
    {
        "key":        "ministral_rslora",
        "model":      "ministral",
        "method":     "rslora",
        "repo":       "ministral-8b-jssp-rslora",
        "base":       "mistralai/Ministral-8B-Instruct-2410",
        "base_pretty":"Ministral-8B-Instruct-2410",
        "ckpt":       "output_ministral8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230",
    },
    {
        "key":        "qwen2_lora",
        "model":      "qwen2",
        "method":     "lora",
        "repo":       "qwen2-7b-jssp-lora",
        "base":       "Qwen/Qwen2-7B-Instruct",
        "base_pretty":"Qwen2-7B-Instruct",
        "ckpt":       "output_qwen2_7b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406",
    },
    {
        "key":        "qwen2_rslora",
        "model":      "qwen2",
        "method":     "rslora",
        "repo":       "qwen2-7b-jssp-rslora",
        "base":       "Qwen/Qwen2-7B-Instruct",
        "base_pretty":"Qwen2-7B-Instruct",
        "ckpt":       "output_qwen2_7b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230",
    },
]


def model_card(entry: dict, metrics: dict) -> str:
    method_pretty = "rsLoRA" if entry["method"] == "rslora" else "LoRA"
    m = metrics  # row from comparison_lora_vs_rslora.json
    return f"""---
library_name: peft
base_model: {entry['base']}
tags:
- jssp
- job-shop-scheduling
- scheduling
- lora
- {entry['method']}
license: cc-by-sa-4.0
datasets:
- henri24/Starjob
---

# {entry['base_pretty']} + {method_pretty} — Job-Shop Scheduling

A {method_pretty} adapter fine-tuned on the [Starjob](https://huggingface.co/datasets/henri24/Starjob)
job-shop scheduling problem (JSSP) dataset. The model takes a natural-language
description of jobs and machines and produces a feasible schedule that minimizes
makespan.

## Training

| Hyperparameter | Value |
|---|---|
| Method | {method_pretty} (`use_rslora = {str(entry['method'] == 'rslora').lower()}`) |
| LoRA rank `r` | 32 |
| LoRA alpha | 32 |
| Max sequence length | 8192 |
| Per-device batch | 1 |
| Gradient accumulation | 8 (effective batch 8) |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Base quantization | bnb 4-bit (Unsloth) |

## Evaluation

200 samples (seed 42) from the small+medium split of Starjob, identical
pipeline for LoRA and rsLoRA. Feasibility validates routing order, machine
non-overlap, and operation completeness.

| Metric | Value |
|---|---|
| Feasibility | {m['feasible_pct']:.1f}% ({m['feasible']}/200) |
| Exact makespan | {m['exact_pct']:.1f}% ({m['exact_makespan']}/200) |
| Mean gap | {m['mean_gap_pct']:.2f}% |
| Median gap | {m['median_gap_pct']:.2f}% |
| Eval time | {m['time_min']:.1f} min |

Full head-to-head LoRA vs rsLoRA comparison and code:
[github.com/tiodh/slm_jssp](https://github.com/tiodh/slm_jssp).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{entry['base']}",
    device_map="auto",
    torch_dtype="auto",
)
tok = AutoTokenizer.from_pretrained("{entry['base']}")
model = PeftModel.from_pretrained(base, "tiodh/{entry['repo']}")

prompt = (
    "Optimize schedule for 3 Jobs (denoted as J) across 3 Machines (denoted as M) "
    "to minimize makespan...\\nJ0:\\nM0:5 M1:3 M2:4\\nJ1:\\nM1:2 M0:4 M2:3\\nJ2:\\nM2:6 M0:1 M1:5\\n"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

## License

CC BY-SA 4.0 (inherits from the Starjob dataset).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="tiodh")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", help="limit to specific keys")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        sys.exit("ERROR: set HF_TOKEN env var (or use --dry-run)")

    with open("comparison_lora_vs_rslora.json") as f:
        comparison = json.load(f)
    metrics_by_key = {f"{r['model']}_{r['method']}": r for r in comparison["rows"]}

    api = HfApi(token=token) if not args.dry_run else None

    for entry in ADAPTERS:
        if args.only and entry["key"] not in args.only:
            continue
        ckpt = Path(entry["ckpt"])
        if not ckpt.exists():
            print(f"SKIP {entry['key']}: missing {ckpt}")
            continue

        repo_id = f"{args.user}/{entry['repo']}"
        card = model_card(entry, metrics_by_key[entry["key"]])
        card_path = ckpt / "README.md"

        # write fresh card (overwrites trainer's auto-README)
        card_path.write_text(card)

        size_mb = sum(f.stat().st_size for f in ckpt.iterdir() if f.is_file()
                      and f.name not in EXCLUDE_PATTERNS) / 1e6
        print(f"\n=== {entry['key']} -> {repo_id} ({size_mb:.0f} MB to upload) ===")

        if args.dry_run:
            print(f"  [dry-run] would create {repo_id} and upload {ckpt}")
            print(f"  excluding: {EXCLUDE_PATTERNS}")
            continue

        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(ckpt),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=EXCLUDE_PATTERNS[:-1],  # keep our README.md
            commit_message=f"Upload {entry['method']} adapter for {entry['model']}",
        )
        print(f"  done -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
