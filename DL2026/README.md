# DL2026 — JSSP Fine-Tuning (LoRA / rsLoRA)

Deep Learning 2026 course assignment. Fine-tune a Large Language Model to
solve the **Job-Shop Scheduling Problem (JSSP)** with a parameter-efficient
approach (LoRA and rsLoRA), then evaluate on an in-distribution dataset (SM)
and two classic out-of-distribution benchmarks (FT, LA).

## Folder layout

```
DL2026/
├── README.md            <- this file
├── main.py              <- CLI entry point (train / eval / infer)
├── requirements.txt
├── src/
│   ├── config.py        <- MODEL_REGISTRY, paths, hyperparameters, BKS
│   ├── data.py          <- SM (train) + FT/LA (eval) loaders
│   ├── model.py         <- FastLanguageModel + LoRA / rsLoRA setup
│   ├── trainer.py       <- SFTTrainer wrapper
│   ├── evaluator.py     <- generate + feasibility + gap-to-BKS
│   └── jssp_checker.py  <- JSSP schedule validator (5 violation categories)
├── data/
│   ├── starjob_train_sm.jsonl              <- SM: training set (108k examples)
│   ├── lawrence_prompt_style.jsonl         <- LA01–LA40 (prompt-style)
│   └── benchmarks/
│       ├── jobshop1.txt                    <- OR-Library: FT + LA + ABZ + ORB + ...
│       └── jobshop2.txt                    <- OR-Library: SWV + YN
└── outputs/             <- trained adapters are saved here
```

## Datasets

| Tag | Source                              | Count   | Role                |
|-----|-------------------------------------|---------|---------------------|
| SM  | `data/starjob_train_sm.jsonl`       | 108,000 | **Training**        |
| FT  | `ft06`, `ft10`, `ft20` (jobshop1)   | 3       | Eval (Fisher–Thompson) |
| LA  | `la01`–`la10`, `la16`–`la20`        | 15      | Eval (Lawrence)     |

Prompt format: **Alpaca** (instruction + input + response). Each input
describes the number of jobs, number of machines, and routing per job; the
output is a complete schedule in the form `J<j>-M<m>: s + d -> e`.

## Setup

```bash
# 1) Create venv (Python 3.10+)
python -m venv venv
source venv/bin/activate

# 2) Install PyTorch matching your CUDA toolkit
#    (this repo was tested with CUDA 12.1):
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Install the remaining dependencies
pip install -r requirements.txt
```

Hardware notes:
- An NVIDIA GPU with ≥ 16 GB VRAM is recommended (4-bit quantization is on by default).
- Tested on an RTX 4090 24 GB. For a 12 GB GPU, drop `--max-seq-length` to 4096
  and/or use `--lora-r 16 --lora-alpha 16`.

### Environment variables (required for training)

Before running `main.py train`, export the following environment variables to
keep training stable (these prevent a `tokenizers.abi3.so` segfault and a
`double free` crash when Unsloth loads the tokenizer in parallel):

```bash
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
```

If `libjemalloc.so.2` is missing: `sudo apt install libjemalloc2`.

## Fine-tuning via `main.py`

### 1. Basic training (LoRA, LLaMA-3.1-8B)

```bash
python main.py train --model llama
```

This will:
1. Load `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit BNB).
2. Attach a LoRA adapter (r=32, α=32, dropout=0.0) on
   `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
3. Load `data/starjob_train_sm.jsonl` and reformat it into the Alpaca template.
4. Run `SFTTrainer` (1 epoch, effective batch = 1 × 8, LR=2e-4, optimizer adamw_8bit).
5. Save a checkpoint every `--save-steps` (default 200) and the final adapter to
   `outputs/llama_lora_r32_a32_seq8192_b1_ga8/final_adapter/`.

### 2. Other model choices

```bash
python main.py train --model qwen2       # Qwen2-7B-Instruct
python main.py train --model granite     # Granite-3.2-8B-Instruct
python main.py train --model ministral   # Ministral-8B-Instruct-2410
```

### 3. Use rsLoRA (α/√r scaling)

```bash
python main.py train --model llama --use-rslora
```

Adapter is saved to `outputs/llama_rslora_*/final_adapter/`.

### 4. Quick smoke test (50 steps)

```bash
python main.py train --model llama --max-steps 50 --save-steps 25 --logging-steps 5
```

Finishes in ~3 minutes on 1× RTX 4090 (≈ 3 s / step).
Loss curve from a reference smoke run (50 steps, LoRA, LLaMA-3.1-8B):

| step | loss   |
|------|--------|
|  5   | 0.9818 |
| 10   | 0.5815 |
| 15   | 0.4551 |
| 20   | 0.4326 |
| 25   | 0.4362 |
| 30   | 0.4079 |
| 35   | 0.4232 |
| 40   | 0.3902 |
| 45   | 0.4256 |
| 50   | 0.4130 |

`train_runtime = 156.8 s`, `train_samples_per_second = 2.55`, adapter saved
to `outputs/llama_lora_r32_a32_seq8192_b1_ga8/final_adapter/` (335 MB).

### 5. Custom hyperparameters

```bash
python main.py train --model llama \
    --use-rslora \
    --lora-r 64 --lora-alpha 64 \
    --learning-rate 1e-4 \
    --epochs 1 \
    --per-device-batch-size 1 --grad-accum 16 \
    --max-seq-length 8192
```

For the full argument list:

```bash
python main.py train --help
```

## Evaluating an adapter

After training, run evaluation on any of the three datasets:

```bash
# Sanity check on SM (in-distribution, first 20 samples)
python main.py eval --model llama --adapter outputs/llama_lora_*/final_adapter \
    --dataset sm --n-sm-samples 20

# OOD: Fisher–Thompson (3 instances: ft06, ft10, ft20)
python main.py eval --model llama --adapter outputs/llama_lora_*/final_adapter \
    --dataset ft

# OOD: Lawrence (15 instances: la01-la10, la16-la20)
python main.py eval --model llama --adapter outputs/llama_lora_*/final_adapter \
    --dataset la

# FT + LA together (18 instances)
python main.py eval --model llama --adapter outputs/llama_lora_*/final_adapter \
    --dataset all
```

Results:
- Saved to `outputs/.../final_adapter/eval_<dataset>.json` (override with `--out`).
- Per instance: `feasible`, `makespan`, `gap_to_bks`, and a violation breakdown
  (precedence, routing, timing, machine-capacity, missing-op, over-op).
- Summary: `feasibility_rate` and `mean_gap_to_bks` (computed over feasible instances only).

## Single-instance inference (debug)

```bash
python main.py infer --model llama --adapter outputs/llama_lora_*/final_adapter \
    --instance la01
```

Prints the raw response and the verdict (feasible / makespan / gap).

## Feasibility metric (5 violation categories)

Implemented in `src/jssp_checker.py`:

| Category                        | Meaning                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `precedence_violations`         | Within a job, op `i` starts before op `i-1` finishes.                |
| `routing_order_violations`      | Machine or duration at routing position `i` is wrong.                |
| `timing_consistency_violations` | A parsed op has `start + duration ≠ end`.                            |
| `machine_capacity_violations`   | Two ops on the same machine overlap in time.                         |
| `missing_op_count`              | Expected operations that were never emitted.                         |

A schedule is **feasible** if all five categories are 0 (and no over-op either).

## Brief technical notes

- **Unsloth** is used as a Triton-fused backend, giving ~2× speedup over vanilla
  HuggingFace while keeping LoRA adapters and 4-bit quantization active at the same time.
- **rsLoRA** (Kalajdzievski 2023) uses `α / √r` scaling instead of `α / r`, which
  is usually more stable at `r ≥ 32`. Toggle with `--use-rslora`.
- **WANDB**: disabled (`report_to="none"` in `TrainingArguments`).
- **Reproducibility**: default seed 42 (not exposed via CLI yet — edit
  `src/config.py:DEFAULT_HYPERPARAMS` if you need to change it).

## Execution flow

```
  ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐
  │ SM (108k)      │───▶│  SFTTrainer     │───▶│ final_adapter/   │
  │ Alpaca jsonl   │    │  LoRA / rsLoRA  │    │ (LoRA weights)   │
  └────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                        │
                              ┌─────────────────────────┴────────┐
                              ▼                                  ▼
                  ┌───────────────────┐               ┌────────────────────┐
                  │ Evaluator         │               │ Per-instance JSON  │
                  │ - generate        │               │ {feasible, gap,    │
                  │ - 5-cat feasib.   │──────────────▶│  makespan, viol}   │
                  │ - gap-to-BKS      │               └────────────────────┘
                  └───────────────────┘
                          │
                 ┌────────┴─────────┬──────────────┐
                 ▼                  ▼              ▼
              SM eval           FT eval        LA eval
            (sanity 20)        (3 inst.)     (15 inst.)
```

## License

The code in this folder was assembled for a Deep Learning coursework
assignment. Each dependency keeps its own license (Unsloth: Apache-2.0;
TRL: Apache-2.0; Llama-3.1: Meta llama3 license; Qwen2: Apache-2.0;
Granite: Apache-2.0; Mistral: research license).
