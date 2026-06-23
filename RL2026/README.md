# RL2026 — CAP-GRPO: Constraint-Aware Policy with GRPO for JSSP

Reinforcement Learning 2026 course assignment. Fine-tune a Large Language Model
to solve the **Job-Shop Scheduling Problem (JSSP)** using **GRPO** (Group
Relative Policy Optimization), guided by a constraint-aware reward with a
**coverage gate** that prevents reward hacking from vacuous zero violations.

## Background

### GRPO (Group Relative Policy Optimization)
GRPO (Shao et al., 2024; used in DeepSeek-R1) is a policy-gradient algorithm
that computes advantages relative to a group of K completions sampled for the
same prompt. No value network is needed — baseline = mean reward of the group.

```
A_i = (r_i - mean(r_1..r_K)) / std(r_1..r_K)
L_GRPO = -E[A_i * log π_θ(y_i|x)] + β * KL(π_θ || π_ref)
```

### CAP (Constraint-Aware Policy)
The reward function encodes 7 JSSP constraints:

| Component | Signal |
|-----------|--------|
| R_format  | +1 if output is parseable (has at least one valid op) |
| R_M       | +1 if no missing ops; else -(n_missing / N_ops) |
| R_R       | +1×cov if no routing violations; else -(n_viol / N_ops); bounded ≥ −1 |
| R_C       | +1×cov if no machine-capacity violations; **unbounded** (sequential count) |
| R_T       | +1×cov if no timing inconsistencies (s+d≠e); **unbounded** (no cap on bad ops) |
| R_P       | +1×cov if no precedence violations |
| R_quality | BKS/C_max if fully feasible (makespan quality bonus) |

**Coverage gate** (`cov = ops_emitted / ops_expected`): structural constraints
give `+cov` instead of `+1` when no violations are found. This prevents an
empty output (0 ops emitted) from collecting free +1 rewards.

Additional mode:
- **hybrid_v7**: adds R_O (over-emit penalty), closing the padding loophole.

### Length Control (V5 technique)
Completions longer than `OVERLEN_FACTOR × gold_est` get their GRPO advantage
zeroed — they contribute no gradient. Prevents length-escape collapse where the
model learns to produce arbitrarily long outputs to avoid constraint checks.

---

## Folder layout

```
RL2026/
├── main.py              ← CLI entry point (train / eval / infer)
├── requirements.txt
├── README.md
├── src/
│   ├── config.py        ← model registry, hyperparams, reward config
│   ├── checker.py       ← 6-category JSSP violation checker
│   ├── reward.py        ← compute_reward (hybrid/V7/stratified/uniform)
│   ├── data.py          ← SM (train) + OOD FT/LA (eval) loaders
│   ├── model.py         ← auto-download or local model loader
│   ├── trainer.py       ← GRPOTrainer + LengthControlledGRPOTrainer
│   └── evaluator.py     ← generate + check + aggregate
└── data/
    ├── starjob_train_sm.jsonl    ← SM training set (108k JSSP instances)
    └── benchmarks/
        └── jobshop1.txt          ← OR-Library: FT + LA + ABZ + ORB + ...
```

---

## Datasets

| Tag | Source                         | Count   | Role |
|-----|--------------------------------|---------|------|
| SM  | `starjob_train_sm.jsonl`       | 108,000 | Training (98% split, seed=42) |
| FT  | ft06, ft10, ft20               | 3       | OOD eval (Fisher–Thompson) |
| LA  | la01–la10, la16–la20           | 15      | OOD eval (Lawrence) |

The 2% test split (seed=42) is held out from GRPO training to match the
SFT evaluation baseline.

---

## Setup

```bash
# 1) Create venv (Python 3.10+)
python -m venv venv-grpo && source venv-grpo/bin/activate

# 2) Install PyTorch (CUDA 12.1 example):
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Install remaining dependencies:
pip install -r requirements.txt
```

> **Note:** Use a dedicated venv (e.g., `venv-grpo`) separate from any SFT venv.
> The `llm_blender` package shipped with some TRL versions conflicts with
> newer `transformers` — a fresh venv avoids this.

Hardware: ≥ 16 GB VRAM recommended (RTX 4090 24 GB tested). GRPO stores
K=4 completions per prompt per step — more memory-intensive than SFT.

### Required environment variables

```bash
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
```

If `libjemalloc` is missing: `sudo apt install libjemalloc2`.

---

## Training via `main.py`

### Option A — start from a local SFT adapter (recommended)

Starting GRPO from a supervised fine-tuned adapter gives the model a working
JSSP output format before RL begins, which dramatically improves sample quality.

```bash
python main.py train \
    --model-path /path/to/sft_final_adapter \
    --reward-mode hybrid \
    --max-steps 500
```

### Option B — start from a base model (auto-download)

```bash
python main.py train --model llama --reward-mode hybrid --max-steps 500
# Other models: qwen2 | granite | ministral
```

The first run downloads the model (~4-5 GB) and caches it automatically.

### Reward mode options

| Mode         | Description | Range |
|--------------|-------------|-------|
| `hybrid`     | V4: 7-component + coverage gate (default) | (-∞, 7]; unparseable floor = −1.0; practical ≈ [−3.8, 7] |
| `hybrid_v7`  | V7: hybrid + over-emit penalty R_O | (-∞, 8]; same floor as hybrid |
| `stratified` | V1: per-category weighted penalty | [-1, 1] |
| `uniform`    | Binary: {0, 1, 7} | {0, 1, 7} |

### Length control

```bash
python main.py train --model-path /path/to/adapter \
    --reward-mode hybrid --length-control
```

Zeroes GRPO advantages for completions exceeding `2.0 × (12.5 × N_ops + 50)`
tokens.

### Smoke test (10 steps)

```bash
python main.py train \
    --model-path /path/to/sft_adapter \
    --reward-mode hybrid \
    --max-steps 10 \
    --max-records 50 \
    --save-every 10
```

### Full argument list

```bash
python main.py train --help
```

### Resume from checkpoint

```bash
python main.py train \
    --model-path outputs/capgrpo_llama_hybrid/checkpoint-200 \
    --resume-from outputs/capgrpo_llama_hybrid/checkpoint-200 \
    --max-steps 500
```

---

## Evaluation

```bash
# In-distribution (SM held-out 2% test split, 20 samples)
python main.py eval --model-path outputs/capgrpo_llama_hybrid/final_adapter \
    --dataset sm --n-sm-samples 20

# OOD: Fisher–Thompson (3 instances)
python main.py eval --model-path outputs/capgrpo_llama_hybrid/final_adapter \
    --dataset ft

# OOD: Lawrence (15 instances)
python main.py eval --model-path outputs/capgrpo_llama_hybrid/final_adapter \
    --dataset la

# FT + LA together (18 instances)
python main.py eval --model-path outputs/capgrpo_llama_hybrid/final_adapter \
    --dataset all --out eval_results.json
```

Output fields per instance: `feasible`, `makespan`, `gap_to_bks`,
`missing_op_count`, `routing_order_violations`, `machine_capacity_violations`,
`timing_consistency_violations`, `precedence_violations`.

---

## Single-instance inference (debug)

```bash
python main.py infer --model-path outputs/capgrpo_llama_hybrid/final_adapter \
    --instance la01
```

Prints the raw response and a JSON verdict with feasibility and per-category
violation counts.

---

## Reward decomposition

```
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality    (hybrid, V4)
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality + R_O  (hybrid_v7)

R_format = +1 if parseable, else -1 (hard floor)
R_M      = +1 if missing==0, else -(missing/N_ops)
R_R      = +cov if routing_viol==0, else -(routing_viol/N_ops)
R_C      = +cov if capacity_viol==0, else -(capacity_viol/N_ops)
R_T      = +cov if timing_viol==0, else -(timing_viol/N_ops)
R_P      = +cov if prec_viol==0, else -(prec_viol/N_ops)
R_quality= BKS/C_max if feasible (capped at 1.0), else 0
R_O      = +1 if over_emit==0, else -(over_emit/N_ops)  [V7 only]

cov = ops_emitted / ops_expected  (coverage gate)
```

---

## Execution flow

```
  ┌────────────────────┐    ┌──────────────────────────┐    ┌─────────────────┐
  │  SM (98% train)    │───▶│  GRPO Training           │───▶│ final_adapter/  │
  │  108k JSSP prompts │    │  K=4 completions/prompt  │    │ (GRPO weights)  │
  └────────────────────┘    │  Reward: CAP 7-component │    └────────┬────────┘
                            │  KL penalty vs SFT ref   │            │
                            └──────────────────────────┘            │
                                                         ┌──────────▼──────────┐
                                                         │   Evaluator         │
                                                         │  - generate(T=0.1)  │
                                                         │  - check_violations │
                                                         │  - gap to BKS       │
                                                         └──────────┬──────────┘
                                              ┌────────────┬────────┴──────────┐
                                              ▼            ▼                   ▼
                                           SM eval      FT eval            LA eval
                                          (20 test)    (3 inst)          (15 inst)
```

---

## References

- Shao et al. (2024) — *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models* (GRPO algorithm)
- Kalajdzievski (2023) — *A Rank Stabilization Scaling Factor for Fine-Tuning
  with LoRA* (rsLoRA)
- Fisher & Thompson (1963) — *Probabilistic Learning Combinations of Local
  Job-Shop Scheduling Rules* (FT benchmarks)
- Lawrence (1984) — *Resource Constrained Project Scheduling* (LA benchmarks)

## License

Code assembled for a Reinforcement Learning coursework assignment. Each
dependency retains its own license (Unsloth: Apache-2.0; TRL: Apache-2.0;
LLaMA-3.1: Meta Llama 3.1 Community License; Qwen2: Apache-2.0; Granite:
Apache-2.0; Mistral: research license).
