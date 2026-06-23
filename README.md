# LoRA vs rsLoRA on Job-Shop Scheduling with Small Open-Weight LLMs

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/henri24/Starjob)

This repository fine-tunes four small/medium open-weight LLMs on the Job-Shop Scheduling Problem (JSSP) using two adapter strategies — **LoRA** and **rsLoRA** (rank-stabilized LoRA) — and evaluates them head-to-head with an identical pipeline (same data, same seed, same feasibility validator).

The dataset comes from [Starjob](https://huggingface.co/datasets/henri24/Starjob).

---

## Experiment

### Goal

Find which low-rank adapter method (LoRA vs rsLoRA) yields more *feasible* and *closer-to-optimal* JSSP schedules across multiple base models, when everything else is held fixed.

### Models

| Model | Base | Quantization |
|---|---|---|
| LLaMA 3.1 8B | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | bnb 4-bit |
| Granite 3.2 8B | `unsloth/granite-3.2-8b-instruct-bnb-4bit` | bnb 4-bit |
| Ministral 8B | `mistralai/Ministral-8B-Instruct-2410` | bnb 4-bit |
| Qwen2 7B | `unsloth/Qwen2-7B-Instruct-bnb-4bit` | bnb 4-bit |

### Training configuration (identical for both methods)

| Hyperparameter | Value |
|---|---|
| LoRA rank `r` | 32 |
| LoRA alpha | 32 |
| Max sequence length | 8192 |
| Batch size (per device) | 1 |
| Gradient accumulation | 8 |
| Effective batch | 8 |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Optimizer | (Unsloth default) AdamW 8-bit |
| `use_rslora` | `False` (LoRA) / `True` (rsLoRA) |

The same `train_<model>.py` script is used for both — pass `--use_rslora False` for plain LoRA.

### Evaluation protocol (SFT and GRPO)

| Setting | Value |
|---|---|
| Eval set | `data/starjob_train_sm.jsonl` (small + medium JSSP instances) |
| Sample count | 200 (random, seed = 42) |
| Generation | `temperature=0.1`, `top_p=0.95`, `max_new_tokens=4096` |
| Feasibility check | routing order + machine non-overlap + complete operations |
| Metrics | feasibility %, exact-makespan %, mean / median gap vs. ground truth |

The same generation settings are used in `eval_lora.py`, `eval_rslora.py`, and `grpo_jssp/evaluate.py` (the GRPO eval pipeline), so SFT and GRPO results are directly comparable. Eval temperature `0.1` is distinct from the **GRPO rollout temperature** used during training — see the GRPO section below.

---

## Results

Side-by-side training and evaluation loss for all 4 models × 2 methods (solid = LoRA, dashed = rsLoRA, same color = same model):

![Combined learning curves](loss_curves/learning_curves_combined.png)

### Head-to-head metrics (n = 200, identical pipeline)

| Method | Model | Time | Feasible | Exact | Mean gap | Median gap |
|---|---|---:|---:|---:|---:|---:|
| **LoRA** | LLaMA 3.1 8B | 21.5 min | **96.5%** | **34.5%** | **6.88%** | **3.47%** |
| rsLoRA | LLaMA 3.1 8B | 24.2 min | 95.0% | 32.0% | 9.80% | 5.29% |
| **LoRA** | Granite 3.2 8B | 134.9 min | **86.5%** | **33.5%** | **56.15%** | **4.76%** |
| rsLoRA | Granite 3.2 8B | 147.9 min | 24.5% | 5.5% | 215.27% | 41.42% |
| **LoRA** | Ministral 8B | 93.8 min | **95.0%** | **32.0%** | **15.67%** | **4.91%** |
| rsLoRA | Ministral 8B | 118.9 min | 64.0% | 24.5% | 42.93% | 9.25% |
| LoRA | Qwen2 7B | 38.6 min | 1.0% | 3.0% | 56.30% | 28.29% |
| **rsLoRA** | Qwen2 7B | 31.8 min | **50.0%** | **27.5%** | **27.81%** | **9.37%** |

Bold = winner per (model, metric). Full structured numbers: [`comparison_lora_vs_rslora.json`](comparison_lora_vs_rslora.json).

### Model Weights

All 8 adapters (~321–402 MB each) are published on Hugging Face. Load with `peft.PeftModel.from_pretrained(base, repo_id)`.

| Base model | LoRA | rsLoRA |
|---|---|---|
| Llama-3.1-8B-Instruct  | [`tiodh/llama3.1-8b-jssp-lora`](https://huggingface.co/tiodh/llama3.1-8b-jssp-lora)   | [`tiodh/llama3.1-8b-jssp-rslora`](https://huggingface.co/tiodh/llama3.1-8b-jssp-rslora) |
| Granite-3.2-8B-Instruct | [`tiodh/granite3.2-8b-jssp-lora`](https://huggingface.co/tiodh/granite3.2-8b-jssp-lora) | [`tiodh/granite3.2-8b-jssp-rslora`](https://huggingface.co/tiodh/granite3.2-8b-jssp-rslora) |
| Ministral-8B-Instruct-2410 | [`tiodh/ministral-8b-jssp-lora`](https://huggingface.co/tiodh/ministral-8b-jssp-lora) | [`tiodh/ministral-8b-jssp-rslora`](https://huggingface.co/tiodh/ministral-8b-jssp-rslora) |
| Qwen2-7B-Instruct | [`tiodh/qwen2-7b-jssp-lora`](https://huggingface.co/tiodh/qwen2-7b-jssp-lora) | [`tiodh/qwen2-7b-jssp-rslora`](https://huggingface.co/tiodh/qwen2-7b-jssp-rslora) |

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto", torch_dtype="auto",
)
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "tiodh/llama3.1-8b-jssp-lora")
```

### Findings

- **LoRA wins on 3 of 4 models** (LLaMA, Granite, Ministral) on every metric — feasibility, exactness, and gap.
- **Granite + rsLoRA fails to converge** within 1 epoch (24.5% feasibility, 215% mean gap). The LoRA variant of the same model trains cleanly to 86.5% feasibility.
- **Qwen2 + LoRA collapses** (1% feasibility) while Qwen2 + rsLoRA reaches 50%. The Qwen2 LoRA learning curve does not show divergence; the failure is at generation time on larger problem sizes (the per-size breakdown in `metrics_lora_qwen2.json` shows 970% mean gap on 10×9 instances).

---

## GRPO Continuation — pushing beyond the SFT baseline

After fine-tuning, we run **GRPO** (Group Relative Policy Optimization, TRL 0.15.2) on top of the LLaMA 3.1-8B + rsLoRA checkpoint to push feasibility further — especially out-of-distribution.

### GRPO training configuration (V3–V6)

| Hyperparameter | Value |
|---|---|
| Base policy | LLaMA 3.1-8B + rsLoRA SFT (ckpt-9800) |
| K samples per prompt | 4 |
| Rollout temperature | **0.7** |
| Rollout `top_p` / `max_new_tokens` | `0.95` / `4096` |
| Learning rate | 5e-6 |
| KL coefficient (β) | 0.05 |
| Gradient accumulation | 4 (effective batch = 4 prompts/update) |
| Warmup steps | 20 |
| Max grad norm | 1.0 |
| Length control | advantage masking, `OVERLEN_FACTOR = 2.0` (V5 onward) |

Eval generation uses `temperature=0.1` (see Evaluation protocol above) — only **training rollouts** use `0.7`. Full settings: [`grpo_jssp/config.py`](grpo_jssp/config.py).

Six prior GRPO variants (V1–V4.2) all **collapsed during training** via two repeating mechanisms: (a) absorbing-state collapse (`reward_std → 0` → no gradient) and (b) length escape (`completion_length` drifts → grad spike → policy leaves the SFT basin). The full design history is in [`grpo_jssp/EXPERIMENT_NOTES.md`](grpo_jssp/EXPERIMENT_NOTES.md).

**V5** is the first GRPO run that completes 500 training steps without collapse, and the first to **strictly improve over the SFT baseline** on every in-distribution metric. The design pairs V4's 7-component hybrid reward with a hard length-control mechanism applied at the **advantage** level (not at the reward level — soft length shaping in V2 was actively harmful):

```python
class LengthControlledGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs):
        out = super()._prepare_inputs(inputs)
        clen = out["completion_mask"].sum(dim=1).float()
        gold_est = 12.5 * n_ops + 50
        over = clen > (2.0 * gold_est)
        out["advantages"] = torch.where(over, torch.zeros_like(adv), adv)
        return out
```

A sample whose completion length exceeds `2 × gold_est` has its advantage zeroed, so it contributes no gradient — neither rewarded nor penalized. The reward function sees no length term at all; length control is fully decoupled and pluggable to any reward function that exposes a per-sample size proxy.

### Results: V5 vs SFT baseline

| Run | Split | Feasibility | Median gap | Mean gap | Routing violations |
|---|---|---:|---:|---:|---:|
| SFT baseline (rsLoRA ckpt-9800) | SM test (200) | 95.0% | 3.87% | 6.73% | 100 |
| **GRPO V5 (500 steps)** | SM test (200) | **97.0%** | **3.02%** | **5.36%** | **9** |
| SFT baseline (rsLoRA ckpt-9800) | OOD held-out (18) | 50.0% | 10.91% | 20.12% | 5 |
| **GRPO V5 (500 steps)** | OOD held-out (18) | **66.7%** | **10.16%** | 17.44% | 3 |

Bold = V5 beats baseline. OOD held-out = 18 FT+LA instances never seen during SFT or GRPO training. SM test = 2% held-out split (seed=42) of the StarJob SM data.

**Highlights:**
- **OOD feasibility +16.7pp** (9/18 → 12/18) — the headline result. GRPO is generalizing beyond the training distribution.
- **SM routing violations −91%** (100 → 9) — V5 also teaches format/routing discipline in-distribution.
- **No collapse:** at step 500 reward 6.92, reward_std 0.033, grad_norm 0.59, completion_length 487, KL 0.31 — all healthy. V4 and V4.2 had collapsed at steps 340 and 315 respectively under the same data and hyperparameters.
- The 6 remaining OOD failures (ft20, la06–la10) are the same LA 5×5 family that fails across **every** GRPO variant, including SFT — likely an OOD shape the model never saw at SFT, not a V5-specific regression.

Full eval JSONs: [`grpo_jssp/eval_results/full_hybrid_lc_n2000_v5_{ood,sm}.json`](grpo_jssp/eval_results/). Trained adapter: `grpo_jssp/runs/full_hybrid_lc_n2000_v5/final_adapter/`.

### V6 — V1 stratified reward + length control (ablation)

V6 keeps V5's advantage-level length control but swaps the V4 hybrid reward (range [−1, +7]) for the original **V1 stratified reward** (range ≈ [−1, +1]). The motivation is to test whether the milder, narrower reward range still benefits from length control, and how it shapes the trade-off between feasibility and makespan tightness.

V6 was budgeted for 500 steps but stopped at **checkpoint-400** due to GPU memory pressure (each resume crashed faster as the model produced longer completions). Training dynamics through step 400 were healthy; ck-400 represents 80% of the planned budget.

| Run | Split | Feasibility | Median gap | Mean gap |
|---|---|---:|---:|---:|
| SFT baseline | SM test (200) | 95.0% | 3.87% | 6.73% |
| GRPO V5 (500 steps) | SM test (200) | **97.0%** | 3.02% | **5.36%** |
| GRPO V6 ck-400 | SM test (200) | 95.0% | **2.97%** | 5.72% |
| SFT baseline | OOD held-out (18) | 50.0% | 10.91% | 20.12% |
| GRPO V5 (500 steps) | OOD held-out (18) | **66.7%** | 10.16% | 17.44% |
| GRPO V6 ck-400 | OOD held-out (18) | 61.1% | 10.91% | **11.30%** |

#### Full metric breakdown — all runs × both splits

Violation counts are summed across all instances in the split. Lower is better for every column except feasibility.

**SM test (200 samples):**

| Run | Feas % | n_feasible | Mean gap | Median gap | missing_op | routing_order | machine_cap | timing | precedence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT baseline | 95.0% | 190/200 | 6.73% | 3.87% | 5 | 100 | 389 | 40 | 70 |
| GRPO V5 (500 steps) | **97.0%** | **194/200** | **5.36%** | 3.02% | 4 | **9** | **1** | **3** | **5** |
| GRPO V6 ck-400 | 95.0% | 190/200 | 5.72% | **2.97%** | **2** | 114 | 415 | 51 | 80 |

**OOD held-out (18 FT+LA instances):**

| Run | Feas % | n_feasible | Mean gap | Median gap | missing_op | routing_order | machine_cap | timing | precedence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT baseline | 50.0% | 9/18 | 20.12% | 10.91% | 177 | 5 | 30 | 14 | 1 |
| GRPO V5 (500 steps) | **66.7%** | **12/18** | 17.44% | 10.16% | 177 | 3 | **0** | **0** | **0** |
| GRPO V6 ck-400 | 61.1% | 11/18 | **11.30%** | 10.91% | 177 | **3** | 31 | 10 | 0 |

Bold = best in column (per split). `missing_op = 177` on OOD is identical across all three runs — a shared structural failure on the same OOD instances (the LA 5×5 family that fails across every variant including SFT, not a V6 regression).

**V6 vs V5 — trade-off, not dominance:**
- **V6 OOD mean gap 11.30% vs V5 17.44%** (−35%): when feasible, V6's makespans are substantially tighter on out-of-distribution instances.
- **V5 OOD feasibility 12/18 vs V6 11/18** (+1 sample): V5's broader reward range pushes harder on the feasibility constraint.
- **SM violation profile differs**: V5 essentially eliminates routing/capacity violations on SM (mc=1, ro=9, tc=3, pr=5); V6 keeps them near baseline scale (mc=415, ro=114, tc=51, pr=80). The narrower V1 reward range provides less gradient pressure on constraint satisfaction.

**Interpretation:** V5's wider reward range optimizes aggressively against violations at the cost of looser makespans. V6's stratified reward retains more of the baseline's solution-quality profile while still improving OOD generalization. Neither variant strictly dominates — V5 is the safer choice when feasibility is paramount; V6 produces shorter feasible schedules on OOD distributions.

Full eval JSONs: [`grpo_jssp/eval_results/full_stratified_lc_n2000_v6_ck400_{ood,sm}.json`](grpo_jssp/eval_results/). Trained adapter: `grpo_jssp/runs/full_stratified_lc_n2000_v6/checkpoint-400/`.

---

## Reproducibility

### Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
pip install -r requirements.txt
```

### Train

```bash
# rsLoRA (default)
python train_llama_3.py
python train_granite_8b.py
python train_ministral_8b.py
python train_qwen2_7b.py

# Plain LoRA
python train_llama_3.py --use_rslora False
# ...etc.
```

### Evaluate

```bash
./run_eval_lora.sh      # all 4 LoRA models
./run_eval_rslora.sh    # all 4 rsLoRA models
```

### Plot

```bash
python plot_learning_curves.py             # LoRA-only
python plot_learning_curves_rslora.py      # rsLoRA-only
python plot_learning_curves_combined.py    # merged + long-form CSV
```

---

## Project Structure

Files grouped by objective.

### Data

- `prepare_dataset.py` — Build train/test splits from raw Starjob.
- `sample_output.py` — Inspect sample model outputs.
- `data/starjob_train_sm.jsonl` — Small+medium training/eval split (LFS-tracked).

### Training (shared LoRA / rsLoRA)

Source code (`--use_rslora` flag, default `True`):
- `train_llama_3.py`, `train_granite_8b.py`, `train_ministral_8b.py`, `train_qwen2_7b.py`
- `run_qwen_granite.sh`, `run_granite_autoresume.sh`

Output checkpoints (gitignored):
- LoRA: `output_alpha32_r32_seq8192_b1_ga8_ep1/`, `output_granite8b_alpha32_r32_seq8192_b1_ga8_ep1/`, `output_ministral8b_alpha32_r32_seq8192_b1_ga8_ep1/`, `output_qwen2_7b_alpha32_r32_seq8192_b1_ga8_ep1/`
- rsLoRA: `output_*_rslora_*/`

### LoRA Evaluation

- `eval_lora.py`, `run_eval_lora.sh` — Unified eval pipeline (n=200).
- `eval_llama.py`, `eval_granite.py`, `eval_ministral.py`, `eval_qwen2.py` — Earlier per-model evals (legacy).
- `eval_makespan.py`, `eval_benchmarks.py`, `eval_llama_benchmarks.py`, `run_all_benchmarks.sh` — Benchmark suite.
- `extract_losses.py`, `plot_learning_curves.py`, `plot_benchmarks.py`
- `metrics_lora_{llama,granite,ministral,qwen2}.json`
- `loss_curves/all_losses.json`, `loss_curves/{model}_{train,eval}.csv`
- `loss_curves/learning_curves.png`, `benchmark_results.png`

### rsLoRA Evaluation

- `eval_rslora.py`, `run_eval_rslora.sh`
- `eval_rslora_benchmarks.py`, `run_rslora_benchmarks.sh`
- `extract_losses_rslora.py`, `plot_learning_curves_rslora.py`
- `metrics_rslora_{llama,granite,ministral,qwen2}.json`, `metrics_rslora_benchmarks_*.json`
- `loss_curves/all_losses_rslora.json`, `loss_curves/learning_curves_rslora.png`

### LoRA vs rsLoRA Comparison

- `plot_learning_curves_combined.py` — Overlay plot + long-form CSV.
- `comparison_lora_vs_rslora.json` — Head-to-head table.
- `loss_curves/all_losses_long.csv` — `model, method, phase, step, loss`.
- `loss_curves/learning_curves_combined.png`

### Misc

- `compute_detailed_metrics.py`, `compute_gap.py` — Metric helpers.
- `make_slides.py`, `starjob_intro_methodology.pptx` — Presentation.
- `requirements.txt`

---

## Dataset

Full 130k instances on Hugging Face: [henri24/Starjob](https://huggingface.co/datasets/henri24/Starjob). Each entry has:

| Field | Type | Description |
|---|---|---|
| `num_jobs` | int | Number of jobs (≤ 16) |
| `num_machines` | int | Number of machines (≤ 16) |
| `instruction` | str | Natural-language problem statement |
| `input` | str | Per-job machine routing and processing times |
| `output` | str | Reference schedule with start/end timestamps |
| `matrix` | object | OR-Tools makespan + matrix-form solution |

For this experiment we use a small/medium subset checked into the repo at `data/starjob_train_sm.jsonl`.

---

## License

Dataset: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
