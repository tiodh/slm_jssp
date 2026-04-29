
# Starjob Dataset designed to train LLMs on JSSP

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/henri24/Starjob)

Dataset is available at [Hugging Face](https://huggingface.co/datasets/henri24/Starjob)

This repository fine-tunes 4 small/medium open-weight LLMs (LLaMA 3.1 8B, Granite 3.2 8B, Ministral 8B, Qwen2 7B) on JSSP (Job-Shop Scheduling Problem) makespan prediction using both **LoRA** and **rsLoRA** adapters, then evaluates them head-to-head.

## Dataset Overview

**Dataset Name:** starjob130k.json
**Number of Entries:** 130,000  
**Number of Fields:** 5  

## Fields Description

1. **num_jobs**
   - **Type:** int64
   - **Number of Unique Values:** 16
   
2. **num_machines**
   - **Type:** int64
   - **Number of Unique Values:** 16
   
3. **instruction**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Initial description of the problem detailing the number of jobs and machines involved.**
     
4. **input**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Description of the problem in LLM format**

5. **output**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Solution in LLM format:** 130,000

6. **matrix**
   - **Type:** object
   - **Number of Unique Values:** 130,000
   - **Input problem OR-Tool makspan and solution in Matrix format** 

   
## Usage

This dataset can be used for training LLMs for job-shop scheduling problems (JSSP). Each entry provides information about the number of jobs, the number of machines, and other relevant details formatted in natural language.


# Setting Up Your Python Environment

Follow these instructions to create a virtual environment and install the necessary libraries.

## Step 1: Create a Virtual Environment

```bash
python3 -m venv llm_env
```

Activate the Virtual Environment
After creating the virtual environment, activate it using the following command:

On Windows
```bash
.\llm_env\Scripts\activate
```

On macOS and Linux
```bash
source llm_env/bin/activate
```

# Install the Required Libraries
```bash
pip install -r requirements.txt
```

# Training
Make sure to put dataset.json under data directory

```bash
python train_llama_3.py
```

---

# Project Structure

Files grouped by objective: **Data**, **Training (shared)**, **LoRA evaluation**, **RSLoRA evaluation**, **LoRA vs RSLoRA comparison**, and **Misc**.

## Data

Source code
- `prepare_dataset.py` — Build train/test splits from raw Starjob.
- `sample_output.py` — Inspect sample model outputs.

Data files
- `data/starjob_train_sm.jsonl` — Small+medium training/eval split (LFS-tracked).

## Training (shared LoRA / RSLoRA)

The training scripts accept `--use_rslora` (default `True`). Set `--use_rslora False` for plain LoRA.

Source code
- `train_llama_3.py` — Train LLaMA 3.1 8B.
- `train_granite_8b.py` — Train Granite 3.2 8B.
- `train_ministral_8b.py` — Train Ministral 8B.
- `train_qwen2_7b.py` — Train Qwen2 7B.
- `run_qwen_granite.sh`, `run_granite_autoresume.sh` — Training launchers.

Logs
- `train.log`, `train_granite.log`, `train_ministral.log`, `train_qwen2.log`
- `run_qwen_granite.log`, `run_rslora.log`

Output checkpoints (gitignored)
- LoRA: `output_alpha32_r32_seq8192_b1_ga8_ep1/`, `output_granite8b_alpha32_r32_seq8192_b1_ga8_ep1/`, `output_ministral8b_alpha32_r32_seq8192_b1_ga8_ep1/`, `output_qwen2_7b_alpha32_r32_seq8192_b1_ga8_ep1/`
- RSLoRA: `output_llama8b_rslora_*/`, `output_granite8b_rslora_*/`, `output_ministral8b_rslora_*/`, `output_qwen2_7b_rslora_*/`

## LoRA Evaluation

Source code
- `eval_lora.py` — Unified eval pipeline for all 4 LoRA models (n=200, starjob_sm, feasibility validator). Mirror of `eval_rslora.py`.
- `eval_llama.py`, `eval_granite.py`, `eval_ministral.py`, `eval_qwen2.py` — Earlier per-model eval scripts (legacy).
- `eval_makespan.py`, `eval_benchmarks.py`, `eval_llama_benchmarks.py` — Benchmark suites.
- `run_eval_lora.sh` — Runs `eval_lora.py` for all 4 models in sequence.
- `run_all_benchmarks.sh` — Runs the benchmark eval.
- `extract_losses.py` — Pull `log_history` from each LoRA `trainer_state.json` into `loss_curves/all_losses.json`.
- `plot_learning_curves.py` — LoRA-only learning curve plot.
- `plot_benchmarks.py` — Benchmark bar plot.

Logs
- `eval_lora_llama.log`, `eval_lora_granite.log`, `eval_lora_ministral.log`, `eval_lora_qwen2.log`
- `eval_llama.log`, `eval_granite.log`, `eval_ministral.log`, `eval_ministral_full.log`, `eval_qwen2.log`
- `eval_bench_*.log`, `eval_llama_benchmarks.log`, `run_all_benchmarks.log`, `run_eval_lora.log`

JSON / CSV
- `metrics_lora_llama.json`, `metrics_lora_granite.json`, `metrics_lora_ministral.json`, `metrics_lora_qwen2.json` — Unified pipeline (n=200).
- `metrics_llama_3_1_8b.json`, `metrics_granite_3_2_8b.json`, `metrics_ministral_8b.json`, `metrics_qwen2_7b.json` — Legacy per-model.
- `metrics_llama_benchmarks.json`, `metrics_benchmarks_llama.json`, `metrics_benchmarks_granite.json`, `metrics_benchmarks_ministral.json`, `metrics_benchmarks_qwen2.json`
- `loss_curves/all_losses.json`
- `loss_curves/llama_3_1_8b_train.csv`, `llama_3_1_8b_eval.csv`, `granite_3_2_8b_train.csv`, `granite_3_2_8b_eval.csv`, `ministral_8b_train.csv`, `ministral_8b_eval.csv`, `qwen2_7b_train.csv`, `qwen2_7b_eval.csv`

Images
- `loss_curves/learning_curves.png` — LoRA train + eval loss curves.
- `benchmark_results.png` — LoRA benchmark bar chart.

## RSLoRA Evaluation

Source code
- `eval_rslora.py` — Unified eval pipeline for all 4 RSLoRA models (n=200, starjob_sm, feasibility validator).
- `eval_rslora_benchmarks.py` — Benchmark suite for RSLoRA.
- `run_eval_rslora.sh` — Runs `eval_rslora.py` for all 4 models.
- `run_all_rslora.sh` — RSLoRA training/eval launcher.
- `run_rslora_benchmarks.sh` — Run benchmark suite.
- `extract_losses_rslora.py` — Pull `log_history` from each RSLoRA `trainer_state.json` into `loss_curves/all_losses_rslora.json`.
- `plot_learning_curves_rslora.py` — RSLoRA-only learning curve plot.

Logs
- `eval_rslora_llama.log`, `eval_rslora_granite.log`, `eval_rslora_granite_retry.log`, `eval_rslora_granite_retry2.log`, `eval_rslora_ministral.log`, `eval_rslora_qwen2.log`
- `eval_rslora_bench_*.log`, `run_eval_rslora.log`

JSON
- `metrics_rslora_llama.json`, `metrics_rslora_granite.json`, `metrics_rslora_ministral.json`, `metrics_rslora_qwen2.json`
- `metrics_rslora_benchmarks_llama.json`, `metrics_rslora_benchmarks_granite.json`, `metrics_rslora_benchmarks_ministral.json`, `metrics_rslora_benchmarks_qwen2.json`
- `loss_curves/all_losses_rslora.json`

Images
- `loss_curves/learning_curves_rslora.png` — RSLoRA train + eval loss curves.

## LoRA vs RSLoRA Comparison

Source code
- `plot_learning_curves_combined.py` — Overlay LoRA (solid) vs RSLoRA (dashed) and dump long-form CSV.

JSON / CSV
- `comparison_lora_vs_rslora.json` — Head-to-head metrics table (n=200, identical pipeline).
- `loss_curves/all_losses_long.csv` — Long-form `model, method, phase, step, loss` for all 4 models × {LoRA, RSLoRA}.

Images
- `loss_curves/learning_curves_combined.png` — Side-by-side merged train + eval loss.

## Misc

- `compute_detailed_metrics.py`, `compute_gap.py` — Metric helpers (gap %, feasibility tally).
- `make_slides.py`, `starjob_intro_methodology.pptx` — Presentation deck.
- `requirements.txt` — Python dependencies.
- `eval.log` — Generic eval log.

---

## License

This dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). For more details, see the [license description](https://creativecommons.org/licenses/by-sa/4.0/). The dataset will remain accessible for an extended period.
