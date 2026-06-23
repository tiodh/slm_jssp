# Evaluation Metrics — Paper Summary
Generated from `scripts/build_paper_metrics.py`. Raw data in `reports/paper_*.csv`.

**Splits:** **OOD** = 18 instances FT+LA (out-of-distribution real benchmarks); **SM** = 200 instances Starjob test (held-out 2%); **Starjob test** = full test split (n=50 subset or n=200 full).

**Variant labels (Starjob test, cat B):**
- `lora_n50`: LoRA fine-tune evaluated by training script at end of training (n=50 subset, includes `eval_loss`).
- `lora`: same LoRA family re-evaluated by `eval_lora.py` on full n=200 split.
- `rslora`: rsLoRA fine-tune (`use_rslora=True`) re-evaluated by `eval_rslora.py` on n=200.

**Variant labels (OOD bench, cat C):**
- `lora_<model>`: LoRA fine-tune on FT+LA OOD (from `eval_benchmarks.py`).
- `rslora_<model>`: rsLoRA fine-tune on FT+LA OOD (from `eval_rslora_benchmarks.py`).

All baselines in this repo are adapter-based (LoRA or rsLoRA) trained via TRL `SFTTrainer`; no pure full-parameter SFT model exists.

---
## A. GRPO Evaluations (OOD = FT+LA 18 inst, SM = Starjob test 200 inst)
Per checkpoint where evaluated; final/single-eval models shown as single row.

| Model | n | OOD feas | OOD feas% | OOD gap% (mean/med) | SM feas | SM feas% | SM gap% (mean/med) |
|---|---|---|---|---|---|---|---|
| baseline_sft | 18 | 9/18 | 50.0 | 20.12 / 10.91 | 190/200 | 95.0 | 6.73 / 3.87 |
| full_hybrid_lc_n2000_v5 | 18 | 12/18 | 66.67 | 17.44 / 10.16 | 194/200 | 97.0 | 5.36 / 3.02 |
| full_lora_hybrid_lc_n2000_v5 | 18 | 12/18 | 66.67 | 17.44 / 10.16 | 194/200 | 97.0 | 5.36 / 3.02 |
| full_lora_hybrid_lc_over_n2000_v7 | 18 | 9/18 | 50.0 | 7.46 / 8.55 | 189/200 | 94.5 | 4.66 / 1.83 |
| full_lora_hybrid_n2000_v4 | 18 | 13/18 | 72.22 | 11.26 / 10.97 | 189/200 | 94.5 | 4.89 / 2.82 |
| full_lora_hybrid_n2000_v4_ckpt100 | 18 | 10/18 | 55.56 | 9.25 / 6.94 | 188/200 | 94.0 | 5.04 / 2.99 |
| full_lora_hybrid_n2000_v4_ckpt150 | 18 | 8/18 | 44.44 | 9.26 / 10.78 | 189/200 | 94.5 | 5.16 / 2.82 |
| full_lora_hybrid_n2000_v4_ckpt200 | 18 | 9/18 | 50.0 | 9.61 / 9.71 | 191/200 | 95.5 | 5.15 / 2.94 |
| full_lora_hybrid_n2000_v4_ckpt250 | 18 | 9/18 | 50.0 | 6.92 / 7.27 | 193/200 | 96.5 | 5.08 / 2.45 |
| full_lora_hybrid_n2000_v4_ckpt300 | 18 | 8/18 | 44.44 | 12.02 / 12.17 | - | - | - |
| full_lora_hybrid_n2000_v4_ckpt350 | 18 | 9/18 | 50.0 | 9.82 / 9.92 | 188/200 | 94.0 | 5.06 / 2.75 |
| full_lora_hybrid_n2000_v4_ckpt400 | 18 | 10/18 | 55.56 | 10.21 / 9.54 | 192/200 | 96.0 | 5.18 / 2.63 |
| full_lora_hybrid_n2000_v4_ckpt50 | 18 | 9/18 | 50.0 | 12.36 / 12.06 | 184/200 | 92.0 | 4.78 / 2.97 |
| full_lora_hybrid_n2000_v4_ckpt500 | 18 | 10/18 | 55.56 | 8.99 / 8.55 | 192/200 | 96.0 | 5.04 / 2.5 |
| full_lora_stratified_2000_v1 | 18 | 11/18 | 61.11 | 7.76 / 9.09 | 196/200 | 98.0 | 5.21 / 2.46 |
| full_lora_stratified_lc_n2000_v6 | 18 | 8/18 | 44.44 | 5.54 / 5.83 | - | - | - |
| full_lora_stratified_n2000_v3 | 18 | 9/18 | 50.0 | 8.25 / 6.63 | 192/200 | 96.0 | 4.89 / 1.95 |
| full_rslora_hybrid_n2000_v4_ckpt100 | 18 | 6/18 | 33.33 | 7.86 / 9.02 | 182/200 | 91.0 | 5.36 / 2.97 |
| full_rslora_hybrid_n2000_v4_ckpt150 | 18 | 6/18 | 33.33 | 11.96 / 13.47 | 186/200 | 93.0 | 5.24 / 3.04 |
| full_rslora_hybrid_n2000_v4_ckpt200 | 18 | 7/18 | 38.89 | 9.32 / 8.88 | 192/200 | 96.0 | 5.21 / 3.82 |
| full_rslora_hybrid_n2000_v4_ckpt250 | 18 | 8/18 | 44.44 | 11.1 / 9.89 | 186/200 | 93.0 | 5.38 / 3.4 |
| full_rslora_hybrid_n2000_v4_ckpt300 | 18 | 8/18 | 44.44 | 9.22 / 7.43 | 190/200 | 95.0 | 5.34 / 3.09 |
| full_rslora_hybrid_n2000_v4_ckpt350 | 18 | 6/18 | 33.33 | 6.5 / 7.17 | 186/200 | 93.0 | 5.27 / 3.4 |
| full_rslora_hybrid_n2000_v4_ckpt400 | 18 | 6/18 | 33.33 | 9.65 / 11.01 | 183/200 | 91.5 | 5.26 / 3.04 |
| full_rslora_hybrid_n2000_v4_ckpt450 | 18 | 8/18 | 44.44 | 8.66 / 9.4 | 187/200 | 93.5 | 5.32 / 3.58 |
| full_rslora_hybrid_n2000_v4_ckpt50 | 18 | 6/18 | 33.33 | 8.78 / 9.02 | 177/200 | 88.5 | 5.34 / 3.16 |
| full_stratified_lc_n2000_v6_ck400 | 18 | 11/18 | 61.11 | 11.3 / 10.91 | 190/200 | 95.0 | 5.72 / 2.97 |
| full_v1_ckpt200 | 18 | 9/18 | 50.0 | 14.29 / 10.91 | - | - | - |
| full_v1_ckpt600 | 18 | 10/18 | 55.56 | 19.27 / 16.29 | 194/200 | 97.0 | 6.08 / 2.92 |
| full_v2_ckpt200 | 18 | 5/18 | 27.78 | 9.61 / 10.91 | 175/200 | 87.5 | 5.92 / 2.55 |
| full_v3_ckpt200 | 18 | 6/18 | 33.33 | 6.55 / 8.1 | 188/200 | 94.0 | 5.55 / 3.87 |
| full_v4_2_ckpt200 | 18 | 11/18 | 61.11 | 22.47 / 17.63 | 193/200 | 96.5 | 6.52 / 3.58 |
| full_v4_ckpt200 | 18 | 8/18 | 44.44 | 8.32 / 7.93 | 187/200 | 93.5 | 5.17 / 3.58 |
| pilot_hybrid_lc_n100_v5 | 18 | 9/18 | 50.0 | 24.68 / 10.91 | 190/200 | 95.0 | 6.84 / 3.91 |
| pilot_hybrid_n100_v4_2 | 18 | 10/18 | 55.56 | 13.31 / 10.03 | 192/200 | 96.0 | 6.28 / 3.11 |
| pilot_lora_v1 | 18 | 7/18 | 38.89 | 9.51 / 9.15 | 186/200 | 93.0 | 5.12 / 3.21 |
| pilot_lora_v2 | 18 | 8/18 | 44.44 | 12.18 / 10.48 | 187/200 | 93.5 | 5.91 / 4.15 |
| pilot_lora_v3 | 18 | 6/18 | 33.33 | 8.81 / 8.51 | 186/200 | 93.0 | 5.21 / 2.75 |
| pilot_lora_v4 | 18 | 12/18 | 66.67 | 12.07 / 12.99 | 185/200 | 92.5 | 5.35 / 3.64 |
| pilot_lora_v5 | 18 | 7/18 | 38.89 | 9.47 / 8.8 | 187/200 | 93.5 | 5.55 / 3.62 |
| pilot_lora_v6 | 18 | 8/18 | 44.44 | 11.36 / 10.88 | 181/200 | 90.5 | 5.48 / 3.64 |
| pilot_stratified_100 | 18 | 11/18 | 61.11 | 39.28 / 17.47 | 187/200 | 93.5 | 6.11 / 3.73 |
| pilot_stratified_100_v2 | 18 | 7/18 | 38.89 | 13.21 / 9.83 | 190/200 | 95.0 | 6.95 / 3.87 |
| pilot_stratified_lc_n100_v6 | 18 | 8/18 | 44.44 | 12.51 / 10.03 | 191/200 | 95.5 | 6.77 / 4.17 |
| pilot_stratified_n100_v3 | 18 | 8/18 | 44.44 | 10.17 / 10.55 | 192/200 | 96.0 | 6.81 / 3.75 |
| pilot_uniform_100 | 18 | 10/18 | 55.56 | 30.8 / 11.82 | 193/200 | 96.5 | 6.49 / 4.56 |
| strict_v1_pilot | 18 | 8/18 | 44.44 | 12.43 / 11.52 | - | - | - |
| strict_v2_pilot | 18 | 11/18 | 61.11 | 13.21 / 8.81 | - | - | - |
| strict_v3_pilot | 18 | 9/18 | 50.0 | 12.94 / 12.54 | - | - | - |
| strict_v4_pilot | 18 | 6/18 | 33.33 | 9.57 / 8.04 | - | - | - |

## B. Adapter-tuned Baselines on Starjob Test (LoRA / rsLoRA via TRL SFTTrainer)

### B.1 Overall (across all sizes)
| Model | Variant | n | exact% | gap≤5% | gap≤10% | gap≤20% | mean_gap% | median_gap% | feas% | eval_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| granite | lora | 200 | 35.0 | None | None | None | 41.03 | 3.76 | 87.0 | None |
| llama | lora | 200 | 34.0 | None | None | None | 6.44 | 3.41 | 95.5 | None |
| ministral | lora | 200 | 30.5 | None | None | None | 13.7 | 5.57 | 93.0 | None |
| qwen2 | lora | 200 | 3.0 | None | None | None | 59.28 | 32.27 | 1.0 | None |
| Qwen2-7B | lora_n50 | 50 | 4.0 | 20.0 | 26.0 | 38.0 | 63.74 | 30.9 | 100.0 | 0.3481 |
| granite_3_2_8b | lora_n50 | 50 | 26.0 | 40.0 | 58.0 | 72.0 | 186.1 | 6.2 | 100.0 | 0.3356 |
| llama_3_1_8b | lora_n50 | 50 | 22.0 | 42.0 | 68.0 | 76.0 | 44.06 | 5.6 | 100.0 | 0.4018 |
| ministral_8b | lora_n50 | 50 | 28.0 | 44.0 | 58.0 | 72.0 | 48.66 | 6.25 | 100.0 | 0.3448 |
| granite | rslora | 200 | 5.5 | None | None | None | 214.48 | 40.72 | 22.0 | None |
| llama | rslora | 200 | 36.0 | None | None | None | 12.69 | 5.3 | 95.0 | None |
| ministral | rslora | 200 | 24.0 | None | None | None | 34.72 | 9.68 | 64.0 | None |
| qwen2 | rslora | 200 | 25.0 | None | None | None | 34.39 | 7.33 | 50.0 | None |

### B.2 Per-size breakdown (small/medium/large)
| Model | Variant | Size | n | exact% | mean_gap% | feas% |
|---|---|---|---|---|---|---|
| granite | lora | small | 41 | 78.05 | 2.09 | 100.0 |
| granite | lora | medium | 159 | 23.9 | 51.06 | 83.65 |
| llama | lora | small | 41 | 78.05 | 1.43 | 97.56 |
| llama | lora | medium | 159 | 22.64 | 7.74 | 94.97 |
| ministral | lora | small | 41 | 65.85 | 3.41 | 97.56 |
| ministral | lora | medium | 159 | 21.38 | 16.35 | 91.82 |
| qwen2 | lora | small | 41 | 14.63 | 14.41 | 4.88 |
| qwen2 | lora | medium | 159 | 0.0 | 70.85 | 0.0 |
| Qwen2-7B | lora_n50 | small | 9 | 22.2 | 7.03 | None |
| Qwen2-7B | lora_n50 | medium | 29 | 0.0 | 65.8 | None |
| Qwen2-7B | lora_n50 | large | 12 | 0.0 | 101.27 | None |
| granite_3_2_8b | lora_n50 | small | 9 | 55.6 | 1.79 | None |
| granite_3_2_8b | lora_n50 | medium | 29 | 27.6 | 195.79 | None |
| granite_3_2_8b | lora_n50 | large | 12 | 0.0 | 300.9 | None |
| llama_3_1_8b | lora_n50 | small | 9 | 55.6 | 2.3 | None |
| llama_3_1_8b | lora_n50 | medium | 29 | 20.7 | 4.77 | None |
| llama_3_1_8b | lora_n50 | large | 12 | 0.0 | 170.34 | None |
| ministral_8b | lora_n50 | small | 9 | 55.6 | 3.43 | None |
| ministral_8b | lora_n50 | medium | 29 | 31.0 | 13.52 | None |
| ministral_8b | lora_n50 | large | 12 | 0.0 | 167.48 | None |
| granite | rslora | small | 41 | 26.83 | 17.14 | 58.54 |
| granite | rslora | medium | 159 | 0.0 | 265.36 | 12.58 |
| llama | rslora | small | 41 | 78.05 | 3.29 | 100.0 |
| llama | rslora | medium | 159 | 25.16 | 15.11 | 93.71 |
| ministral | rslora | small | 41 | 51.22 | 5.24 | 97.56 |
| ministral | rslora | medium | 159 | 16.98 | 42.33 | 55.35 |
| qwen2 | rslora | small | 41 | 60.98 | 2.86 | 90.24 |
| qwen2 | rslora | medium | 159 | 15.72 | 42.52 | 39.62 |

## C. OOD Benchmarks (FT+LA real instances, 18 total)
| Model | n | feasible | feas% | all_gap% | feas_gap% | prec | route | time | mc | miss_op | cost ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lora_granite | 18 | 6 | 33.33 | 202.4 | 6.11 | 30 | 121 | 19 | 162 | 22 | None |
| lora_llama | 18 | 10 | 55.56 | 30.47 | 9.1 | 0 | 5 | 21 | 138 | 25 | None |
| lora_ministral | 18 | 6 | 33.33 | 124.1 | 6.43 | 14 | 45 | 38 | 177 | 78 | None |
| lora_qwen2 | 18 | 0 | 0.0 | 220.81 | None | 133 | 818 | 714 | 377 | 163 | None |
| openai/gpt-4o | 18 | 0 | 0.0 | 3.57 | None | 134 | 378 | 0 | 435 | 0 | 0.1865 |
| openai/gpt-4o-mini | 18 | 0 | 0.0 | -66.96 | None | 41 | 217 | 0 | 1 | 1090 | 0.0033 |
| openai/gpt-5 | 18 | 1 | 5.56 | 47.14 | 449.35 | 4 | 54 | 0 | 334 | 0 | 0.1799 |
| openai/o3 (medium) | 18 | 13 | 72.22 | 234.74 | 234.96 | 0 | 0 | 0 | 0 | 286 | 9.1559 |
| openai/o3-mini (high) | 18 | 2 | 11.11 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 1236 | 1.9674 |
| openai/o3-mini (medium) | 18 | 5 | 27.78 | 148.23 | 243.34 | 2 | 62 | 3 | 1 | 676 | 0.9164 |
| rslora_granite | 18 | 0 | 0.0 | 384.75 | None | 55 | 312 | 312 | 169 | 69 | None |
| rslora_llama | 18 | 8 | 44.44 | 39.65 | 10.73 | 2 | 1 | 31 | 362 | 175 | None |
| rslora_ministral | 18 | 6 | 33.33 | 165.62 | 20.56 | 18 | 169 | 345 | 527 | 183 | None |
| rslora_qwen2 | 18 | 1 | 5.56 | 462.38 | 23.64 | 98 | 320 | 292 | 229 | 250 | None |
