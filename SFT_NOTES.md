# SFT — Catatan Eksperimen

Supervised fine-tuning rsLoRA pada LLaMA 3.1-8B untuk JSSP. Output adapter
ini dipakai sebagai **starting point GRPO** (lihat `grpo_jssp/EXPERIMENT_NOTES.md`).

## Setup

- **Base model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit ~5.4 GB)
- **Adapter dir:** `output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/`
- **Method:** rsLoRA r=32, α=32, sequence length 8192
- **Training script:** `train_llama_3.py`
- **Train set:** `data/starjob_train_sm.jsonl` (StarJob SM, jobs/machines ≤10)
- **OOD eval set:** `data/benchmarks/jobshop1.txt` → 18 instance FT+LA dengan BKS hardcoded

## Pemilihan checkpoint

| Checkpoint | Eval loss | OOD feasibility (T=0.1, do_sample) | Catatan |
|---|---|---|---|
| **checkpoint-9800** | **terendah** | 9/18 = 50.0% | Match baseline yang dilaporkan di `metrics_rslora_llama.json` (190/200 SM). **Dipakai sebagai SFT base GRPO.** |
| checkpoint-13230 | final epoch | 12/18 = 66.7% | Eksperimen sebelumnya menunjukkan lebih baik di OOD, **tapi tidak match baseline yang dilaporkan**. Eval loss bukan proxy ideal untuk OOD generalization di JSSP. |

Pilihan akhir: **checkpoint-9800** untuk konsistensi dengan baseline reported.

## Baseline eval (checkpoint-9800, T=0.1 + do_sample=True)

Re-run via pipeline GRPO dengan setup yang match SFT eval lama (held-out 2% test split, seed=42, best checkpoint dari `trainer_state`):

| Split | Feasibility | Mean gap | Median gap | Failure dominan |
|---|---|---|---|---|
| **SM (200, held-out test split)** | **190/200 = 95.0%** | 6.7% | 3.9% | capacity (64.4%) |
| **OOD (18 FT+LA)** | **9/18 = 50.0%** | 20.1% | 10.9% | missing/truncation (78%) |

Output: `grpo_jssp/eval_results/baseline_sft_{sm,ood}.json`.

## Detail eval OOD (2026-05-15, checkpoint-13230, T=0.1+do_sample=True)

**Hasil: 12/18 feasible = 66.7% | mean gap-to-BKS = 13.1% | median gap = 12.5%**

| Instance | Hasil | cmax | BKS | gap |
|---|---|---|---|---|
| ft06 | OK | 58 | 55 | +5.5% |
| ft10 | OK | 1340 | 930 | +44.1% |
| ft20 | FAIL | 867 | 1165 | viol=50 |
| la01 | OK | 770 | 666 | +15.6% |
| la02 | OK | 684 | 655 | +4.4% |
| la03 | OK | 695 | 597 | +16.4% |
| la04 | OK | 616 | 590 | +4.4% |
| la05 | OK | 593 | 593 | **+0.0%** |
| la06 | FAIL | 737 | 926 | viol=25 |
| la07 | FAIL | 628 | 890 | viol=25 |
| la08 | FAIL | 645 | 863 | viol=25 |
| la09 | FAIL | 688 | 951 | viol=25 |
| la10 | FAIL | 683 | 958 | viol=25 |
| la16 | OK | 1054 | 945 | +11.5% |
| la17 | OK | 891 | 784 | +13.6% |
| la18 | OK | 1000 | 848 | +17.9% |
| la19 | OK | 955 | 842 | +13.4% |
| la20 | OK | 991 | 902 | +9.9% |

**Pola failure:** semua gagal di instance >10×10 (la06–10 = 15×5; ft20 = 20×5). **Semua 175 violation adalah `missing_op_count`** — routing/capacity/timing/precedence = 0.

## Pola failure & implikasi untuk GRPO

- **SM**: 64% violation adalah `machine_capacity` → reward stratified akan banyak menekan tipe ini.
- **OOD**: 78% violation adalah `missing_op_count` (truncation/early-stop di instance >10×10) → reward yang fit SM tidak menyentuh failure mode utama OOD.
- **Konsekuensi**: GRPO di SM kemungkinan **memperbaiki SM** lebih dari **memperbaiki OOD**. Ini batasan struktural — perlu disebut di analisis hasil.

## Eval scripts terkait

| Script | Untuk | Output |
|---|---|---|
| `eval_rslora.py` | SM test split | `metrics_rslora_llama.json` |
| `eval_rslora_benchmarks.py` | OOD benchmarks (FT+LA) | `metrics_rslora_benchmarks_llama.json` |
| `grpo_jssp/_run_eval.py` | Sama tapi via pipeline GRPO (saat re-eval SFT base) | `grpo_jssp/eval_results/baseline_sft_{sm,ood}.json` |

## Catatan leakage bobot (warning sebelum GRPO)

Bobot stratified awal yang sempat dipakai (`missing=0.09, routing=0.32, capacity=0.28, timing=0.26, precedence=0.06`) diturunkan dari distribusi violation OOD versi lama (pool 4 model). Untuk model LLaMA-rsLoRA sekarang, distribusi violation real OOD jauh berbeda (100% missing).

**Bobot harus di-recompute dari SM** (tidak boleh dari OOD karena leakage test set ke training signal). Bobot final (di-recompute dari SM, 233 violation di 200 sampel) ada di `grpo_jssp/EXPERIMENT_NOTES.md`.

---

**Lanjutan eksperimen GRPO**: lihat `grpo_jssp/EXPERIMENT_NOTES.md` untuk run V1, V2, V3 dan analisis collapse/patch.
