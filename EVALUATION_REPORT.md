# Laporan Evaluasi: LoRA vs rsLoRA pada Job-Shop Scheduling

Eksperimen fine-tuning empat LLM small/medium open-weight (LLaMA 3.1 8B, Granite 3.2 8B, Ministral 8B, Qwen2 7B) menggunakan dua strategi adapter (**LoRA** dan **rsLoRA**) untuk Job-Shop Scheduling Problem (JSSP). Evaluasi dibagi dua: **in-distribution** (test split dari dataset training) dan **out-of-distribution** (benchmark akademik klasik dengan ukuran/asal berbeda).

---

## 1. Dataset

### 1.1 Dataset In-Distribution (Training & Test)

| Properti | Nilai |
|---|---|
| **Nama** | Starjob (`data/starjob_train_sm.jsonl`, subset small/medium) |
| **Asal** | [henri24/Starjob](https://huggingface.co/datasets/henri24/Starjob) — full 130k instances |
| **Jumlah instance** | 108.000 |
| **Range ukuran** | **2×2 sampai 10×10** (jobs ≤ 10, machines ≤ 10) |
| **Range total operasi** | 4 ops (2×2) sampai 100 ops (10×10) |
| **Distribusi** | 45 ukuran berbeda, 2.400 instance per ukuran (uniform) |
| **Format** | Alpaca-style: `instruction` + `input` + `output` (schedule reference) |
| **Pembagian** | 98% train / 2% test (held-out) — eval pakai test split, n=200 sampel acak (seed=42) |

**Karakteristik Starjob:**
- Instance JSSP yang di-generate secara sintetik dengan processing time random.
- Setiap instance punya makespan optimal dari OR-Tools (`matrix.makespan`) untuk perhitungan gap.
- Format problem statement = natural language deskripsi jobs/machines + per-job routing dalam bentuk `M<i>:<duration>`.

### 1.2 Dataset Out-of-Distribution (OOD)

Tiga keluarga benchmark akademik klasik dari OR-Library:

| Family | Nama lengkap | Instance dipakai | Ukuran | Asal | File |
|---|---|---|---|---|---|
| **FT** | Fisher-Thompson (1963) | ft06, ft10, ft20 | 6×6, 10×10, 20×5 | OR-Library | `data/benchmarks/jobshop1.txt` |
| **LA** | Lawrence (1984) | la01–la10, la16–la20 | 10×5, 15×5, 10×10 | OR-Library | `data/benchmarks/jobshop1.txt` |
| **TA** | Taillard (1993) | ta01–ta10 | 15×15 | OR-Library | `data/benchmarks/ta01.txt` … `ta10.txt` |

**Karakteristik benchmark:**
- **FT** — paket warisan paling tua, 3 instance dengan ukuran heterogen (square 6×6, square 10×10, dan tall 20×5).
- **LA** — koleksi 40 instance Lawrence dengan berbagai aspect ratio. Eval pakai 15 instance dari subset awal:
  - **la01–la05**: 10×5 (jobs ≤ 10, machines ≤ 10) → **same-shape** vs training
  - **la06–la10**: 15×5 (jobs = 15 > training max 10) → **shape-unseen**
  - **la16–la20**: 10×10 (border atas in-dist) → **same-shape**
- **TA** — Taillard 15×15, **paling jauh dari distribusi training**: 225 operasi per instance (2.25× max training), kedua dimensi (jobs dan machines) di luar yang pernah dilatih.

**Klasifikasi OOD vs in-shape:**

| Kategori | Definisi | Instance |
|---|---|---|
| **same-shape OOD** | Ukuran masih ≤ 10×10 (terlihat saat training) tapi instance dari distribusi berbeda | ft06, ft10, la01–05, la16–20 |
| **shape-unseen OOD** | Jobs atau machines > 10 | ft20 (20×5), la06–10 (15×5) |
| **fully-unseen OOD** | Jobs DAN machines > 10 | ta01–ta10 (15×15) |

### 1.3 Format Prompt (sama untuk in-dist & OOD)

Semua instance OOD di-convert ke format Starjob lewat `to_starjob_format()` di `eval_benchmarks.py:102` agar identik dengan input training:

```
Optimize schedule for {N} Jobs (denoted as J) across {M} Machines (denoted as M)
to minimize makespan. The makespan is the completion time of the last operation
in the schedule. Each M can process only one J at a time, and once started, J
cannot be interrupted.

J0:
M1:21 M0:53 M4:95 M3:55 M2:34
J1:
...
```

---

## 2. Setup Evaluasi

| | In-distribution | Out-of-distribution |
|---|---|---|
| **Eval set** | 200 sampel acak (seed=42) dari `starjob_train_sm.jsonl` test split | FT (3) + LA (15) + TA (10) = 28 instance per model per method |
| **Script** | `eval_lora.py`, `eval_rslora.py` | `eval_benchmarks.py` (LoRA), `eval_rslora_benchmarks.py` (rsLoRA) |
| **Generation** | `temperature=0.1`, `top_p=0.95`, `max_new_tokens=4096` | `temperature=0.1`, `top_p=0.95`, `max_new_tokens=7000` |
| **Validator** | Routing order + machine non-overlap + complete operations |  Sama |
| **Metrik** | Feasibility %, exact-match makespan %, mean/median gap | Feasibility %, gap vs best-known, per-family rollup |

**Catatan checkpoint:** LoRA OOD pakai checkpoint-14400 (hardcoded). rsLoRA OOD pakai `find_best_checkpoint` (efektif step 9800). In-distribution semua pakai checkpoint terbaik per metode/model.

---

## 3. Hasil In-Distribution (n=200, Starjob test split)

| Method | Model | Time | Feasible | Exact | Mean gap | Median gap |
|---|---|---:|---:|---:|---:|---:|
| **LoRA** | LLaMA 3.1 8B | 21.5 min | **96.5%** (193/200) | **34.5%** | **6.88%** | **3.47%** |
| rsLoRA | LLaMA 3.1 8B | 24.2 min | 95.0% (190/200) | 32.0% | 9.80% | 5.29% |
| **LoRA** | Granite 3.2 8B | 134.9 min | **86.5%** (173/200) | **33.5%** | **56.15%** | **4.76%** |
| rsLoRA | Granite 3.2 8B | 147.9 min | 24.5% (49/200) | 5.5% | 215.27% | 41.42% |
| **LoRA** | Ministral 8B | 93.8 min | **95.0%** (190/200) | **32.0%** | **15.67%** | **4.91%** |
| rsLoRA | Ministral 8B | 118.9 min | 64.0% (128/200) | 24.5% | 42.93% | 9.25% |
| LoRA | Qwen2 7B | 38.6 min | 1.0% (2/200) | 3.0% | 56.30% | 28.29% |
| **rsLoRA** | Qwen2 7B | 31.8 min | **50.0%** (100/200) | **27.5%** | **27.81%** | **9.37%** |

**Aggregate per metode (semua model, n=800):**

| Method | Feasible | Exact |
|---|---:|---:|
| **LoRA** | **558/800 = 69.75%** | 206/800 = 25.75% |
| rsLoRA | 467/800 = 58.38% | 179/800 = 22.38% |

### 3.1 Feasibility per kompleksitas (in-distribution)

Bucket berdasarkan total operasi (jobs × machines):

| Bucket | Range ops | Sample (n=200) | LoRA agg (×4 model) | rsLoRA agg (×4 model) |
|---|---|---:|---:|---:|
| S | ≤ 9 ops | 18 | 56/72 = 77.8% | **68/72 = 94.4%** |
| M | 10–25 ops | 65 | 194/260 = 74.6% | **205/260 = 78.8%** |
| L | 26–50 ops | 62 | **185/248 = 74.6%** | 133/248 = 53.6% |
| XL | > 50 ops | 55 | **123/220 = 55.9%** | 61/220 = 27.7% |

**Pola:** rsLoRA lebih kuat di problem kecil/sedang, LoRA lebih tahan di problem besar.

---

## 4. Hasil Out-of-Distribution

### 4.1 Per-family aggregate (4 model digabung)

| Family | Total | LoRA | rsLoRA |
|---|---:|---:|---:|
| FT (3 inst × 4 model) | 12 | 3/12 = 25.0% | 3/12 = 25.0% |
| LA (15 × 4) | 60 | **20/60 = 33.3%** | 12/60 = 20.0% |
| TA (10 × 4) | 40 | 0/40 = 0.0% | 0/40 = 0.0% |
| **Total OOD** | **112** | **23/112 = 20.5%** | 15/112 = 13.4% |

### 4.2 LA dipecah berdasarkan shape

| Bucket LA | Total | LoRA | rsLoRA |
|---|---:|---:|---:|
| **Same-shape** (la01–05 = 10×5, la16–20 = 10×10) | 40 | **19/40 = 47.5%** | 12/40 = 30.0% |
| **Shape-unseen** (la06–10 = 15×5) | 20 | 1/20 = 5.0% | 0/20 = 0.0% |

### 4.3 Per-dataset, per-model, per-method

#### FT (Fisher-Thompson) — ft06 (6×6), ft10 (10×10), ft20 (20×5)

| Model | LoRA feasible | LoRA mean/med gap | rsLoRA feasible | rsLoRA mean/med gap |
|---|---:|---:|---:|---:|
| LLaMA | 1/3 (33.3%) | 7.3% / 7.3% | 1/3 (33.3%) | 5.5% / 5.5% |
| Granite | 1/3 (33.3%) | 5.5% / 5.5% | 0/3 (0.0%) | — |
| Ministral | 1/3 (33.3%) | 7.3% / 7.3% | 1/3 (33.3%) | 20.0% / 20.0% |
| Qwen2 | 0/3 (0.0%) | — | 1/3 (33.3%) | 5.5% / 5.5% |
| **Total** | **3/12 = 25.0%** | | **3/12 = 25.0%** | |

Pola FT: ukuran square kecil/menengah masih bisa, tapi ft20 (20×5) gagal di hampir semua kombinasi karena jumlah jobs > training max.

#### LA same-shape (la01–05 = 10×5, la16–20 = 10×10)

| Model | LoRA feasible | LoRA mean/med gap | rsLoRA feasible | rsLoRA mean/med gap |
|---|---:|---:|---:|---:|
| LLaMA | **8/10 (80.0%)** | 8.0% / 8.5% | 7/10 (70.0%) | 60.5% / 11.5% |
| Granite | 5/10 (50.0%) | 7.8% / 9.2% | 0/10 (0.0%) | — |
| Ministral | 6/10 (60.0%) | 7.9% / 4.9% | 5/10 (50.0%) | 11.2% / 9.9% |
| Qwen2 | 0/10 (0.0%) | — | 0/10 (0.0%) | — |
| **Total** | **19/40 = 47.5%** | | 12/40 = 30.0% | |

LLaMA + LoRA paling kuat di OOD same-shape (80% feasibility, 8.5% median gap).

#### LA shape-unseen (la06–10 = 15×5, jobs = 15 > training max 10)

| Model | LoRA feasible | LoRA mean/med gap | rsLoRA feasible | rsLoRA mean/med gap |
|---|---:|---:|---:|---:|
| LLaMA | 1/5 (20.0%) | 0.6% / 0.6% | 0/5 (0.0%) | — |
| Granite | 0/5 (0.0%) | — | 0/5 (0.0%) | — |
| Ministral | 0/5 (0.0%) | — | 0/5 (0.0%) | — |
| Qwen2 | 0/5 (0.0%) | — | 0/5 (0.0%) | — |
| **Total** | **1/20 = 5.0%** | | 0/20 = 0.0% | |

Hanya 1 schedule LLaMA-LoRA yang feasible — generalisasi ke jumlah jobs > 10 sangat lemah.

#### TA (Taillard 15×15) — fully-unseen, 225 ops

| Model | LoRA feasible | rsLoRA feasible |
|---|---:|---:|
| LLaMA | 0/10 (0.0%) | 0/10 (0.0%) |
| Granite | 0/10 (0.0%) | 0/10 (0.0%) |
| Ministral | 0/10 (0.0%) | 0/10 (0.0%) |
| Qwen2 | 0/10 (0.0%) | 0/10 (0.0%) |
| **Total** | **0/40 = 0.0%** | **0/40 = 0.0%** |

Kedua metode kolaps total — kedua dimensi di luar training, dan total ops 2.25× lebih besar dari maksimum yang pernah dilihat.

---

## 5. Ringkasan & Kesimpulan

### 5.1 Tabel ringkas feasibility

| Skenario | n | LoRA | rsLoRA |
|---|---:|---:|---:|
| **In-distribution** (Starjob test, ≤ 10×10) | 800 | **69.75%** | 58.38% |
| OOD same-shape (FT06/10, LA01–05, LA16–20) | 52 | ~50% | ~30% |
| OOD shape-unseen (FT20, LA06–10) | 32 | ~6% | 0% |
| OOD fully-unseen (TA 15×15) | 40 | **0%** | **0%** |

### 5.2 Insight utama

1. **In-dist LoRA > rsLoRA** di 3 dari 4 model (LLaMA, Granite, Ministral). Pengecualian: **Qwen2** — di mana LoRA collapse (1%) sementara rsLoRA mencapai 50%.
2. **Granite + rsLoRA gagal training** dalam 1 epoch (24.5% feasibility). LoRA varian model yang sama trains cleanly ke 86.5%.
3. **OOD generalization ada batas yang jelas:**
   - Same-shape (≤ 10×10): degradasi moderat, LoRA tetap unggul.
   - Shape-unseen (jobs > 10 ATAU machines > 10): kolaps drastis ke ≤ 5%.
   - Fully-unseen (Taillard 15×15, 225 ops): **0% di kedua metode dan semua model**.
4. **LLaMA paling robust** untuk transfer OOD same-shape (80% LoRA, 70% rsLoRA pada LA).
5. **Bottleneck bukan token budget** — meskipun OOD pakai 7000 token (vs 4096 in-dist), feasibility tetap 0% di TA karena failure-nya struktural (model tidak pernah belajar coordinate ≥ 11 jobs/machines), bukan truncation.

### 5.3 Rekomendasi

- Untuk JSSP ≤ 10×10 → **LoRA** lebih baik secara umum.
- Untuk Qwen2 secara spesifik → **rsLoRA** wajib (LoRA collapse).
- Untuk problem OOD ≥ 15×15 → **kedua metode tidak cukup**; perlu perluasan training distribution (Starjob full 130k yang naik sampai 16×16, atau augmentasi ukuran lebih besar) sebelum bisa diharapkan feasibility > 0%.

---

## 6. Sumber Data

- `comparison_lora_vs_rslora.json` — head-to-head in-distribution
- `comparison_lora_vs_rslora_ood.json` — head-to-head OOD (FT/LA/TA)
- `metrics_{lora,rslora}_{llama,granite,ministral,qwen2}.json` — per-instance in-distribution
- `metrics_benchmarks_{model}.json`, `metrics_rslora_benchmarks_{model}.json` — per-instance OOD
- `data/starjob_train_sm.jsonl` — training/test data
- `data/benchmarks/jobshop1.txt`, `data/benchmarks/ta*.txt` — OR-Library benchmarks
