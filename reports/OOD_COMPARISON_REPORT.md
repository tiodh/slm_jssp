# OOD Comparison Report — All Models on 18 FT+LA Instances

**Date**: 2026-06-05
**Models**: SFT-LoRA, LoRA-LLaMA, rsLoRA-LLaMA, GRPO-V5 (hybrid+LC), GRPO-V6 ck400 (stratified+LC)

**Training distribution** (StarJob SM): jobs 2–10, machines 2–10 (45 shape pairs × 2400 instance).

**Shape classification**:
- **In-distribution** (jobs ≤ 10): ft06, ft10, la01–la05, la16–la20 — 12 instance
- **OOD shape** (jobs > 10): ft20 (20×5), la06–la10 (15×5) — 6 instance

---

## 1. SFT-LoRA Baseline — Per-Instance Violations

### 1a. OOD (18 instance)

```
idx,name,n_ops,bks,feasible,makespan,missing_op_count,routing_order_violations,machine_capacity_violations,timing_consistency_violations,precedence_violations,total_violations,ops_emitted,ops_expected,gen_time_s
,ft06,36,55,True,61,0,0,0,0,0,0,36,36,6.13
,ft10,100,930,True,1602,0,0,0,0,0,0,136,100,24.23
,ft20,100,1165,False,798,50,0,0,0,0,50,50,100,8.33
,la01,50,666,True,666,0,0,0,0,0,0,50,50,8.05
,la02,50,655,True,721,0,0,0,0,0,0,50,50,8.05
,la03,50,597,True,687,0,0,0,0,0,0,50,50,8.05
,la04,50,590,True,701,0,0,0,0,0,0,50,50,8.03
,la05,50,593,True,605,0,0,0,0,0,0,50,50,8.04
,la06,75,926,False,824,25,0,0,0,0,25,51,75,8.38
,la07,75,890,False,664,25,0,0,0,0,25,50,75,8.20
,la08,75,863,False,614,25,0,0,0,0,25,50,75,8.22
,la09,75,951,False,703,25,0,0,0,1,26,50,75,8.20
,la10,75,958,False,746,25,0,0,0,0,25,50,75,8.21
,la16,100,945,False,5879,0,3,13,7,0,23,299,100,63.88
,la17,100,784,True,1113,0,0,0,0,0,0,100,100,17.15
,la18,100,848,False,4504,0,0,16,7,0,23,300,100,63.89
,la19,100,842,False,1216,2,2,1,0,0,5,108,100,18.79
,la20,100,902,True,992,0,0,0,0,0,0,100,100,16.89
```

**Summary**: 9 feasible / 18 OOD = 50%. Mean gap (feas) +20.12%, median +10.91%.

### 1b. SM (200 StarJob test split)

200 instance — 190 feasible (95%). 10 infeasible terkonsentrasi di 3 instance (idx 13=223 viol, idx 108=213, idx 50=152) yang menyumbang ~93% total violations. Breakdown:
- `machine_capacity_violations_sum`: 389
- `routing_order_violations_sum`: 100
- `precedence_violations_sum`: 70
- `timing_consistency_violations_sum`: 40
- `missing_op_count_sum`: 5

CSV detail: `sft_per_instance_violations_sm.csv` (200 rows).

---

## 2. Per-Instance Comparison — All Models (OOD)

### 2a. Feasibility (original checker, sebelum strict patch)

| name | shape  | bks  | SFT-LoRA | LoRA-LLaMA | rsLoRA-LLaMA | GRPO-V5 | GRPO-V6 ck400 |
|------|--------|------|----------|------------|--------------|---------|---------------|
| ft06 | 6×6    | 55   | ✓        | ✓          | ✓            | ✓       | ✓             |
| ft10 | 10×10  | 930  | ✓        | ✗          | ✗            | ✓       | ✗             |
| ft20 | 20×5   | 1165 | ✗        | ✗          | ✗            | ✗       | ✗             |
| la01 | 10×5   | 666  | ✓        | ✓          | ✓            | ✓       | ✓             |
| la02 | 10×5   | 655  | ✓        | ✓          | ✓            | ✓       | ✓             |
| la03 | 10×5   | 597  | ✓        | ✓          | ✓            | ✓       | ✓             |
| la04 | 10×5   | 590  | ✓        | ✓          | ✓            | ✓       | ✓             |
| la05 | 10×5   | 593  | ✓        | ✓          | ✓            | ✓       | ✓             |
| la06 | 15×5   | 926  | ✗        | ✗          | ✗            | ✗       | ✗             |
| la07 | 15×5   | 890  | ✗        | ✓          | ✗            | ✗       | ✗             |
| la08 | 15×5   | 863  | ✗        | ✗          | ✗            | ✗       | ✗             |
| la09 | 15×5   | 951  | ✗        | ✓          | ✗            | ✗       | ✗             |
| la10 | 15×5   | 958  | ✗        | ✗          | ✗            | ✗       | ✗             |
| la16 | 10×10  | 945  | ✗        | ✓          | ✗            | ✓       | ✓             |
| la17 | 10×10  | 784  | ✓        | ✗          | ✗            | ✓       | ✓             |
| la18 | 10×10  | 848  | ✗        | ✗          | ✓            | ✓       | ✓             |
| la19 | 10×10  | 842  | ✗        | ✓          | ✓            | ✓       | ✓             |
| la20 | 10×10  | 902  | ✓        | ✗          | ✗            | ✓       | ✓             |
| **Total** | |     | **9/18 (50%)** | **10/18 (56%)** | **8/18 (44%)** | **12/18 (67%)** | **11/18 (61%)** |

### 2b. Makespan & gap% (— = infeasible)

| name | bks  | SFT-LoRA       | LoRA-LLaMA     | rsLoRA-LLaMA   | GRPO-V5        | GRPO-V6 ck400  |
|------|------|----------------|----------------|----------------|----------------|----------------|
| ft06 | 55   | 61 (+10.9%)    | 59 (+7.3%)     | 61 (+10.9%)    | 59 (+7.3%)     | 61 (+10.9%)    |
| ft10 | 930  | 1602 (+72.3%)  | —              | —              | 1637 (+76.0%)  | —              |
| ft20 | 1165 | —              | —              | —              | —              | —              |
| la01 | 666  | 666 (+0.0%)    | 693 (+4.1%)    | 666 (+0.0%)    | 666 (+0.0%)    | 697 (+4.7%)    |
| la02 | 655  | 721 (+10.1%)   | 763 (+16.5%)   | 732 (+11.8%)   | 730 (+11.5%)   | 698 (+6.6%)    |
| la03 | 597  | 687 (+15.1%)   | 652 (+9.2%)    | 650 (+8.9%)    | 650 (+8.9%)    | 653 (+9.4%)    |
| la04 | 590  | 701 (+18.8%)   | 662 (+12.2%)   | 687 (+16.4%)   | 637 (+8.0%)    | 637 (+8.0%)    |
| la05 | 593  | 605 (+2.0%)    | 593 (+0.0%)    | 593 (+0.0%)    | 593 (+0.0%)    | 593 (+0.0%)    |
| la06 | 926  | —              | —              | —              | —              | —              |
| la07 | 890  | —              | 975 (+9.6%)    | —              | —              | —              |
| la08 | 863  | —              | —              | —              | —              | —              |
| la09 | 951  | —              | 951 (+0.0%)    | —              | —              | —              |
| la10 | 958  | —              | —              | —              | —              | —              |
| la16 | 945  | —              | 1145 (+21.2%)  | —              | 1153 (+22.0%)  | 1092 (+15.6%)  |
| la17 | 784  | 1113 (+42.0%)  | —              | —              | 818 (+4.3%)    | 933 (+19.0%)   |
| la18 | 848  | —              | —              | 1055 (+24.4%)  | 982 (+15.8%)   | 957 (+12.9%)   |
| la19 | 842  | —              | 935 (+11.0%)   | 955 (+13.4%)   | 1130 (+34.2%)  | 1011 (+20.1%)  |
| la20 | 902  | 992 (+10.0%)   | —              | —              | 1095 (+21.4%)  | 1058 (+17.3%)  |
| **Mean gap (feas)** | | +20.12% | +9.10% | +10.73% | +17.44% | **+11.30%** |
| **Median gap (feas)** | | +10.91% | +9.38% | +11.33% | **+10.16%** | +10.91% |

### 2c. ops_emitted / ops_expected (bold = mismatch)

| name      | exp | SFT-LoRA | LoRA-LLaMA | rsLoRA-LLaMA | GRPO-V5 | GRPO-V6 ck400 |
|-----------|-----|----------|------------|--------------|---------|---------------|
| ft06      | 36  | 36       | 36         | 36           | 36      | 36            |
| ft10      | 100 | **136**  | 100        | **101**      | **131** | **295**       |
| ft20      | 100 | **50**   | **118**    | **50**       | **51**  | **49**        |
| la01      | 50  | 50       | 50         | 50           | 50      | 50            |
| la02      | 50  | 50       | 50         | 50           | 50      | 50            |
| la03      | 50  | 50       | 50         | 50           | 50      | 50            |
| la04      | 50  | 50       | 50         | 50           | 50      | 50            |
| la05      | 50  | 50       | 50         | 50           | 50      | 50            |
| la06      | 75  | **51**   | **71**     | **54**       | **51**  | **51**        |
| la07      | 75  | **50**   | **77**     | **50**       | **50**  | **50**        |
| la08      | 75  | **50**   | **70**     | **50**       | **50**  | **50**        |
| la09      | 75  | **50**   | 75         | **50**       | **50**  | **50**        |
| la10      | 75  | **50**   | **70**     | **50**       | **50**  | **50**        |
| la16      | 100 | **299**  | **104**    | **504**      | **104** | 100           |
| la17      | 100 | 100      | **105**    | **104**      | 100     | **102**       |
| la18      | 100 | **300**  | **98**     | 100          | 100     | 100           |
| la19      | 100 | **108**  | 100        | 100          | **105** | **103**       |
| la20      | 100 | 100      | **493**    | **494**      | **105** | **101**       |

---

## 3. Split by Shape Distribution

### 3a. In-distribution shape (12 instance, jobs ≤ 10)

**Feasibility** (original checker):

| Model | In-dist | OOD shape | Overall |
|---|---|---|---|
| SFT-LoRA | 9/12 (75%) | 0/6 (0%) | 9/18 (50%) |
| LoRA-LLaMA | 8/12 (67%) | 2/6 (33%) | 10/18 (56%) |
| rsLoRA-LLaMA | 7/12 (58%) | 0/6 (0%) | 7/18* (39%) |
| GRPO-V5 | 12/12 (100%) | 0/6 (0%) | 12/18 (67%) |
| GRPO-V6 ck400 | 11/12 (92%) | 0/6 (0%) | 11/18 (61%) |

\* rsLoRA total = 7 (la07 ✗ tidak 8); ref breakdown 2a.

### 3b. OOD shape (6 instance, jobs > 10) — semua model truncate

Mode failure konsisten: emit ~50 ops untuk 15×5 (expected 75) dan 20×5 (expected 100). SFT prior membatasi length ke ~50 ops/output, GRPO tidak override.

| Model | OOD shape feasibility |
|---|---|
| SFT-LoRA | 0/6 |
| LoRA-LLaMA | 2/6 (la07, la09 — eval setup berbeda) |
| rsLoRA-LLaMA | 0/6 |
| GRPO-V5 | 0/6 |
| GRPO-V6 ck400 | 0/6 |

---

## 4. STRICT Re-evaluation (Patched Feasibility)

**Definisi strict_feasible**: original checker pass DAN tidak ada job dengan `ops_emitted_j > expected_j` (no padding).

**Mengapa perlu strict**: `feasibility.py:78` hanya validasi N op pertama per job untuk routing/precedence. Model bisa pad output dengan duplicate ops; lolos `feasible=True` selama extras tidak overlap di machine. V5 ft10 emit J0=18 ops, J8=20 ops (expected 10 masing-masing) tapi lolos feasible.

### 4a. Strict feasibility per instance

| name | bks  | SFT-LoRA       | LoRA-LLaMA     | rsLoRA-LLaMA   | GRPO-V5        | GRPO-V6 ck400  |
|------|------|----------------|----------------|----------------|----------------|----------------|
| ft06 | 55   | ✓              | ✓              | ✓              | ✓              | ✓              |
| ft10 | 930  | ✗ (e=136)      | ✗*             | ✗ (e=101)      | ✗ (e=131)      | ✗ (e=295)      |
| ft20 | 1165 | ✗              | ✗              | ✗              | ✗              | ✗              |
| la01 | 666  | ✓              | ✓              | ✓              | ✓              | ✓              |
| la02 | 655  | ✓              | ✓              | ✓              | ✓              | ✓              |
| la03 | 597  | ✓              | ✓              | ✓              | ✓              | ✓              |
| la04 | 590  | ✓              | ✓              | ✓              | ✓              | ✓              |
| la05 | 593  | ✓              | ✓              | ✓              | ✓              | ✓              |
| la06–la10 | 75 | ✗ all          | ✗ except la09  | ✗ all          | ✗ all          | ✗ all          |
| la16 | 945  | ✗              | ✗              | ✗              | ✗              | **✓**          |
| la17 | 784  | ✓              | ✗              | ✗              | ✓              | ✗              |
| la18 | 848  | ✗              | ✗*             | ✓              | ✓              | ✓              |
| la19 | 842  | ✗              | ✓              | ✓              | ✗              | ✗              |
| la20 | 902  | ✓              | ✗              | ✗              | ✗              | ✗              |
| **Strict total** | | **8/18 (44%)** | **8/18 (44%)** | **8/18 (44%)** | **8/18 (44%)** | **8/18 (44%)** |

\* LoRA-LLaMA ft10 (100/100) dan la18 (98/100): ops match tapi tetap infeasible karena ada violations original (mcap/missing).

### 4b. Trimmed makespan & gap% (strict-feasible only; first N ops per job)

| name | bks | SFT-LoRA      | LoRA-LLaMA      | rsLoRA-LLaMA    | GRPO-V5          | GRPO-V6 ck400  |
|------|-----|---------------|-----------------|-----------------|------------------|----------------|
| ft06 | 55  | 61 (+10.9%)   | 59 (+7.3%)      | 61 (+10.9%)     | 59 (+7.3%)       | 61 (+10.9%)    |
| la01 | 666 | 666 (+0.0%)   | 693 (+4.1%)     | 666 (+0.0%)     | 666 (+0.0%)      | 697 (+4.7%)    |
| la02 | 655 | 721 (+10.1%)  | 763 (+16.5%)    | 732 (+11.8%)    | 730 (+11.5%)     | 698 (+6.6%)    |
| la03 | 597 | 687 (+15.1%)  | 652 (+9.2%)     | 650 (+8.9%)     | 650 (+8.9%)      | 653 (+9.4%)    |
| la04 | 590 | 701 (+18.8%)  | 662 (+12.2%)    | 687 (+16.4%)    | 637 (+8.0%)      | 637 (+8.0%)    |
| la05 | 593 | 605 (+2.0%)   | 593 (+0.0%)     | 593 (+0.0%)     | 593 (+0.0%)      | 593 (+0.0%)    |
| la09 | 951 | —             | **951 (+0.0%)** | —               | —                | —              |
| la16 | 945 | —             | —               | —               | —                | **1092 (+15.6%)** |
| la17 | 784 | 1113 (+42.0%) | —               | —               | **818 (+4.3%)**  | —              |
| la18 | 848 | —             | —               | 1055 (+24.4%)   | **982 (+15.8%)** | 957 (+12.9%)   |
| la19 | 842 | —             | 935 (+11.0%)    | 955 (+13.4%)    | —                | —              |
| la20 | 902 | 992 (+10.0%)  | —               | —               | —                | —              |
| **n (strict feas)**  | | **8** | **8** | **8** | **8** | **8** |
| **Mean gap**         | | **+13.61%** | **+7.53%** | **+10.73%** | **+6.96%** 🥇 | **+8.49%** |
| **Median gap**       | | **+10.49%** | **+8.24%** | **+11.33%** | **+7.62%** 🥇 | **+8.67%** |

### 4c. Apples-to-apples ranking (strict, 8/18 each — same set size, different sets)

**Mean gap (lower=better):**
1. **GRPO-V5: +6.96%** 🥇
2. LoRA-LLaMA: +7.53%
3. GRPO-V6 ck400: +8.49%
4. rsLoRA-LLaMA: +10.73%
5. SFT-LoRA: +13.61%

### 4d. Per-instance pass distribution (which 8 instances each model passes)

| Instance | SFT-LoRA | LoRA-LLaMA | rsLoRA-LLaMA | GRPO-V5 | GRPO-V6 ck400 | Passed by |
|----------|----------|------------|--------------|---------|---------------|-----------|
| ft06 + la01–la05 (6) | ✓ | ✓ | ✓ | ✓ | ✓ | semua model |
| la09 (15×5)          | ✗ | **✓** | ✗ | ✗ | ✗ | LoRA only |
| la16 (10×10)         | ✗ | ✗ | ✗ | ✗ | **✓** | V6 only |
| la17 (10×10)         | **✓** | ✗ | ✗ | **✓** | ✗ | SFT, V5 |
| la18 (10×10)         | ✗ | ✗ | **✓** | **✓** | **✓** | rsLoRA, V5, V6 |
| la19 (10×10)         | ✗ | **✓** | **✓** | ✗ | ✗ | LoRA, rsLoRA |
| la20 (10×10)         | **✓** | ✗ | ✗ | ✗ | ✗ | SFT only |

---

## 5. Key Findings

1. **Klaim "GRPO V5 = 12/12 in-dist feasibility" salah** post-strict check. Setiap model lolos exactly **8/18 strict** — angka identik. Klaim feasibility gain V5/V6 sebelumnya inflated oleh checker quirk.

2. **GRPO tetap meningkatkan kualitas** pada subset valid. V5 mean gap +6.96% (terbaik); SFT +13.61% (terburuk). Gap turun ~50% relative dari SFT ke V5.

3. **8 instance yang lolos berbeda antar model.** Universal pass: 6 small problems (ft06 + la01–la05). Variable: hanya 2 dari 10×10 LA pass per model (kombinasi berbeda).

4. **Mode failure utama = padding/duplikasi ops per job.** Length-control reward V5/V6 cegah looping ekstrim (404+ ops) tapi tidak cegah over-emit moderat (+4–31 ops, distributed). V5 ft10: J0=18, J8=20 ops (expected 10 each).

5. **OOD shape (jobs > 10) = truncation universal.** Semua model emit ~50 ops untuk problem yang butuh 75/100 ops. SFT prior length ~50 tidak override-able dengan RL.

6. **GRPO V6 ck400 satu-satunya yang strict-feasible di la16** (100 ops exact). V5 over-emit 4 ops di la16 → strict infeasible. V6 length-control lebih ketat di emit count tapi gagal di ft10 (loop 295 ops).

7. **Checker bug**: `feasibility.py:78` perlu strict mode — `if len(ops_j) != len(expected): missing_op_count += abs(diff)` (sekarang hanya menghukum `<`, mengabaikan `>`).

---

## 6. Next Step: GRPO V7 Plan

Tercatat di project memory: `grpo_v7_emitted_control_plan.md`.

**Inti**:
1. Reward V7: ganti length-control → **emitted-control per-job** (`Σⱼ max(0, len(ops_j) − expected_j)`).
2. Patch `feasibility.py` agar strict by default.
3. Starter config V7 = V6 ck400 + reward swap.
4. Target strict ≥10/18 feasibility, mean gap ≤ +8%.
5. Baseline historical strict = 8/18 untuk semua model.

---

## 7. CSV Files (raw data)

- `sft_per_instance_violations_sm.csv` — SFT-LoRA SM (200 rows)
- `sft_per_instance_violations_ood.csv` — SFT-LoRA OOD (18 rows)
- `rslora_llama_per_instance_ood.csv` — rsLoRA-LLaMA OOD
- `grpo_v5_per_instance_violations_ood.csv` — GRPO V5 OOD (SFT-format header)
- `grpo_v5_full_per_instance_ood.csv` — GRPO V5 OOD (extended header)
- `grpo_v6_ck400_per_instance_violations_ood.csv` — GRPO V6 OOD (SFT-format header)
- `grpo_v6_ck400_full_per_instance_ood.csv` — GRPO V6 OOD (extended header)
- `ood_comparison_all_models.csv` — wide-format comparison (per-instance × 5 models)
- `strict_recheck_ood_all_models.csv` — strict feasibility + trimmed makespan (5 models)
