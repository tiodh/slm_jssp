# Project Chronology — SFT → V7 Plan

Catatan kronologis diskusi & milestone dari awal proyek SLM-on-JSSP hingga sekarang (2026-06-05).
Sumber: git log, memory files, eval result timestamps, existing reports.

---

## Phase 1 — Initial Pipeline & Multi-Model SFT (Apr 28–29)

**Goal**: Fine-tune 4 small LLMs (LLaMA 3.1-8B, Granite 3.2-8B, Ministral 8B, Qwen2 7B) untuk JSSP, bandingkan LoRA vs rsLoRA.

| Date | Milestone |
|---|---|
| 2026-04-28 | Initial commit — training/eval pipeline `SLM-on-JSSP` |
| 2026-04-29 | LoRA evaluation pipeline + LoRA vs rsLoRA comparison |
| 2026-04-29 | README → experiment report format |
| 2026-04-30 | `EVALUATION_REPORT.md` finalized — dataset (StarJob 108k, 2×2–10×10), 18 OOD FT+LA holdout, 4 model × 2 method = 8 setup |

**Result Phase 1**:
- LLaMA + rsLoRA = setup terbaik di SM (190/200 = 95%) dan OOD (lihat metrics_rslora_benchmarks_llama.json)
- OOD baseline 9/18 (50%) with mean gap +20.12%
- Dipilih sebagai SFT base untuk GRPO eksperimen lanjutan

---

## Phase 2 — Violation Analysis (May 4–6)

| Date | Milestone |
|---|---|
| 2026-05-04 | `VIOLATION_REPORT.md` — 5-kategori breakdown semua model |
| 2026-05-06 | `MAKESPAN_DISTRIBUTION.md` — distribusi makespan per shape (SM + OOD) |

**Diskusi key**:
- SM violation dominan: `machine_capacity` (64%)
- OOD violation dominan: `missing_op_count` (78%, truncation di instance jobs>10)
- **Insight**: GRPO yang dirancang untuk SM tidak akan otomatis fix OOD (failure mode beda)

---

## Phase 3 — GRPO V1 Initial + Hardware Hang (May 15–17)

| Date | Milestone |
|---|---|
| 2026-05-15 | Pilot V1 stratified (n=100) + pilot uniform (n=100); SFT baseline eval files dibuat |
| 2026-05-15 | `claude_code_prompt_grpo.md` — GRPO setup notes |
| 2026-05-16 | **Hardware hang at step 859** during full V1 stratified (collapse at step 700, freeze at 859) |
| 2026-05-17 | Diagnosis + mitigation memo: `grpo_collapse_findings.md`, `user_prefers_sft_aligned_generation_cap.md` |

**Diagnosis collapse V1** (see `memory/grpo_collapse_findings.md`):
1. Step 0–500: training normal
2. Step 500–700: **length drift** — stratified reward `-Σwᵢ × nᵢ/N_ops` mendorong "length hacking" (lebih banyak baris parser = ratio missing turun)
3. Step 700–710: grad_norm spike 1.3 → 11.76, **no `max_grad_norm`** clamp
4. Post-spike: LoRA weights keluar SFT basin, EOS distribution rusak, output gibberish hingga 4096 tok cap
5. Absorbing state: K=4 sample dapat reward identik → reward_std=0 → advantage=0 → no gradient
6. `KL_COEF=0.04` terlalu lemah anchor back ke SFT

**Bukti 4096 tok = pathology, bukan complexity**: gold response max 1336 tok (10×10), training data max 100 ops legitim. Post-collapse semua instance termasuk 5×5 hit 4096.

**Mitigasi disepakati 2026-05-17**:
- `max_grad_norm` = 1.0 (cegah grad spike)
- `KL_COEF` 0.04 → 0.10 (anchor cumulative drift)
- Reward + length penalty + EOS bonus (V2)
- `max_new_tokens` TETAP 4096 (user tolak penurunan — `user_prefers_sft_aligned_generation_cap.md`)

**Hardware**: hang bukan dari training collapse — diduga PSU/GPU driver/Kraken cooler. Pre-check `lm-sensors` + stress-test 10 min.

---

## Phase 4 — V1–V4.2 All Collapse (May 17–21)

| Date | Milestone |
|---|---|
| 2026-05-17 | Pilot V2 stratified+length-shape (n=100) |
| 2026-05-18 | Pilot V3 stratified n=2000 records / 200 steps |
| 2026-05-19 | Full V1, V2, V3 checkpoint-200/600 — all crashed/collapsed |
| 2026-05-21 | Pilot V4 hybrid + V4.2 hybrid variants |
| 2026-05-21 | Commit "GRPO-JSSP experiments: stratified → hybrid P-GRPO reward (V1–V4.2)" + `EXPERIMENT_NOTES.md` |
| 2026-05-21 | `SFT_NOTES.md` finalized |

**Result Phase 4**:
- V1 (stratified, KL=0.04) → collapse step 700
- V2 (stratified + soft length + EOS bonus, KL=0.10) → collapse step 210 (worst survival)
- V3 (stratified, n=2000, ga=4, T=0.7) → collapse step ~200
- V4 (hybrid 7-component, KL=0.05) → collapse step ~200
- V4.2 (hybrid variant) → collapse

**Insight kunci** (per `EXPERIMENT_NOTES.md` TL;DR):
> "Length must be controlled at the **advantage** level (zero advantage for over-length samples), not the reward level. Soft length shaping (V2) was actively harmful; advantage masking sidesteps both the reward cliff and absorbing-state collapse."

---

## Phase 5 — V5 Survival + V6 (May 21–29)

| Date | Milestone |
|---|---|
| 2026-05-21 | Pilot V5 hybrid + length-control (advantage masking `clen > 2.0×gold_est`) — n=100 → survives |
| 2026-05-23 | **GRPO V5 full** (n=2000, 500 steps, hybrid+LC) — first run to survive all 500 steps and beat SFT |
| 2026-05-23 | Pilot V6 stratified+LC (n=100) |
| 2026-05-24 | **GRPO V6 full** (n=2000, stratified+LC) — checkpoint-400 selected as best (training continued but ck-400 best per evaluation) |
| 2026-05-28 | `REWARD_AND_LENGTH_CONTROL.md` — formal documentation V5 reward + LC mechanism |
| 2026-05-28 | `V5_vs_V6_comparison.md` — head-to-head SFT vs V5 vs V6 |
| 2026-05-29 | README updated dengan V5+V6 results |

**Result V5 (claimed at the time, pre-strict-recheck)**:
- SM: 194/200 (97.0%, +2.0pp vs SFT)
- OOD: 12/18 (66.7%, +16.7pp vs SFT)
- Fixed looping di la16, la20 yang rsLoRA-LLaMA SFT loop

**Result V6 ck-400** (pre-strict-recheck):
- SM: 195/200 (97.5%)
- OOD: 11/18 (61.1%)
- Mean gap lebih rendah dari V5 di feasibles

**Key design insight V5**: V5 = V4 reward (hybrid 7-component, range [-1, +7]) + **advantage masking** untuk overlength samples. Bukan reward shaping (V2 mistake) — masking di trainer level.

---

## Phase 6 — Session LoRA Base (Jun 4)

**Goal session**: Replicate V1–V6 pipeline tapi dengan **LoRA SFT base** (alpha32_r32, ga=8) bukan rsLoRA.

| Date | Milestone |
|---|---|
| 2026-06-04 | Pilot LoRA V1–V6 (n=25 atau n=100) all trained |
| 2026-06-04 14:42–16:12 | Pilot LoRA eval batch — V1, V2, V4 done; V3 SM crash (libcuda exit=132), V5 SM crash (tokenizers segfault), V6 SM+OOD crash (abort) |
| 2026-06-04 16:12–16:13 | Full LoRA V1 training launched, crashed at step 0 (SIGSEGV exit=139) |
| 2026-06-04 16:13 | Pipeline `set -e` propagation killed entire remaining queue |
| 2026-06-04 late | Diagnosis: NOT hardware (GPU 46°C, healthy) — driver/CUDA software flake (libcuda.so 580.126.09 invalid opcode, recurring across history) |
| 2026-06-04 | User enable `nvidia-smi -pm 1` (persistence mode) |
| 2026-06-04 | Rewrite scripts v2: NO `set -e`, retry 2-3×, 60s cooldown, skip-if-output-exists, `_cuda_env.sh` (expandable_segments + LAZY module loading + OMP=4) |

---

## Phase 7 — Strict Re-check & V7 Plan (Jun 5)

| Time | Discussion point |
|---|---|
| 07:00–07:30 | User minta breakdown per-instance violation SFT (SM + OOD) |
| 07:30–07:40 | User minta breakdown rsLoRA-LLaMA OOD per-instance — failure modes: truncation (LA 15×5) + looping (la16/la20) |
| 07:40–07:45 | User minta sama untuk GRPO V5 dan V6 ck-400 — confirmed V5 fixed looping, V6 better gap |
| 07:45–07:48 | User notice: "Bukannya di V5 semuanya exact ya?" → trace ops_emitted vs ops_expected → V5 ft10 emit 131/100, la16/19/20 over 4-5 ops |
| 07:48–07:53 | User minta verify lewat eval script → temukan **checker quirk** `feasibility.py:78` (cek N pertama per job only, tidak hukum `len(ops_j) > expected_j`). V5 ft10 detail: J0=18 ops, J8=20 ops, semua violations=0 → feasible=True padding |
| 07:53–07:56 | User: "Emitted itu apa? Berarti cap-GRPO lebih buruk?" → klarifikasi: GRPO TIDAK lebih buruk, semua model termasuk SFT padding di ft10 (SFT=136 ops); rsLoRA "lebih baik" gap karena 101 ops dekat tapi infeasible |
| 07:56–08:00 | Build wide-format OOD comparison tabel semua 5 model |
| 08:00–08:05 | Split tabel by shape: in-dist (jobs ≤10) vs OOD shape (jobs >10) — temuan "generalization cliff": V5 12/12 in-dist (100%) tapi 0/6 OOD shape |
| 08:05–08:08 | Final consolidated copy-paste markdown report |
| 08:08–08:15 | **User minta strict re-check** → patch in Python: `strict_feas = orig_feas AND no_over_per_job` + trimmed makespan |
| 08:15 | **Strict re-check verdict**: SEMUA 5 model = **8/18 strict feasibility identik**. Claim "V5 12/18" inflated by checker quirk. |
| 08:15 | Strict mean gap on feasibles: V5 **+6.96%** (terbaik), V6 +8.49%, LoRA-LLaMA +7.53%, rsLoRA +10.73%, SFT +13.61% |
| 08:20 | User minta save report → `reports/OOD_COMPARISON_REPORT.md` + 9 CSV files |
| 08:25 | **V7 plan tercatat**: ganti length-control dengan emitted-control per-job |
| 08:30 | Pipeline relaunched: tmux `grpo_pipeline` PID 91457 PPID=1, V5 symlinked to skip retraining (hemat 12 jam), V6 fresh |
| 08:45 | User question critical: "Bukankah V5/V6 sudah ada bobot completeness?" → trace reward.py: V5 hybrid PUNYA `r_m` (under-emit), coverage gate, length_control mask total token. TIDAK PUNYA per-job over-emit penalty. |
| 08:50 | V5 ft10 reward math konkret: 6.568/7.0 (~94% max) padahal J0=18, J8=20 → RL diajari "padding aman" |
| 08:55 | **V7 redefinisi: minimal delta dari V5** — patch checker tambah `over_op_count`, reward hybrid tambah `r_o = 1.0 jika over=0 else -(over/n_ops)`. NO ganti length_control (tetap berguna cegah looping ekstrim). V7 reward range [-1, 8] |
| 09:00 | Update `memory/grpo_v7_emitted_control_plan.md` dengan minimal-delta strategy |

---

## Phase 8 — Current State (Jun 5, 09:00+)

**Pipeline running** (tmux `grpo_pipeline` PID 91457, PPID=1):
- Pilot eval recovery: V3 SM ✗→retry, V5 SM ✗→retry, V6 SM+OOD ✗→retry
- Full train: V1, V2, V3, V4 from scratch; V5 SKIP (symlinked); V6 fresh
- ETA ~60-73 jam total

**Task list**:
- #42 (pilot eval recovery) — in_progress
- #43 (full GRPO V1-V6 LoRA) — in_progress
- #44 (V7 design: hybrid + r_o + checker patch) — pending, blocked by #43

**Key documents**:
- Memory: `grpo_collapse_findings.md`, `user_prefers_sft_aligned_generation_cap.md`, `grpo_v7_emitted_control_plan.md`
- Reports: `EVALUATION_REPORT.md`, `VIOLATION_REPORT.md`, `MAKESPAN_DISTRIBUTION.md`, `SFT_NOTES.md`, `REWARD_AND_LENGTH_CONTROL.md`, `V5_vs_V6_comparison.md`, `grpo_jssp/EXPERIMENT_NOTES.md`
- Latest: `reports/OOD_COMPARISON_REPORT.md` + 9 CSVs + this chronology

---

## Major Insights Distilled

1. **Length & gradient stability**: V1–V4.2 all collapsed dari length drift + grad spike. Solusi bukan reward shaping (V2 gagal) tapi **advantage masking** (V5 first survival).

2. **Reward dominasi failure mode**: Stratified weight dari OOD violation pattern (missing-heavy) ≠ SM pattern (capacity-heavy). Bobot harus di-recompute dari SM untuk avoid leakage.

3. **Generalization cliff**: GRPO mempertahankan/improve in-distribution feasibility (jobs ≤10), tapi 0/6 di OOD shape (jobs >10). SFT length prior cap ~50 ops/output tidak override-able oleh RL. Solution butuh extend SFT data ke jobs 11-25.

4. **Checker quirk silent bug**: `feasibility.py:78` cek N pertama per job, tidak hukum over-emit. Semua "feasibility gain V5/V6" sebelumnya inflated. Strict re-check: semua model 8/18 identik. Quality (gap) masih signifikan: V5 +6.96% vs SFT +13.61%.

5. **V5/V6 reward NOT penalize per-job over**: V5 hybrid PUNYA under-emit penalty (`r_m`), coverage gate (capped at 1.0 — over di-clip), length_control mask (total token vs threshold). **Per-job over-emit lolos semua signal**.

6. **V7 minimal delta**: tambah `over_op_count` counter di checker + 1 reward komponen `r_o` di hybrid. NO redesign besar.

---

## Open Questions / Pending

- Bagaimana V7 berinteraksi dengan length_control? (Length_control cegah looping total-token; r_o cegah padding per-job. Komplementer, tidak overlap.)
- Apakah strict checker patch perlu retroactive baseline update di README dan EVALUATION_REPORT? (Ya — claim V5 12/18 perlu di-update ke "8/18 strict, 12/18 lenient" atau hanya strict.)
- Apakah V1-V4 (rsLoRA) full LoRA dapat reward stratified yang merefleksikan SM weights yang sudah recomputed? (Need check — apakah `V1_WEIGHTS` di config.py masih dari SM atau OOD lama.)
- ETA aktual pipeline V1-V4+V6 (saat ini estimasi 60-73h berdasar pilot timing).
