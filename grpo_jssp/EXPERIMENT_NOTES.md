# GRPO-JSSP — Catatan Eksperimen

Lanjutan dari rsLoRA-SFT LLaMA 3.1-8B pada StarJob JSSP. Tujuan: meningkatkan
feasibility rate (terutama OOD) lewat Group Relative Policy Optimization
dengan reward stratified.

## Hipotesis

- Reward stratified yang membobotkan 5 jenis violation (routing 0.32, capacity
  0.28, timing 0.26, missing 0.09, precedence 0.06) memberi sinyal lebih kaya
  dibanding reward uniform ±1.
- GRPO dari SFT base bisa memperkecil gap SM → OOD.

## Baseline (SFT) — referensi

Detail SFT rsLoRA (training, eval baseline, pola failure) ada di file terpisah:
**[`../SFT_NOTES.md`](../SFT_NOTES.md)**

Ringkasan untuk konteks GRPO:

| Split | SFT Feasibility | Mean gap | Failure dominan |
|---|---|---|---|
| SM (200, held-out test split) | 190/200 = 95.0% | 6.7% | capacity (64.4%) |
| OOD (18 FT+LA) | 9/18 = 50.0% | 20.1% | missing/truncation (78%) |

SFT base GRPO: `output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800`.

## Setup

### Env (`venv-grpo`)

| Package | Version |
|---|---|
| python | 3.12.3 |
| torch | 2.5.1+cu121 |
| transformers | 4.49.0 |
| trl | 0.15.2 (GRPOTrainer) |
| peft | 0.18.1 |
| accelerate | 1.4.0 |
| unsloth | 2025.3.19 |
| unsloth_zoo | 2025.3.17 |
| bitsandbytes | 0.45.3 |
| xformers | 0.0.28.post3 |
| wandb | 0.27.0 (offline default) |

### Aset

- SFT adapter: `output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230` (PEFT 0.18.1, rsLoRA r=32 α=32)
- Base model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit ~5.4 GB)
- Train set: `data/starjob_train_sm.jsonl` (StarJob SM, jobs/machines ≤10)
- OOD: `data/benchmarks/jobshop1.txt` → 18 instance FT+LA dengan BKS hardcoded

### Catatan setup yang harus diingat
- Unsloth 2025.3.x butuh nama repo lowercase di cache (`models--unsloth--meta-llama-...-bnb-4bit`). Symlink ke cache uppercase lama supaya tidak re-download 5 GB.
- `top_p` bukan field valid di GRPOConfig 0.15.2 — sudah dihapus dari konfigurasi.
- API key wandb sudah tersimpan di `~/.netrc`. Default `WANDB_MODE=offline` di `run.py`; sync manual dengan `wandb sync wandb/offline-run-<id>`.

## Hyperparam (default `grpo_jssp/config.py`)

| Field | Value |
|---|---|
| K_SAMPLES | 4 |
| LEARNING_RATE | 5e-6 |
| NUM_TRAIN_STEPS | 2000 |
| MAX_NEW_TOKENS | 4096 |
| TEMPERATURE | 0.8 |
| KL_COEF (β) | 0.04 |
| WARMUP_STEPS | 20 |
| SAVE_EVERY | 200 |
| GRAD_ACCUM_STEPS | 1 |
| optim | adamw_8bit |
| bf16 | True |
| gradient_checkpointing | True |
| use_vllm | False |

## Reward

```
stratified:
  feasible   → R = min(BKS / Cmax, 1.0)   (1.0 jika BKS tidak ada)
  infeasible → R = -Σ wᵢ · nᵢ / N_ops
uniform:
  feasible   → +1.0; infeasible → -1.0
```

**Bobot stratified (SM-derived, llama+rsLoRA, 200 sampel, 233 violation):**

| Kategori | Count | Bobot |
|---|---|---|
| missing | 4 | 0.0172 |
| routing | 33 | 0.1416 |
| capacity | **145** | **0.6223** |
| timing | 33 | 0.1416 |
| precedence | 18 | 0.0773 |

Sumber: `violations_per_instance_starjob.csv` filter `model=llama AND method=rsLoRA`. Bobot lama (pool 4 model di OOD) sudah ditarik karena (a) leakage, (b) tidak match failure profile model.

BKS untuk SM diturunkan dari max end-time pada gold response; BKS OOD dari tabel `BEST_KNOWN` di `config.py`.

## Log Run

### 2026-05-15 — Smoke test (stratified, 3 step, 5 record)

| Metric | Value |
|---|---|
| train_runtime | 113.3 s |
| time/step | ~38 s |
| reward (stratified) | 0.934 |
| reward_std | 0.052 |
| completion_length | 709.5 |
| KL | 0.262 |
| train_loss | 0.0105 |
| epoch | 0.6 |

Adapter tersimpan: `grpo_jssp/runs/smoke_stratified/{checkpoint-3,final_adapter}`. wandb offline run: `wandb/offline-run-20260515_133456-plm4s5nc`.

**Observasi**: `reward_std` rendah → K=4 sample untuk instance SM cenderung mirip kualitasnya (model SFT sudah sangat baik di SM). Group-relative advantage lemah.

**Aturan held-out (penting):** 18 instance FT+LA OOD **tidak boleh** dipakai untuk training (SFT maupun GRPO). Mereka harus tetap unseen agar valid sebagai test set OOD.

Mitigasi sah (tanpa menyentuh OOD test set):
- Naikkan TEMPERATURE training (0.8 → 1.0–1.2) untuk diversitas sample
- Filter SM training set ke instance lebih sulit (mis. hanya 9×9 / 10×10, atau yang feasibility SFT-nya rendah)
- Pakai `starjob_train.jsonl` (StarJob full, bukan SM saja) — selama ukuran/seed instance-nya berbeda dari benchmark FT+LA
- K↑ (8) — tapi 2× lebih lambat

### 2026-05-15 — Pilot 100 step stratified (ckpt-9800 base, V1)

*Detail eval SFT (T=0.1+do_sample=True, baseline yang dipakai untuk Δ comparison) ada di [`../SFT_NOTES.md`](../SFT_NOTES.md).*


Run: `runs/pilot_stratified_100/final_adapter`. Train runtime 44:38, train_loss 0.0131.

| Split | Baseline (SFT) | Pilot (GRPO 100) | Δ |
|---|---|---|---|
| SM feasibility (200) | 95.0% | 93.5% | −1.5% |
| SM mean gap | 6.7% | 6.1% | −0.6% |
| SM capacity violations | 389 | 132 | **−66%** |
| OOD feasibility (18) | 50.0% | **61.1%** | **+11.1%** |
| OOD mean gap | 20.1% | 39.3% | +19.2% |
| OOD capacity | 30 | **0** | **−100%** |
| OOD missing | 177 | 176 | ≈ |

**Kesimpulan pilot:**
- Reward stratified jelas efektif menekan capacity/routing/timing/precedence (kategori yang punya bobot besar). Capacity hilang 66% di SM dan 100% di OOD.
- Feasibility OOD naik 11% dari memperbaiki la16/la18/la19 (yang baseline-nya rusak besar dengan cmax 4-6× BKS).
- Ft20 + la06–10 (truncation/missing-driven failures) **tidak berubah** — reward stratified tidak menyentuh failure mode ini karena bobot missing kecil (0.017).
- SM feasibility turun tipis −1.5%, kemungkinan noise.

Sinyal: lanjut full 2000 step stratified.

### 2026-05-16 → 2026-05-17 — Full GRPO stratified 2000 step: COLLAPSE

Run: `runs/full_runs/checkpoint-859` (mati pre-finish karena HW hang).

**Outcome:** collapse di step ~700, reward stuck di −0.0172, comp_len pinned 4096, reward_std=0.

**Mekanisme** (lihat memory `grpo_collapse_findings.md` untuk detail):
1. Step 500–700: reward stratified mendorong length hacking (lebih banyak baris → ratio missing turun) — comp_len drift 1700 → 2800
2. Step 700–710: gradient spike grad_norm 1.3 → 11.76 (tidak ada `max_grad_norm` di config lama)
3. Bobot LoRA lompat keluar basin SFT, EOS probability runtuh, output gibberish/loop sampai cap 4096
4. Semua K=4 sample dapat reward identik → reward_std=0 → advantage=0 → absorbing state
5. `KL_COEF=0.04` terlalu lemah untuk tarik balik

**Bukti 4096 token BUKAN karena task complexity:** tokenisasi gold response SM menunjukkan 10×10 (100 ops, ukuran max training) butuh ≤1336 token; linear scaling ~12.5 tok/op. Break-even 4096 di ~18×18 (324 ops), jauh di luar distribusi training. Post-collapse, semua task termasuk 5×5 (gold 300 tok) hit 4096 → pathology model.

**HW hang** terjadi step 859 (independen dari collapse — training sudah mati lebih dulu). Hard freeze 09:31:42 KST tanpa kernel panic/OOM/thermal log. Dugaan: PSU instability under sustained 500W+ load, GPU driver hang, atau NZXT Kraken water cooler (chronic kernel warning "is SATA power connected?"). Sebelum retry panjang: lm-sensors, stress test, cek koneksi Kraken.

### 2026-05-17 — Patch mitigasi (memutuskan + implementasi)

**INVALIDATION NOTICE:** hasil pilot stratified 100 step (2026-05-15) dan full collapse run pakai **reward function lama tanpa length penalty / EOS bonus**. **Tidak comparable** dengan eksperimen pasca-patch ini. Tetap disimpan sebagai bukti mekanisme collapse, **tidak masuk** tabel hasil final.

Hyperparam patch:

| Field | Sebelum | Sesudah |
|---|---|---|
| `KL_COEF` (β) | 0.04 | **0.10** |
| `max_grad_norm` | unset | **1.0** |
| `MAX_NEW_TOKENS` | 4096 | **4096 (TETAP)** — align dengan max_seq_length SFT, tidak diturunkan |

Reward function patch (additive ke base reward, untuk kedua mode stratified/uniform):

```
gold_est       = 12.5 × n_ops + 50
length_pen     = -α × max(0, (gen_len − gold_est) / gold_est),   α=0.10
eos_bonus      = +β  if ended_with_eos AND 0.5·gold_est ≤ gen_len ≤ 1.5·gold_est, β=0.05
                  else 0
```

Empirical shaping table (sanity-tested, n_ops=100, gold_est=1300):

| Skenario | Shaping | Total reward (stratified, feasible BKS=900/Cmax=1000) |
|---|---|---|
| Output bagus 1300 tok + EOS | +0.05 | +0.95 |
| Drift 2000 tok + EOS | −0.054 | n/a |
| **Collapse 4096 tanpa EOS** | **−0.215** | **−0.232** (vs lama −0.017 → 13× lebih punishing) |
| Empty-EOS exploit (10 tok) | 0 | (no bonus) |

### 2026-05-18 — V3: revert shaping + GRAD_ACCUM=4

**Motivasi V3**: V2 collapse di step ~210 (3.3× lebih cepat dari V1 yang collapse di 700).
Analisis menunjukkan length penalty + EOS bonus **mempercepat** absorbing state karena:
1. Reward range menyempit → reward_std lebih cepat shrink ke 0
2. EOS bonus window terlalu sempit, jarang terpicu
3. Length penalty mati saat semua sample saturasi 4096 (konstan ≠ gradient signal)

Akar masalah sebenarnya: **per-group K=4 normalization terlalu sempit** — kalau 1 group
collapse, advantage=0, gradient mati. Solusi struktural: tambah **diversitas prompt antar-update**
via `GRAD_ACCUM=4` (4 prompt unik per weight update, bukan 1).

**Perubahan vs V2:**
- Hapus length penalty + EOS bonus → kembali ke base reward V1
- `KL_COEF`: 0.10 → 0.05 (relax)
- `TEMPERATURE`: 0.8 → 0.7 (sedikit kurangi stokastisitas)
- `GRAD_ACCUM_STEPS`: 1 → **4** ← perubahan utama
- `MAX_GRAD_NORM`: tetap 1.0 (dari V2)
- K samples tetap 4

**Pilot V3 (2026-05-18, N=100, 25 step accum=4, ~44 menit):**

Training metrics akhir (step 25):
- reward=0.836, reward_std=0.145, comp_len=491, kl=0.32, grad_norm=0.29
- **TIDAK ada collapse**: std tetap 0.06-0.15, comp_len 491-575 (jauh dari 4096)

OOD eval (18 instance):
- feasibility=44.4% (8/18) — lebih baik dari V2 (38.9%), di bawah V1 (61.1%)
- **mean_gap_to_bks=10.2%** ← TERBAIK di antara V1/V2/V3
- capacity_violations=17 (V2: 130, V1: 0)
- Hanya 25 weight update, masih undertrained — sudah dekat baseline rsLoRA

SM eval (200 instance) pilot V3: 96.0% feasibility, mean_gap 6.81%.

### 2026-05-18 → 2026-05-19 — Full V3 stratified: COLLAPSE step 265

Run: `runs/full_stratified_n2000_v3/checkpoint-200` (final adapter tidak tersimpan, dihentikan di step ~288).

**Outcome:** collapse di step ~265, mirip V2 tapi 26% lebih lama:
- Step 245: grad_norm spike 3.0 (pre-tipping warning)
- Step 255: reward turun ke +0.24, clen 1053 (tipping)
- Step 260: reward -0.18, clen 1826 (cascade)
- Step 265: reward -0.04, clen 3540, kl drop 0.34→0.17
- Step 270: reward -0.017, clen 4096, std=0, kl=0.066 (collapsed)

Setelah collapse berlangsung 23 step lagi (sampai step 288), step time membengkak dari 110s → 837s/step karena clen saturated 4096. Run dihentikan manual.

**Pelajaran**: `GRAD_ACCUM=4` menunda collapse dari V2 step 210 ke V3 step 265 (+26%), tapi **tidak mencegah**. Pre-tipping signal adalah grad_norm spike, bukan reward_std drop seperti V1/V2.

### 2026-05-19 — V3 UNIFORM mode (mode=uniform, K=4 sample binary reward): DEGRADASI step 130

Run: `runs/full_uniform_n2000_v3/` (tidak ada checkpoint tersimpan, dihentikan sebelum SAVE_EVERY=200).

**Setup**: V3 hyperparams (KL=0.05, accum=4, grad_clip=1.0, T=0.7) + reward mode uniform (binary ±1).

**Hipotesis sebelum run**: binary reward range (±1) menghasilkan reward_std antar-group lebih hidup secara natural → tahan collapse.

**Trajectory:**
- Step 5-100: reward 0.5-0.95 (sehat), std 0.05-0.59 (sangat varied)
- Step 105: reward turun 0.50, clen 842 (tipping)
- Step 110: reward 0.175, clen 1085, grad_norm spike 1.58
- Step 115: reward -0.20, clen 1372 (negatif)
- Step 120: recovery sesaat ke +0.60
- Step 130: catastrophic spike grad_norm 6.59, reward -0.55, clen 1088
- Step 135: reward -0.78, clen 2101 — run dihentikan manual

**Mekanisme berbeda dari V3 stratified**:
- ✅ reward_std TETAP HIDUP (0.30-0.77, tidak mati seperti V3 stratified)
- ✅ kl TIDAK drop (tetap 0.27-0.50)
- ❌ Reward OSCILLATION wild (bouncing ±0.6) bukan monotonic decrease
- ❌ Grad spike masif 6.59 (5× threshold clip 1.0)
- ❌ comp_len drift naik 935→2100

**Pelajaran**: Uniform binary reward bukan jaminan stabil. Model "blind" tanpa per-type signal → tidak bisa identifikasi violation mana yang harus diperbaiki → oscillation → eventual catastrophic gradient spike. **Lebih buruk dari V3 stratified** (tipping di step 105 vs 255 — 2.4× lebih cepat).

### 2026-05-19 — V4-equal (stratified, semua bobot=0.2): DIBATALKAN

Direncanakan tapi setelah analisis methodologi disadari:
- Reward landscape equal-weight justru lebih SEMPIT dari V1 original (worst case -0.2 vs V1 -0.62)
- Konseptual mirip uniform tapi magnitude lebih kecil — hipotesis anti-collapse lemah
- Compute lebih baik dipakai untuk pendekatan yang punya rationale lebih kuat

Dihentikan ~3 menit setelah launch. Config WEIGHTS direvert ke V1 original.

### 2026-05-20 — V4 (Hybrid P-GRPO reward): COLLAPSE step 340

> **CATATAN NAMA:** "V4" mulai entri ini = redesain reward **hybrid P-GRPO**.
> BUKAN "V4-equal" di atas (equal-weights, dibatalkan). Dua hal berbeda.

**Motivasi**: V1-V3 semua collapse via absorbing state — saat ke-K=4 sample
homogen, `reward_std→0` → `advantage→0` → gradient mati. V4 meredesain reward
function agar reward variance tetap hidup walau semua sample sama-sama gagal.

**Reward V4 — 7-komponen aditif** (`grpo_jssp/reward.py`, mode `hybrid`):
```
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality      range [-1, 7]

R_format  : +1 (parseable) / -1 (unparseable — hard floor)
R_k       : +1 jika constraint k puas ; -(n_k / N_ops) jika dilanggar
            k ∈ {missing, routing, capacity, timing, precedence}
            4 constraint struktural (R/C/T/P): +1 di-gate coverage
            coverage = ops_emitted / ops_expected
R_quality : min(BKS/Cmax, 1.0) — HANYA jika fully feasible (gated)
```
Beda kunci vs V1: (a) bobot dihapus, tiap constraint setara — hilangkan tuning
& OOD-leakage; (b) range melebar [-1,1]→[-1,7]; (c) constraint dilanggar memberi
penalti NEGATIF, bukan sekadar mengurangi skor → spread feasible↔infeasible jauh
lebih lebar. **Coverage gating** menutup celah "vacuous +1": tanpa gating, output
~kosong dapat +1 gratis dari 4 constraint yang tak ada isinya untuk dilanggar →
skor palsu ~4.0 (≈ attempt asli). Dengan gating, output 1-op-dari-100 → 0.05.

**Hyperparam**: identik V3 (K=4, T=0.7, KL=0.05, grad_accum=4, max_grad_norm=1.0).
Single-variable experiment — HANYA reward function yang berubah dari V3.

**Full run** (`runs/full_hybrid_n2000_v4/`, max_steps=500, N=2000, grad_accum=4):

| step | %data | reward | clen | grad_norm | catatan |
|---|---|---|---|---|---|
| ~285 | 57% | 6.9 | ~400 | <1 | akhir fase sehat |
| 300 | 60% | 3.81 | 1527 | 1.78 | erosi |
| 335 | 67% | 4.63 | 1000 | **3.45** | tipping (grad spike) |
| 340 | 68% | 1.67 | 2377 | 12.1 | cascade |
| 345 | 69% | -0.56 | 3954 | 10.5 | ~saturasi cap |

Dihentikan manual step 378 (collapse berjalan 38 step, step-time ~700s).

**Collapse step 335-340 = 67% data — TERLAMA dari semua run** (V1 30%, V2 9%,
V3-strat 52%, V3-uniform 23%).

**Temuan kunci — desain V4 SEBAGIAN berhasil:**
- ✅ `reward_std` TIDAK pernah 0 (0.8-2.0 bahkan saat collapse) → absorbing-state
  death yang membunuh V1-V3 TIDAK terjadi. Spread lebar + coverage gating bekerja.
- ❌ V4 tetap collapse — lewat mekanisme BERBEDA: **length escape**. Reward V4
  tidak punya kontrol panjang; output feasible-panjang skornya = feasible-pendek
  → model random-walk di ruang panjang → kena output panjang yang merusak
  struktur → grad spike → policy escape SFT basin.
- KL_COEF=0.05 tidak menarik balik (kl naik 0.32→1.4 saat collapse tapi kalah
  oleh gradient reward, grad_norm 12-34). KL = leash buta, tidak spesifik ke
  ruang-panjang tempat drift terjadi.

**Eval V4 checkpoint-200** (step 200 = 40% data, pre-collapse, sehat):

| Split | Feasibility | Mean gap |
|---|---|---|
| SM (200) | 187/200 = 93.5% | 5.2% |
| OOD ≤10×10 (12) | 8/12 = 67% | 8.3% |
| OOD all-18 | 8/18 = 44% | 8.3% |

### 2026-05-21 — V4.2 (V4 hybrid + grad_accum=2): COLLAPSE step 315

**Motivasi**: hipotesis — V1 feasibility tinggi karena update lebih sering
(grad_accum=1). Bukti pendukung level pilot: pilot V1 (grad_accum=1, 100 update)
OOD ≤10×10 92% vs pilot V3 (grad_accum=4, 25 update) 67% — reward sama, hanya
grad_accum beda. Uji full: reward V4 DITAHAN, hanya `GRAD_ACCUM_STEPS` 4→2.
max_steps=1000 → 1000×2 = 2000 prompt = 1 epoch (budget data sama V4).

**Pilot** (`runs/pilot_hybrid_n100_v4_2/`, 100 rec / 50 step): trajectory sehat,
grad_norm 0.14-0.32, reward 6.3-6.9, tidak ada destabilisasi awal. Sanity lulus.

**Full run** (`runs/full_hybrid_n2000_v4_2/`, grad_accum=2):

| step | %data | reward | clen | grad_norm |
|---|---|---|---|---|
| ~290 | 29% | 6.90 | 271 | 0.93 |
| 300 | 30% | 5.52 | 682 | 0.84 |
| 305 | 30% | 4.99 | 1141 | 2.42 |
| 310 | 31% | 5.29 | 832 | 3.20 |
| 315 | 31.5% | **0.30** | 3518 | cascade |

Dihentikan manual setelah cascade terkonfirmasi step 315.

**Collapse step 315 = 31.5% data — vs V4 68%.** grad_accum=2 collapse ~2× lebih
cepat dari V4. Hasil single-variable bersih (hanya grad_accum yang berubah).

**Kesimpulan V4.2 — hipotesis terbantahkan untuk stabilitas:**
Update lebih sering = pisau bermata dua. Pilot (V1-vs-V3) menunjukkan update
sering → feasibility naik **di horizon pendek**; full run (V4-vs-V4.2)
menunjukkan update sering → **collapse jauh lebih cepat**. Untuk full run,
collapse dini menang. **grad_accum=4 (V4) tetap konfigurasi terbaik.**

**Eval V4.2 checkpoint-200** (step 200 = 20% data, pre-collapse):

| Split | Feasibility | Mean gap |
|---|---|---|
| SM (200) | 193/200 = 96.5% | 6.5% |
| OOD ≤10×10 (12) | 11/12 = 92% | 22.5% |
| OOD all-18 | 11/18 = 61% | 22.5% |

> **Catatan**: 92% OOD menyesatkan kalau dibaca sebagai keunggulan grad_accum=2.
> Itu checkpoint **20%-data** — belum cukup terlatih untuk mendegradasi
> feasibility *maupun* memperbaiki gap (gap 22.5% malah > SFT 20.1%). Pola
> universal: makin banyak data dilatih, feasibility OOD turun & gap membaik.

---

## RANGKUMAN EKSPERIMEN GRPO (V1 → V2 → V3 → V3 uniform)

### Tabel hasil final — full run pre-collapse checkpoints + pilot

#### SM (200 instance)

| Run | Adapter | Feasibility | Median gap | Mean gap | Max gap | Total violation |
|---|---|---|---|---|---|---|
| **SFT (no RL)** | rsLoRA ckpt-9800 | 95.0% | 3.9% | 6.7% | 34.0% | 604 |
| V1 pilot | 100 step | 93.5% | 3.7% | 6.1% | 31.0% | 246 |
| **V1 full ckpt-600** | 600 step pre-collapse | **97.0%** ✅ | **2.9%** | 6.1% | 55.9% | **7** ✅ |
| V2 pilot | 100 step | 95.0% | 3.9% | 7.0% | 65.6% | 957 |
| V2 full ckpt-200 | 200 step pre-collapse | 87.5% ⚠ | 2.5% | 5.9% | 49.1% | 3405 |
| V3 pilot | 25 step (N=100) | 96.0% | 3.8% | 6.8% | 97.9% | 404 |
| **V3 full ckpt-200** | 200 step pre-collapse | 94.0% | 3.9% | **5.5%** | 55.6% | 926 |
| Uniform pilot | 100 step | 96.5% | 4.6% | 6.5% | 41.9% | 311 |
| V3 uniform full | DEGRADED, no adapter | — | — | — | — | — |

#### OOD (18 instance, abaikan missing/truncation — model tidak dilatih untuk instance >10×10)

| Run | Adapter | Feasibility | Median gap | Mean gap | Max gap |
|---|---|---|---|---|---|
| SFT (no RL) | rsLoRA ckpt-9800 | 50.0% | 10.9% | 20.1% | 72.3% |
| **V1 pilot** | 100 step | **61.1%** ✅ | 17.5% | 39.3% | 239.7% ⚠ |
| V1 full ckpt-600 | 600 step pre-collapse | 55.6% | 19.6% | 19.3% | 59.8% |
| V2 pilot | 100 step | 38.9% | 9.8% | 13.2% | 42.6% |
| V2 full ckpt-200 | 200 step pre-collapse | 27.8% ⚠ | 10.9% | 9.6% | 15.1% |
| V3 pilot | 25 step (N=100) | 44.4% | 12.2% | 10.2% | 29.1% |
| **V3 full ckpt-200** | 200 step pre-collapse | 33.3% | **8.9%** | **6.6%** ✅ | **12.2%** ✅ |
| Uniform pilot | 100 step | 55.6% | 12.7% | 30.8% | 169.9% |
| V3 uniform full | DEGRADED, no adapter | — | — | — | — |

### Tipping point analysis (kapan + bagaimana setiap run mulai gagal)

| Run | Tipping step | Collapse step | Window | Mekanisme |
|---|---|---|---|---|
| V1 | **600** | 700 | 100 step (slow) | Length hacking gradual, no shaping no clip |
| V2 | **160** | 210 | 50 step | Reward ridge cliff — length penalty kick-in |
| V3 strat | **255** | 265 | 10 step | Gradient instability — pre-tipping grad spike 3.0 |
| V3 uniform | **105** | ~130 | 25 step (osc) | Reward signal blindness — oscillation + grad spike 6.59 |

### Detail mekanisme failure tiap run

#### V1 (no safeguards) — length hacking
- Reward stratified meng-reward partial matches (parser cocok N/M ops)
- Slow drift: model output lebih panjang → lebih banyak match → reward naik
- clen growth: 500 → 1682 → 4096 dalam 100 step
- Tidak ada grad spike — gradient drift gradual
- Tidak ada length penalty → tidak ada signal untuk berhenti
- Tidak ada max_grad_norm → policy bebas drift
- **Akhir**: clen=4096 saturated, EOS prob runtuh, reward stuck

#### V2 (length penalty + EOS bonus) — reward landscape cliff
- Length penalty α=0.10 hanya aktif saat gen > gold (~1300 tok)
- Sebelum tipping: clen ≤ 600, penalty=0, model bebas tumbuh
- Tipping (step 165): clen jump 514 → 1870 dalam 5 step
- Penalty kick in → reward landscape "cliff" → batch homogen ke reward negatif
- reward_std crash → advantage 0 → gradient mati
- **Length penalty justru mempercepat collapse vs V1**

#### V3 stratified (revert shaping + grad_accum=4) — gradient instability
- Pre-tipping (step 245): grad_norm 3.0 (clipped to 1.0 oleh max_grad_norm)
- Clip mengurangi magnitude tapi tidak mencegah arah update yang buruk
- Step 255: reward turun ke 0.24 (tipping)
- 10-step cascade: clen 1053 → 4096, kl drop 0.34 → 0.05
- **GRAD_ACCUM=4 menunda collapse 26% tapi tidak cegah** — diversitas prompt 4× tidak cukup melawan gradient instability

#### V3 uniform (binary reward + V3 hyperparams) — signal blindness
- Binary reward (±1) tidak memberi gradient direction yang stabil
- Model tidak tahu jenis violation mana yang harus dikurangi
- Reward oscillation kuat: +0.60 → -0.20 → +0.60 → -0.78
- reward_std tetap hidup (0.30-0.77) karena binary
- Grad spike masif 6.59 step 130 (klasik unstable optimizer state)
- **Worst tipping point dari semua run** (step 105 vs V1 600)

### KESIMPULAN: kenapa hasilnya tidak baik

#### 1. Failure mode utama OOD tidak teratasi
**Semua run mempertahankan ~175 missing_op_count di OOD** (~78% violation OOD adalah truncation di instance >10×10). Reward stratified bobot missing=0.017 terlalu kecil untuk memberi signal. Reward uniform tidak granular sama sekali. Model **tidak dilatih untuk instance besar** sejak SFT — keterbatasan ini tidak teratasi oleh GRPO.

#### 2. GRPO inheren rentan collapse di setup K kecil
**Semua run akhirnya collapse (V1, V2, V3 stratified) atau degraded (V3 uniform).** Penyebab struktural:
- Per-group K=4 normalization → satu group homogen = advantage 0 = no gradient
- Tidak ada force keluar dari absorbing state setelah masuk
- Soft reward shaping mati di absorbing state (penalty konstan tidak menghasilkan gradient)

#### 3. Setiap intervensi punya trade-off baru
| Intervensi | Yang diperbaiki | Yang dirusak |
|---|---|---|
| Length penalty (V2) | Mencegah length drift di awal | Mempercepat collapse via reward cliff |
| KL_COEF 0.10 (V2) | Anchor lebih ketat | Konflik dengan length penalty |
| max_grad_norm=1.0 (V2/V3) | Cegah single-step spike | Tidak cegah arah update yang salah |
| GRAD_ACCUM=4 (V3) | Diversitas prompt 4× | Compute 4× lebih besar, tidak cegah collapse |
| Uniform binary (V3 uniform) | Reward range besar (±1) | Hilang gradient direction → oscillation |

#### 4. Tipping point signals tidak konsisten antar run
| Pre-tipping warning | V1 | V2 | V3 strat | V3 uniform |
|---|---|---|---|---|
| grad_norm spike | tidak | tidak | **YA (3.0)** | YA (1.58) |
| reward_std turun | sedikit | tidak | tidak | **NAIK** (osc) |
| clen drift gradual | **YA** | tidak (sudden) | sedikit | YA + osc |
| kl drop | tidak | tidak | sedikit | tidak |

Tidak ada single early-warning yang universal. Detection collapse hanya bisa **post-hoc** dari log.

#### 5. Trade-off feasibility vs gap quality
| Run | Feasibility (recall) | Gap quality (precision) |
|---|---|---|
| V1 | Tinggi (97% SM, 61% OOD) | Lemah (gap 39% OOD, max 240%) |
| V3 stratified | Mid (94% SM, 33% OOD) | **Kuat (gap 5.5% SM, 6.6% OOD)** |
| V3 uniform | (degraded, no data) | (degraded, no data) |

V1 banyak schedule valid tapi kualitas variable. V3 lebih sedikit schedule valid tapi kualitas konsisten dekat optimal. Tidak ada Pareto winner — trade-off inheren.

### Verdict global

**GRPO untuk JSSP dengan setup ini (K=4, LLaMA 8B 4-bit, stratified/uniform reward) tidak menghasilkan superior model dibanding SFT baseline:**

| Metric | Best GRPO | SFT baseline | Δ |
|---|---|---|---|
| SM feasibility | 97.0% (V1 full) | 95.0% | +2% |
| SM total violation | 7 (V1 full) | 604 | −99% ✅ |
| OOD feasibility | 61.1% (V1 pilot) | 50.0% | +11% |
| OOD mean gap (feasible) | 6.6% (V3 full) | 20.1% | −67% ✅ |

GRPO **berhasil** untuk:
- ✅ SM capacity violations (V1 full ckpt-600 dari 389 → 1)
- ✅ OOD gap quality untuk yang feasible (V3 dari 20% → 6.6%)
- ✅ Match BKS optimal lebih sering (V3 25% optimal vs SFT 11%)

GRPO **gagal** untuk:
- ❌ Stability (semua run collapse atau degraded)
- ❌ OOD missing failure mode (constant 175 di semua run)
- ❌ Reliable training pipeline (max_steps ≥ 250 sangat berisiko)

### Implikasi metodologis

1. **Group-relative normalization (per K=4) adalah bottleneck struktural**. Solusi level batch (cross-batch reward norm, V3.1 future) berpotensi mengatasi tapi belum diimplementasi.

2. **Soft reward shaping (length penalty + EOS bonus) terbukti counterproductive** di GRPO K kecil. Shaping aktif saat distribusi bervariasi, mati di absorbing state — tidak ada signal untuk recovery.

3. **Diversitas struktural (grad_accum) > diversitas stokastik (K/temperature)** — tapi tidak cukup. K=4 × accum=4 = 16 reward/update menunda collapse 26%, tidak mencegah.

4. **Uniform binary reward bukan jaminan stabil** — kehilangan gradient direction menyebabkan oscillation yang lebih cepat berakhir buruk dari stratified.

5. **Setiap intervensi pertukaran satu failure mode dengan failure mode lain.** Tidak ada silver bullet — perlu kombinasi (mis. hard cap clen + cross-batch norm + early stopping berdasarkan grad_norm spike).

6. **Truncation OOD fundamentally tidak teratasi GRPO** — model perlu dilatih dengan SFT pada instance lebih besar dulu, kemudian baru GRPO bisa fine-tune.

### Rekomendasi untuk eksperimen berikutnya (di luar scope saat ini)

1. **V3.1 Cross-batch reward normalization**: normalize advantage menggunakan 16 reward total (4 prompt × K=4 sample) bukan per K=4 group. Subclass `GRPOTrainer._prepare_inputs()`. Estimasi ~80-120 baris kode.

2. **Hard truncation cap**: reject sample dengan gen_len > 2× gold_est (treat as -1 reward + zero log_prob gradient). Mencegah length drift mendekati 4096.

3. **Early stopping berdasarkan grad_norm spike**: stop training jika grad_norm (pre-clip) > 5.0 selama 3 step berturut-turut.

4. **Re-train SFT pada instance lebih besar** (15×5, 18×18) untuk menutup gap OOD truncation, kemudian GRPO bisa fine-tune.

5. **Skema reward berbasis BKS bukan violation count** — saat ini reward stratified infeasible = -Σ wᵢ × nᵢ / N_ops tidak bisa membedakan "near-miss" vs "catastrophic violation". Reward kontinyu yang menghadiahi parsial feasibility lebih informatif.

---

## RANGKUMAN ERA V4 (hybrid reward) — V4 + V4.2

### Eval kualitas — semua checkpoint, diurut per % data

Penting: "ckpt-200" TIDAK setara antar versi karena `grad_accum` berbeda
(V1/V2 accum=1, V3/V4 accum=4, V4.2 accum=2). 1 step ≠ jumlah prompt yang sama.
Bandingkan per **% data**, bukan per step.

#### SM (test split 200)

| checkpoint | %data | feasibility | mean gap | total violation |
|---|---|---|---|---|
| SFT baseline | 0% | 95.0% | 6.7% | 599 |
| V2 c200 | 10% | 87.5% | 5.9% | 3405 (collapsed) |
| V4.2 c200 | 20% | 96.5% | 6.5% | 26 |
| V1 c600 | 30% | 97.0% | 6.1% | 7 |
| V3 c200 | 40% | 94.0% | 5.5% | 923 |
| V4 c200 | 40% | 93.5% | 5.2% | 1112 |

#### OOD ≤10×10 (12 instance — instance >10×10 = 0 feasible di semua run)

| checkpoint | %data | feasibility | mean gap |
|---|---|---|---|
| SFT baseline | 0% | 75% | 20.1% |
| V2 c200 | 10% | 42% | 9.6% |
| V4.2 c200 | 20% | 92% | 22.5% |
| V1 c600 | 30% | 83% | 19.3% |
| V3 c200 | 40% | 50% | 6.6% |
| V4 c200 | 40% | 67% | 8.3% |

**Pola universal** (terlihat saat diurut per data): feasibility OOD tinggi di
awal lalu **TURUN**; gap jelek di awal lalu **MEMBAIK** tajam setelah ~30-40%
data. Ada "titik balik" — model berhenti sekadar menjaga feasibility dan mulai
mengoptimasi gap agresif (gap 19%→7%, ongkos feasibility turun). Checkpoint
pre-collapse yang tampak "bagus" di feasibility (V4.2 c200 92%) sebetulnya cuma
"SFT yang baru sedikit bergerak" — bukan prestasi.

### Tipping point — semua run, diurut per % data

| Run | grad_accum | reward | Collapse @ %data | Mekanisme |
|---|---|---|---|---|
| V2 | 1 | stratified+shaping | 9% | reward cliff |
| V3-uniform | 4 | binary ±1 | 23% | signal blindness |
| V1 | 1 | stratified | 30% | length hacking |
| V4.2 | 2 | hybrid | 31% | length escape (accum kecil → noisy) |
| V3-strat | 4 | stratified | 52% | gradient instability |
| **V4** | 4 | hybrid | **68%** | length escape |

V4 = survival terlama. Reward hybrid V4 **memutus** absorbing-state failure-mode
(`reward_std` tak pernah 0), tapi memunculkan **length escape** sebagai
bottleneck baru.

### Verdict era V4

- **V4 (hybrid reward, grad_accum=4) = run terbaik** — collapse terlama (68%
  data), peak reward-utilization tertinggi (99%), satu-satunya yang mematahkan
  absorbing-state death. Tapi belum stabil penuh.
- **V4.2 (grad_accum=2) lebih buruk** — collapse 2× lebih cepat. grad_accum kecil
  → gradient noisy → destabilisasi dini. Hipotesis "update sering lebih baik"
  terbantahkan untuk stabilitas (benar untuk feasibility jangka pendek saja).
- **Failure-mode V4 = length escape**: `completion_length` drift tak terkendali
  → grad spike → policy escape. Reward V4 tidak punya kontrol panjang.
- **Arah V5**: reward V4 + **kontrol panjang langsung** (hard cap completion saat
  training, atau zero-advantage untuk sample `clen > 2× gold_est`). Menyerang
  ruang tempat drift terjadi — bukan lewat KL (leash buta) atau grad_accum.

---

## TODO (revisi 2026-05-21)

- [x] Baseline eval SM + OOD (SFT rsLoRA ckpt-9800)
- [x] V1 pilot 100 step stratified
- [x] V1 full 2000 step → COLLAPSE step 700
- [x] V1 ckpt-600 eval (best pre-collapse)
- [x] V2 (length penalty + EOS bonus): pilot + diagnosis + patch
- [x] V2 full → COLLAPSE step 210 (3.3× lebih cepat dari V1)
- [x] V2 ckpt-200 eval (pre-collapse)
- [x] V3 (revert shaping + grad_accum=4): pilot 25 step LULUS
- [x] V3 full → COLLAPSE step 265 (+26% vs V2 tapi tetap collapse)
- [x] V3 ckpt-200 eval (pre-collapse)
- [x] V3 uniform full → DEGRADED step 130 (oscillation pattern)
- [x] V4-equal (equal-weights) → dibatalkan setelah analisis (terlalu mirip uniform)
- [x] Tipping point analysis untuk V1-V3
- [x] Tabel comparison V1-V3 + analisis kegagalan
- [x] V4 (hybrid P-GRPO reward): full run → COLLAPSE step 340 (68% data, terlama)
- [x] V4 ckpt-200 eval (pre-collapse, 40% data)
- [x] V4.2 (V4 + grad_accum=2): pilot LULUS + full run → COLLAPSE step 315 (31% data)
- [x] V4.2 ckpt-200 eval (pre-collapse, 20% data)
- [x] Rangkuman era V4 + verdict (grad_accum=4 terbaik, hipotesis V4.2 terbantahkan)
- [ ] (Future) **V5 = reward V4 + kontrol panjang** (hard cap clen / zero-advantage clen>2×gold_est)
- [ ] (Future) V3.1 Cross-batch reward normalization (subclass TRL ~80-120 baris)
- [ ] (Future) Early stopping pada grad spike (grad_norm pre-clip > 3.0 berturut)
- [ ] (Future) Re-SFT pada instance >10×10 untuk fix OOD truncation
- [ ] (Future) Reward berbasis BKS bukan violation count

> Catatan config: `GRAD_ACCUM_STEPS` di `config.py` saat ini = 2 (sisa V4.2).
> Untuk run berbasis V4 berikutnya, kembalikan ke 4.
