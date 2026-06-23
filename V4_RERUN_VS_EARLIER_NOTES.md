# V4 Rerun (Jun 7) vs Earlier V4 (May 20) — Why no collapse the second time

## TL;DR

**Reward design (formula) tidak berubah.** Komponen dan bobotnya sama persis:
```
R = r_format + r_m + r_r + r_c + r_t + r_p + r_quality
```

Yang berubah adalah **implementasi**: commit V5 (`c314bc6`, 23 Mei 2026) menambahkan
**coverage gate** ke 4 structural reward (r_r, r_c, r_t, r_p) — bukan komponen baru,
hanya bugfix untuk menutup celah reward-hacking. Hyperparameter, SFT base, LoRA
rank, dan library versions semua identik.

## Evidence: Earlier V4 collapse di step ~370

```
step 5    compl_len=407    reward=6.85  ✅
step 200  compl_len=444    reward=6.75  ✅
step 370+ compl_len jumps:
          971 → 999 → 2376 → 3953 → 2998 → 2504 → 1847 → 2786 → 1820 → 1592
step 378  reward 0–3, parseable 2/4, run killed
```

Per-step time membengkak 200s → 600s/it — signature length-escape collapse.

## Apa yang sama antara dua run

| | Earlier V4 (May 20) | Current V4 (Jun 7) |
|---|---|---|
| Trainable params | 83M LoRA r=32 | 83M LoRA r=32 |
| Hyperparameters | K=4, T=0.7, KL=0.05, ga=4 | identik |
| max_steps | 500 | 500 |
| Torch / Unsloth / Triton | 2.5.1 / 2025.3.19 / 3.1.0 | identik |
| SFT base | LLaMA-3.1-8B rsLoRA SFT | identik |
| Reward design | hybrid 7-komponen | identik |

Launch script `_run_full_v4.sh` komen sendiri:
> "Only the reward function changes vs V3 — single-variable experiment."

## Yang berbeda: reward function implementation

`grpo_jssp/reward.py` adalah **new file** di commit V5 (`c314bc6`):
```
diff --git a/grpo_jssp/reward.py b/grpo_jssp/reward.py
new file mode 100644
```

Earlier V4 jalan **3 hari sebelum** file ini di-commit. Implementasi reward
sebelumnya bersifat inline/uncommitted (di `grpo_trainer.py` versi lama atau
script eksperimental).

### Yang ditambahkan di V5 commit (relevan untuk hybrid mode V4)

**Coverage gate** (`reward.py:21-32`):
```python
def _coverage(violations: dict) -> float:
    emitted = violations.get("ops_emitted", 0)
    expected = violations.get("ops_expected", 0)
    return min(emitted / expected, 1.0)
```

Coverage di-multiplied ke 4 structural rewards (r_r, r_c, r_t, r_p). Quote dari
docstring `reward.py`:

> "The four structural constraints report zero violations *vacuously* when the
> model emits too few ops to test them: an (almost) empty output has nothing
> to violate, so the naive design would hand out free +1s. That makes an empty
> output worth ~4.0 — **a reward-hacking magnet under GRPO**."

## Mekanisme collapse earlier V4 (hipotesis)

Tanpa coverage gate, struktur reward menciptakan saddle point:
1. Model emit sedikit op → 4 structural rewards "vacuously satisfied" = +4 gratis
2. Pre-V7 checker juga tidak menghukum over-emit (`over_op_count` belum jadi
   violation sampai commit `553237d`, 8 Juni)
3. Model menemukan jalan tengah: generate panjang dengan banyak op
   duplikat/noise — dapat vacuous-bonus + r_quality dari Cmax kecil yang
   diraport di akhir
4. Length escapes: compl_len 407 → 3953 over ~170 step

## Strictly: ada perubahan reward design?

**Tidak.** Komponen 7-piece dan bobot ±1.0 per komponen sama persis. Yang
berubah:

1. **`c314bc6` (V5, 23 Mei)** — bugfix: coverage gate menutup vacuous-reward
   exploit untuk r_r, r_c, r_t, r_p
2. **`553237d` (V7, 8 Juni)** — bugfix checker: `over_op_count` masuk
   `total_violations` dan menggugurkan feasibility (sebelumnya silently sliced)

Keduanya bugfix, bukan redesign. Hybrid reward formula original tetap utuh.

## Konsekuensi untuk paper

1. **"V1-V4 all collapsed" tidak akurat lagi.** Pernyataan itu berasal dari run
   sebelum commit V5. Dengan implementasi reward saat ini (V5-onwards), V1, V3,
   V4 stabil sampai step 500 dan menghasilkan adapter yang dipakai.

2. **V5 hybrid_lc = V4 hybrid + LC** kurang akurat sebagai story. Lebih tepat:
   ```
   V4 (May 20)    : hybrid + buggy implementation        → collapsed
   V5 (May 23)    : hybrid + coverage gate + LC          → stable
   V4-rerun (Jun7): hybrid + coverage gate (no LC)       → stable too
   ```
   **LC bukan komponen load-bearing untuk stabilitas.** Coverage gate yang
   load-bearing.

3. **Strict OOD comparison sekarang adil**: V4 (no LC) 13/18 vs V5 (LC) 8/18.
   Tanpa LC malah lebih banyak feasible — LC sedikit mendorong under-emit.

4. **Yang masih load-bearing untuk paper**: bagaimana coverage gate +
   `over_op_count` checker patch sama-sama dibutuhkan untuk hasil yang reliable.
   Ini *implementation quality* story, bukan *reward design* story.

## References

- Earlier V4 trajectory: `grpo_jssp/runs/full_hybrid_n2000_v4/training.log` (ends step 378)
- Current V4 trajectory: `grpo_jssp/runs/full_lora_hybrid_n2000_v4/checkpoint-500/trainer_state.json`
- Coverage gate commit: `c314bc6 GRPO V5: first run to survive 500 steps and beat SFT baseline`
- Over-emit checker patch: `553237d V7: hybrid_v7 reward mode + over_op_count checker patch`
- Launch params: `grpo_jssp/_run_full_v4.sh`
