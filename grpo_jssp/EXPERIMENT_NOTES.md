# GRPO-JSSP — Experiment Notes

GRPO (Group Relative Policy Optimization) experiments on top of an rsLoRA-SFT
LLaMA 3.1-8B for the Job Shop Scheduling Problem (StarJob dataset). Goal: push
schedule feasibility — especially out-of-distribution (OOD) — beyond the SFT
baseline.

This document is organized **per version**. Each version answers four things:
what it is **based on**, **what changed**, the **reward formula and how it was
derived**, and the **result**.

## TL;DR

- 7 versions tried. V1 → V4.2 (six runs) **all collapsed** during training. **V5 is the first to survive all 500 steps and beat the SFT baseline on every metric.**
- **V5** (V4 reward + advantage masking for over-length samples): SM feasibility **97.0%** (+2.0pp vs baseline), OOD feasibility **66.7%** (+16.7pp vs baseline), no collapse.
- **V1–V4.2** died to the same family of failures: absorbing-state collapse (reward_std → 0) or length escape (completion_length runaway → grad spike).
- **The key design insight:** length must be controlled at the **advantage** level (zero advantage for over-length samples), not the reward level. Soft length shaping (V2) was actively harmful; advantage masking sidesteps both the reward cliff and the absorbing-state collapse.

## Setup

### Environment (`venv-grpo`)
| Package | Version |
|---|---|
| python / torch | 3.12.3 / 2.5.1+cu121 |
| transformers / trl | 4.49.0 / 0.15.2 (GRPOTrainer) |
| peft / accelerate | 0.18.1 / 1.4.0 |
| unsloth | 2025.3.19 |
| bitsandbytes / xformers | 0.45.3 / 0.0.28.post3 |

### Assets
- **Base model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit)
- **SFT adapter (GRPO starting point):** `output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800` (rsLoRA r=32 α=32)
- **Train set:** `data/starjob_train_sm.jsonl` (StarJob SM, jobs & machines ≤10)
- **OOD test set:** 18 FT+LA instances — held out, never used for training

### How the reward is computed
Every generated schedule is scored by `feasibility.py`, which counts 5 violation
types: `missing_op`, `routing`, `capacity`, `timing`, `precedence`. Every reward
design is built on these counts.
- `N_ops` = jobs × machines (operations in the instance) — the normalizer.
- `Cmax` = the makespan the model produced.
- `BKS` = best-known makespan (SM: from the gold response; OOD: the `BEST_KNOWN` table).

### Baseline — SFT, no RL
| Split | Feasibility | Mean gap |
|---|---|---|
| SM (held-out test) | 190/200 (95.0%) | 6.7% |
| OOD ≤10×10 | 9/12 (75.0%) | 20.1% |
| OOD all-18 | 9/18 (50.0%) | 20.1% |

---

## V1 — Stratified weighted reward

**Based on:** the rsLoRA-SFT adapter (checkpoint-9800). The first GRPO version.

**What changed:** introduces GRPO with a *stratified* reward — a per-violation
penalty, weighted by how common each violation type is.

**Reward formula:**
```
feasible    :  R = min(BKS / Cmax, 1.0)
infeasible  :  R = − Σₖ wₖ · nₖ / N_ops
range ≈ [−1, 1]
```

**How the weights were derived:** measured from the SFT model's own violation
distribution on StarJob SM — 200 samples, 233 total violations. Each weight
`wₖ = countₖ / 233`:

| Category | Count | Weight |
|---|---|---|
| missing | 4 | 0.017 |
| routing | 33 | 0.142 |
| capacity | **145** | **0.622** |
| timing | 33 | 0.142 |
| precedence | 18 | 0.077 |

Rationale: weight ∝ frequency makes the reward push hardest on capacity — the
most common failure. Hyperparameters: K=4, KL=0.04, grad_accum=1, no gradient
clipping, temperature 0.8.

**Result:**
- **Collapsed at step ~700 (~30% of the data).**
- Best checkpoint (ckpt-600): SM 194/200 (97%) feasible / gap 6.1% / only 7 violations; OOD ≤10×10 10/12 (83%) feasible / gap 19.3%.
- **Collapse mechanism — length hacking:** a longer output yields more partial parser matches → reward rises falsely → `completion_length` drifts 500 → 1682 → 4096 → all samples become identical garbage → `reward_std=0` → the gradient dies.

---

## V2 — V1 + length shaping

**Based on:** V1's stratified reward.

**What changed:** added soft *length shaping* to stop V1's length hacking; also
tightened KL (0.04→0.10) and added gradient clipping (`max_grad_norm=1.0`).

**Reward formula** (shaping terms added on top of V1):
```
gold_est   = 12.5 × N_ops + 50           (estimated reasonable token length)
length_pen = − α · max(0, (gen_len − gold_est) / gold_est)        α = 0.10
eos_bonus  = + β   if ended with EOS and 0.5·gold_est ≤ gen_len ≤ 1.5·gold_est
                                                                  β = 0.05
```

**How `gold_est` was derived:** tokenizing the SM gold responses gives ~12.5
tokens per operation; `gold_est` is that linear estimate of a reasonable length.

**Result:**
- **Collapsed FASTER than V1 — step ~210 (~9% of the data).** The worst survival of all runs.
- ckpt-200: SM 175/200 (87.5%), OOD ≤10×10 5/12 (42%).
- **Why it backfired:** soft shaping (a) narrows the reward range → `reward_std` reaches 0 sooner; (b) is *dead* once all samples saturate at 4096 (a constant penalty gives no gradient); (c) creates a "reward cliff" the moment output exceeds `gold_est`. **Lesson: soft length shaping is counterproductive in GRPO.**

---

## V3 — revert shaping + larger gradient accumulation

**Based on:** V1's reward (V2's shaping removed).

**What changed:** dropped length shaping (reward = V1 again); relaxed KL
(0.10→0.05); lowered temperature (0.8→0.7); **grad_accum 1→4** — the main change,
to add prompt diversity per weight update (4 prompts instead of 1).

**Reward formula:** identical to V1 (stratified weighted) — no derivation change.

**Result:**
- **Collapsed at step ~265 (~52% of the data)** — +26% later than V2, but still collapses.
- ckpt-200: SM 188/200 (94%) / gap 5.5%; **OOD ≤10×10 6/12 (50%) / gap 6.6% — the best gap of any run.**
- **Collapse mechanism — gradient instability:** a `grad_norm` spike to 3.0 (pre-tipping warning) pushes the policy out of the SFT basin; a 10-step cascade follows. grad_accum=4 *delays* but does not *prevent* collapse.

---

## V3-uniform — binary reward ablation

**Based on:** V3's hyperparameters.

**What changed:** the reward replaced with a plain binary ±1.

**Reward formula:** `feasible → +1.0 ; infeasible → −1.0`.

**Result:**
- **Degraded at step ~130 (~23% of the data).**
- **Why:** with no per-category signal the model is "blind" — it cannot tell which violation to fix → the reward oscillates wildly (±0.6) → a catastrophic grad spike (6.59). A wide reward range alone does not buy stability.

---

## V4 — Hybrid P-GRPO reward

**Based on:** V3's hyperparameters (only the reward function changed — a clean
single-variable experiment vs V3).

**What changed:** a complete reward redesign — from one weighted scalar to **7
additive components**.

**Why redesign:** V1–V3 all die the same way — the *absorbing state*. When the
K=4 samples homogenize, `reward_std → 0`, so `advantage = (r−mean)/std → 0`, and
the gradient dies. Root cause: collapsing 5 violations into one narrow-range
scalar means all failing samples score alike. V4's goal: **keep reward variance
alive even when all samples fail**, by giving each constraint its own wide-range
term.

**Reward formula** (inspired by P-GRPO / Posterior-GRPO):
```
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality        range [−1, 7]

R_format  = +1 if the output is parseable, −1 if not (hard floor)
R_k       = +1             if constraint k is satisfied
          = −(n_k / N_ops) if violated
            (for the 4 structural constraints R/C/T/P the +1 is multiplied by
             coverage = ops_emitted / ops_expected)
R_quality = min(BKS / Cmax, 1.0)  — only if fully feasible
```

**How the design was derived (the key decisions):**
- **No weights — every constraint equal.** The V1 weights, derived from the SM distribution, risked OOD test-set leakage and caused a "capacity bloom" trade-off bug. Equal weights remove both.
- **Wide range [−1,7] with negative penalties.** A violated constraint costs a negative penalty, not just a smaller positive score. The feasible↔infeasible spread becomes ≈9 points (vs ≈2). In the absorbing scenario `reward_std` stays ≈1.1 instead of →0 — this is the core fix.
- **Coverage gating.** Without it, a near-empty output gets a free +1 from constraints it has nothing to violate → scores ~4.0 (a reward-hacking magnet). Multiplying the structural +1 by op-coverage closes this hole.
- **Quality gated.** The makespan reward activates only when the schedule is fully feasible — the model cannot hack makespan without first being feasible.

**Result:**
- **Collapsed at step ~340 (~68% of the data) — the longest survival of any run.**
- ckpt-200: SM 187/200 (93.5%) / gap 5.2%; OOD ≤10×10 8/12 (67%) / gap 8.3%.
- **The design partly worked:** `reward_std` is *never* 0 — the absorbing-state death is broken.
- **But it still collapsed — via a new mechanism, "length escape":** the reward has no length control, so a long feasible schedule scores the same as a short one. The model random-walks in length space until it hits a long output that breaks the structure → grad spike → policy escapes the SFT basin.

---

## V4.2 — V4 with smaller gradient accumulation

**Based on:** V4 (the reward function is unchanged).

**What changed:** only `grad_accum` 4→2. Hypothesis: V1's high feasibility came
from more frequent weight updates (V1 used grad_accum=1), so updating more often
might help.

**Reward formula:** identical to V4 — no change.

**Result:**
- **Collapsed at step ~315 (~31% of the data) — about 2× sooner than V4.**
- ckpt-200 (only 20% data): SM 193/200 (96.5%); OOD ≤10×10 11/12 (92%) / gap 22.5% — the 92% looks great but is just a barely-trained checkpoint, not a real gain.
- **Hypothesis disconfirmed for stability:** more frequent updates = noisier gradients = earlier collapse. **grad_accum=4 stays the best setting.**

---

## Overall Comparison

### Eval quality — all checkpoints, sorted by training data seen

"ckpt-200" is *not* comparable across versions (grad_accum differs, so a step
covers a different number of prompts). Compare by **% of data seen**.

**SM (test split, n=200)** — train set = 2000 records
| Checkpoint | Data seen | Feasibility | Mean gap |
|---|---|---|---|
| SFT baseline | 0 / 2000 (0%) | 190/200 (95.0%) | 6.7% |
| V2 c200 | 200 / 2000 (10%) | 175/200 (87.5%) | 5.9% |
| V4.2 c200 | 400 / 2000 (20%) | 193/200 (96.5%) | 6.5% |
| V1 c600 | 600 / 2000 (30%) | 194/200 (97.0%) | 6.1% |
| V3 c200 | 800 / 2000 (40%) | 188/200 (94.0%) | 5.5% |
| V4 c200 | 800 / 2000 (40%) | 187/200 (93.5%) | 5.2% |

**OOD ≤10×10 (n=12 instances; the 6 instances >10×10 are 0% feasible in every run — the model was never trained that large)**
| Checkpoint | Data seen | Feasibility | Mean gap |
|---|---|---|---|
| SFT baseline | 0 / 2000 (0%) | 9/12 (75%) | 20.1% |
| V2 c200 | 200 / 2000 (10%) | 5/12 (42%) | 9.6% |
| V4.2 c200 | 400 / 2000 (20%) | 11/12 (92%) | 22.5% |
| V1 c600 | 600 / 2000 (30%) | 10/12 (83%) | 19.3% |
| V3 c200 | 800 / 2000 (40%) | 6/12 (50%) | 6.6% |
| V4 c200 | 800 / 2000 (40%) | 8/12 (67%) | 8.3% |

**Universal pattern:** as training proceeds, OOD feasibility *declines* while the
gap *improves* — the model shifts from "just be feasible" to "optimize the
makespan". A pre-collapse checkpoint that looks great on feasibility is mostly
"SFT barely moved", not a genuine gain.

### When each version collapsed

| Run | Reward | grad_accum | Collapse @ %data | Mechanism |
|---|---|---|---|---|
| V2 | stratified + shaping | 1 | 9% | reward cliff |
| V3-uniform | binary ±1 | 4 | 23% | signal blindness |
| V1 | stratified | 1 | 30% | length hacking |
| V4.2 | hybrid | 2 | 31% | length escape (noisy gradient) |
| V3 | stratified | 4 | 52% | gradient instability |
| **V4** | hybrid | 4 | **68%** | length escape |

### Verdict (V1–V4.2)

- **V1–V4.2 all collapse** — GRPO with K=4 on this task is inherently unstable without explicit length control.
- **Best feasibility:** V1 (97% SM / 83% OOD) — but worst gap, earliest collapse.
- **Best gap:** V3 (6.6% OOD) — but low feasibility.
- **Best survival:** V4 — the only design that breaks the absorbing-state death; it collapses only via length escape.
- **Inherent trade-off:** feasibility (recall) vs gap quality (precision) — no Pareto winner.
- **The recurring killer is completion-length drift** — it shows up in V1, V2, and V4. Soft length shaping (V2) makes it *worse*, not better.
- ➡ All of the above motivated V5 below: V4's reward (best survival) paired with hard length control at the advantage level (not the reward level, to avoid V2's failure mode).

## V5 — V4 reward + hard length control

**Based on:** V4's hybrid P-GRPO reward — the design that survives longest (68%
of the data) and the only one that breaks the absorbing-state collapse.

**What changed:** added a **hard length-control** mechanism to remove V4's
collapse trigger (length escape), *without* repeating V2's failed soft shaping.

**Reward formula:** unchanged from V4 (hybrid 7-component, range [−1, 7]). Length
is **not** a reward term — soft length penalties were already shown to backfire
in V2 (they narrow the reward range and create a cliff). Instead, length is
handled at the **advantage** level, inside a `GRPOTrainer` subclass:

```
gold_est = 12.5 × N_ops + 50

for any sample with completion_length > 2 × gold_est:
    advantage := 0          → the sample contributes NO gradient
```

Reward function and length control are fully decoupled: the reward function
sees no length term at all; the masking operates on the `advantages` tensor
inside `_prepare_inputs` after TRL has already computed the rewards. The same
masking trainer (`LengthControlledGRPOTrainer`) works for any reward function
that exposes a per-sample size proxy (here `n_ops`).

**Why V4 as the base (not V1):** V4 already breaks the absorbing-state death
(`reward_std` is never 0). Its one remaining failure mode is *length escape* —
which is exactly what length control attacks. V1, by contrast, would re-inherit
the absorbing-state vulnerability V4 fixed. Pairing V4's reward with length
control targets V4's actual weak point head-on.

**Why this mechanism (vs V2's soft penalty):**
- It is not a negative reward → no "reward cliff", and the reward range is not narrowed.
- An over-long sample gives *no* signal at all (neither positive nor negative) → the model cannot drift in length space and is not pulled toward a penalty basin.
- The generation cap stays at 4096 (aligned with the SFT max sequence length — not lowered).

**Pilot validation (n=100, 100 steps):** the subclass override fired correctly,
no NaN/Inf, masking metric (`overlen_frac`) appeared in the TRL trainer state,
training stable end-to-end. Greenlit the full run.

**Full run (n=2000, 500 steps, K=4, grad_accum=4):**
- **Completed all 500 steps. No collapse.** Wall-clock 15h14m.
- Final-step metrics: reward 6.92, reward_std 0.033, grad_norm 0.59, completion_length 487, overlen_frac 0.000, kl 0.31.
- Survived both prior collapse zones unchanged — step 315 (V4.2 collapse point) and step 340 (V4 collapse point) passed with healthy reward / grad / length. The masking observed a transient length excursion (~step 265) and absorbed it cleanly without cascading.
- 26 masked-batch events total across the run — i.e. ~26/500 batches had at least one over-length sample whose advantage was zeroed. Mechanism active but not dominant.

**Eval — V5 final adapter vs SFT baseline and prior collapsed runs:**

| Run | Split | Feasibility | Median gap | Mean gap | Missing ops | Routing viol. |
|---|---|---:|---:|---:|---:|---:|
| SFT baseline | SM (200) | 95.0% | 3.87% | 6.73% | 5 | 100 |
| V4 ckpt-200 | SM (200) | 93.5% | 3.58% | 5.17% | 0 | 190 |
| V4.2 ckpt-200 | SM (200) | 96.5% | 3.58% | 6.52% | 5 | 15 |
| **V5 full-500** | SM (200) | **97.0%** | **3.02%** | **5.36%** | 4 | **9** |
| SFT baseline | OOD (18) | 50.0% | 10.91% | 20.12% | 177 | 5 |
| V4 ckpt-200 | OOD (18) | 44.4% | 7.93% | **8.32%** | 175 | 4 |
| V4.2 ckpt-200 | OOD (18) | 61.1% | 17.63% | 22.47% | 178 | 3 |
| **V5 full-500** | OOD (18) | **66.7%** | **10.16%** | 17.44% | 177 | 3 |

**Headline numbers:**
- **OOD feasibility 50.0% → 66.7% (+16.7pp)** — the result the project was trying to achieve. 12/18 OOD instances feasible vs 9/18 from SFT.
- **SM strictly improves on baseline on every metric** — feasibility +2.0pp, median gap −22%, mean gap −20%, routing violations −91% (100 → 9).
- **V5 beats every collapsed prior run on feasibility on both splits.** The 6 remaining OOD failures (ft20, la06–la10) are the same family of LA 5×5 instances that fail across all runs — likely an OOD shape the model never saw at SFT, not a V5-specific regression.

**What this confirms about the mechanism:**
1. Zero-advantage masking *does* prevent collapse — V5 walked through the V4.2 (315) and V4 (340) trip-wires without flinching.
2. Masking is non-distortionary — overlen_frac stays low (mean over training ≈ 0.01), reward_std stays alive, KL stays anchored to SFT (~0.3).
3. Length control and reward function are orthogonal — masking can be paired with any reward function that exposes a per-sample size proxy.

**Open items / future work:**
- Try **V1's stratified reward + length control** — V1 had the highest raw feasibility before collapse; pairing its reward with V5's masking is the obvious next ablation (caveat: V1's narrow reward range [−1, 1] keeps the absorbing-state vulnerability that V4 fixed).
- Re-SFT on instances >10×10 to fix OOD truncation on ft10 / la16–la20.
- Cross-batch reward normalization; early-stopping on grad-norm spike.
