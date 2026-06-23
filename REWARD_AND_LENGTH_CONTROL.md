# GRPO-JSSP — Reward Function and Length Control

This document describes the **reward function** and **length-control** mechanism used
in the V5 GRPO-JSSP run (LLaMA-3.1-8B + rsLoRA-SFT adapter, continued with GRPO on the
Job-Shop Scheduling Problem).

Source files:
- Reward: `grpo_jssp/reward.py`
- Length control: `grpo_jssp/grpo_trainer.py` (`LengthControlledGRPOTrainer`)
- Constants: `grpo_jssp/config.py`

---

## 1. Reward function (V5 = V4 "Hybrid P-GRPO")

V5 reuses **V4's reward verbatim** — the only change from V4 to V5 is the addition of
length control (Section 2). The reward is a **7-component additive scalar** with range
**[-1, +7]**:

```
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality

R_format  = +1 if the output is parseable, -1 if not          (hard floor)
R_M       = +1 if missing_op_count == 0, else -(n_missing / N_ops)
R_k       = +1 if constraint k satisfied, else -(n_k / N_ops)  for k in {R, C, T, P}
            (for the 4 structural constraints the +1 is multiplied by
             coverage = ops_emitted / ops_expected)
R_quality = min(BKS / Cmax, 1.0)   — only when the schedule is fully feasible
```

Constraint terms:
- **R_M** — missing operations
- **R_R** — routing-order violations
- **R_C** — machine-capacity violations
- **R_T** — timing-consistency violations
- **R_P** — precedence violations

`N_ops = jobs x machines` is the per-instance penalty normalizer.

### Key design decisions

- **No per-category weights — every constraint is equal.** The earlier V1 reward derived
  weights from the SM violation distribution, which risked OOD test-set leakage and caused
  a "capacity bloom" trade-off bug. Equal weights remove both problems.
- **Wide range [-1, +7] with negative penalties.** A violated constraint costs a *negative*
  penalty, not merely a smaller positive score. The feasible<->infeasible spread becomes
  ~9 points (vs ~2 for the narrow V1 reward). This keeps `reward_std` alive (~1.1) even
  when all K=4 samples fail — directly fixing the *absorbing-state* collapse where
  `reward_std -> 0` kills the advantage `(r - mean)/std -> 0` and the gradient dies.
- **Coverage gating.** Without it, a near-empty output earns a free +1 from the four
  structural constraints it has nothing to violate, scoring ~4.0 — a reward-hacking magnet.
  Multiplying each structural +1 by `coverage = ops_emitted / ops_expected` closes the hole.
  `R_M` is *not* gated, because the missing-op count *is itself* the coverage signal.
- **Quality gating.** The makespan reward `BKS/Cmax` activates **only when the schedule is
  fully feasible**, so the model cannot hack makespan without first being feasible.

---

## 2. Is the reward inspired by P-GRPO? — Yes

The V5/V4 reward is explicitly documented as **inspired by P-GRPO / Posterior-GRPO**
(`config.py:56` labels it "V4 Hybrid P-GRPO"; `EXPERIMENT_NOTES.md:161` states the formula
is "inspired by P-GRPO / Posterior-GRPO"; the V5 section, `EXPERIMENT_NOTES.md:259`, is
"Based on V4's hybrid P-GRPO reward").

How the P-GRPO idea maps onto this design:

- **Posterior / outcome gating.** P-GRPO's central idea is to reward the *process* but gate
  it on the *posterior* (final correctness). Here the **quality term `R_quality` is gated on
  full feasibility** — makespan optimization only counts once the schedule is a valid
  solution. The model cannot collect quality reward by "reasoning well" toward an infeasible
  schedule.
- **Process-level, per-step structure.** Instead of a single outcome scalar, the reward is
  **decomposed per constraint** (`R_M, R_R, R_C, R_T, R_P`), each graded by severity
  `-(n_k / N_ops)`. This gives a dense, process-aware signal — the policy learns *which*
  constraint to fix — analogous to P-GRPO rewarding intermediate process quality rather than
  only the final answer.

Note: this is an **adaptation inspired by** the P-GRPO/Posterior-GRPO principle (gate
process/quality reward on outcome correctness), not a verbatim reproduction of any single
published algorithm. The JSSP-specific constraint decomposition, coverage gating, and the
[-1, 7] scale are this project's own contributions.

---

## 3. Length control (advantage masking)

V4's reward broke the absorbing-state collapse but still died via **"length escape"**: the
reward has no length term, so a long feasible schedule scores the same as a short one. The
model random-walks in length space until a long output breaks structure -> gradient spike ->
the policy escapes the SFT basin.

V5 fixes this **at the advantage level, not the reward level** (a soft length *penalty* was
already shown to backfire in V2 — it narrows the reward range and creates a cliff). Inside a
`GRPOTrainer` subclass (`LengthControlledGRPOTrainer._prepare_inputs`):

```
gold_est = GOLD_EST_SLOPE * N_ops + GOLD_EST_BASE      # = 12.5 * N_ops + 50

for any sample with completion_length > OVERLEN_FACTOR * gold_est:   # factor = 2.0
    advantage := 0     ->  the sample contributes NO gradient (neither reward nor penalty)
```

### Why this mechanism (vs V2's soft penalty)

- **Not a negative reward** -> no reward cliff, and the reward range is not narrowed.
- An over-long sample gives **no signal at all** -> the model cannot drift in length space
  and is not pulled toward a penalty basin.
- **Decoupled from the reward.** The reward function sees no length term; masking operates on
  the `advantages` tensor *after* TRL has computed the rewards. The same trainer works for any
  reward that exposes a per-sample size proxy (here `n_ops`).
- **Generation cap stays at 4096** — aligned with the SFT max sequence length, never lowered.

---

## 4. Constants (config.py)

| Constant            | Value | Meaning                                             |
|---------------------|-------|-----------------------------------------------------|
| `GOLD_EST_SLOPE`    | 12.5  | tokens per operation in the gold-length estimate    |
| `GOLD_EST_BASE`     | 50    | base token budget in the gold-length estimate       |
| `OVERLEN_FACTOR`    | 2.0   | a sample is "over-long" past 2x gold_est            |
| `MAX_NEW_TOKENS`    | 4096  | generation cap (= SFT max seq, not lowered)         |
| `K_SAMPLES`         | 4     | GRPO group size (generations per prompt)            |
| `GRAD_ACCUM_STEPS`  | 4     | prompts per weight update                           |
| `KL_COEF` (beta)    | 0.05  | soft KL anchor to the SFT policy                    |
| `LEARNING_RATE`     | 5e-6  | LoRA-friendly mid-range                             |

---

## 5. Result (V5 full run, n=2000, 500 steps)

- **Completed all 500 steps with no collapse** (15h14m wall-clock) — the first run to do so.
  Walked through both prior collapse trip-wires (step 315 = V4.2, step 340 = V4) unharmed.
- Final-step metrics: reward 6.92, reward_std 0.033, grad_norm 0.59, completion_length 487,
  overlen_frac 0.000, KL 0.31.
- Only ~26/500 batches ever triggered masking — the mechanism is active but not dominant.
- **OOD feasibility 50.0% -> 66.7%** (9/18 -> 12/18) and **SM improves on every metric** vs
  the SFT baseline (feasibility 95.0% -> 97.0%, mean gap 6.73% -> 5.36%, routing violations
  100 -> 9).

This confirms the mechanism: **zero-advantage masking prevents collapse, is non-distortionary
(reward_std stays alive, KL stays anchored), and is orthogonal to the reward function.**
