# GRPO-JSSP — SFT baseline vs V5 (hybrid) vs V6 (stratified)

Both GRPO runs start from the **same rsLoRA-SFT adapter** and use **identical length
control** (advantage masking, `clen > 2.0 * gold_est`, `gold_est = 12.5 * N_ops + 50`)
with the **same data budget** (2000 records, ~1 epoch, 500 steps x 4 grad-accum). The
**only difference between V5 and V6 is the reward function**.

- **SFT** = rsLoRA-SFT baseline (`checkpoint-9800`), no GRPO
- **V5** = `final_adapter`, reward mode **hybrid** (V4), range **[-1, +7]**
- **V6** = `checkpoint-400`, reward mode **stratified** (V1), range **[-1, +1]**

Eval sources:
- `grpo_jssp/eval_results/baseline_sft_{sm,ood}.json`
- `grpo_jssp/eval_results/full_hybrid_lc_n2000_v5_{sm,ood}.json`
- `grpo_jssp/eval_results/full_stratified_lc_n2000_v6_ck400_{sm,ood}.json`

---

## SM test (200 held-out samples, seed=42)

| Metric          | SFT baseline    | V5 hybrid           | V6 stratified   | Best |
|-----------------|-----------------|---------------------|-----------------|------|
| Feasible        | 190/200 (95.0%) | **194/200 (97.0%)** | 190/200 (95.0%) | V5   |
| Mean gap -> BKS | 6.73%           | **5.36%**           | 5.72%           | V5   |
| Median gap->BKS | 3.87%           | 3.02%               | **2.97%**       | V6   |
| missing_op      | 5               | 4                   | **2**           | V6   |
| routing         | 100             | **9**               | 114             | V5   |
| capacity        | 389             | **1**               | 415             | V5   |
| timing          | 40              | **3**               | 51              | V5   |
| precedence      | 70              | **5**               | 80              | V5   |

## OOD (18 FT+LA instances, never trained)

| Metric          | SFT baseline   | V5 hybrid         | V6 stratified | Best |
|-----------------|----------------|-------------------|---------------|------|
| Feasible        | 9/18 (50.0%)   | **12/18 (66.7%)** | 11/18 (61.1%) | V5   |
| Mean gap -> BKS | 20.12%         | 17.44%            | **11.30%**    | V6   |
| Median gap->BKS | **10.91%**     | 10.16%            | 10.91%        | V5   |
| missing_op      | 177            | 177               | 177           | tie (inherited LA 5x5 failure) |
| capacity        | 30             | **0**             | 31            | V5   |
| timing          | 14             | **0**             | 10            | V5   |
| precedence      | 1              | **0**             | **0**         | tie  |

---

## Interpretation

**1. Both GRPO variants beat SFT on the headline metrics.**
On SM, feasibility goes 95% -> 97% (V5) and gap 6.73% -> 5.36% (V5) / 5.72% (V6).
On OOD, feasibility climbs 50% -> 66.7% (V5, +3 instances) / 61.1% (V6, +2), and the
gap shrinks from 20.12% to 17.44% (V5) and 11.30% (V6, **-44% vs SFT**).

**2. The reward choice decides *how* GRPO improves over SFT.**

- **Hybrid (V5) reshapes behavior away from SFT.** It nearly eliminates the SFT
  violation profile: capacity 389 -> 1, routing 100 -> 9, timing 40 -> 3, precedence
  70 -> 5 on SM; and 0 capacity/timing on OOD. This is what drives the feasibility gain.
- **Stratified (V6) leaves the SFT violation profile almost intact.** V6's SM violations
  (routing 114, capacity 415, timing 51, precedence 80) are essentially SFT's numbers
  (100 / 389 / 40 / 70) — and OOD capacity/timing (31/10) match SFT (30/14). The V1
  weights are large in *relative* terms but tiny in *magnitude* (range [-1,1]), so the
  absolute gradient barely moves SFT's behavior. V6 mainly tightens **makespan quality**
  (gap), not feasibility.

**3. Failure profiles diverge.**
On SM, V5 fails only 6 samples with ~22 total violations (near-feasible misses), while
V6 fails 10 samples with ~660 total violations (capacity alone 415) — i.e. V6's failures
are catastrophic, just like SFT's. Hybrid's large-magnitude per-constraint penalty pushes
even failed samples toward near-feasibility; stratified's weak signal does not.

**4. The 177 missing-op count is identical across SFT, V5, and V6 on OOD.**
This confirms the LA 5x5 family failure is **inherited from SFT**, not introduced by
GRPO — neither reward fixes it.

**5. Practical conclusion.**
- Priority = **feasibility / reliability** -> **V5 (hybrid)**: best on SM (97%) and OOD
  (12/18), softest failures, biggest departure from SFT.
- Priority = **solution quality on feasible cases** -> **V6 (stratified)**: best OOD gap
  (11.30%), but inherits SFT's catastrophic-failure mode.
- **SFT baseline** is dominated on every axis by at least one GRPO variant; GRPO
  continuation is a net win, and **hybrid is the stronger general-purpose configuration**.
