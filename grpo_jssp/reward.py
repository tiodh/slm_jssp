"""GRPO-JSSP reward functions.

Four modes:
- "hybrid" (V4): 7-component additive reward, range [-1, 7].
- "uniform" (V4-scale ablation): {-1 unparseable, +1 infeasible, +7 feasible}.
- "stratified" (V1): per-category weighted penalty, range approx [-1, 1].
- "stratified_v2" (V2): V1 stratified + soft length penalty + EOS bonus.

V4 Hybrid (default; used by V4/V5)
----------------------------------
    R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality

Per constraint:
    +1.0          if no violations (satisfied)
    -(n_k/N_ops)  if violations exist (penalty proportional to severity)

R_quality = BKS/Cmax, gated on full feasibility (no makespan hacking).
Range: [-1.0 (unparseable floor) .. 7.0 (feasible + optimal)].

V7 Hybrid (hybrid + over-emit penalty)
--------------------------------------
    R = R_v4_hybrid + R_O                  (V7 only)

R_O graded by count like R_M, no coverage gate (over-emit is by definition
not vacuous). Closes the V5 padding loophole where the model emitted extra
ops per job to satisfy missing-count without producing a valid schedule.
Range: [-1.0 .. 8.0]. V4/V5 hybrid is preserved unchanged for reproducibility.

Coverage gate
-------------
The four structural constraints (routing/capacity/timing/precedence) report
zero violations *vacuously* when the model emits too few ops to test them: an
(almost) empty output has nothing to violate, so the naive design would hand
out free +1s. That makes an empty output worth ~4.0 — close to a genuine
light-violation attempt and far easier to reach, a reward-hacking magnet under
GRPO. So the +1 for those four is gated by op coverage = ops_emitted/ops_expected.
R_missing is always graded by count (the missing count *is* the coverage signal,
so it needs no gate).

V1 Stratified
-------------
    feasible    :  R = min(BKS/Cmax, 1.0)        (or 1.0 if BKS unavailable)
    infeasible  :  R = - Σₖ wₖ · (n_k / N_ops)
Weights frozen at the V1 derivation (config.V1_WEIGHTS). No parseability floor
or coverage gate -- intentionally preserved as the original V1 design so V6
(V1 + length control) is a clean ablation against V1 the way it was originally
run.

V2 Stratified + Soft Length Shaping
-----------------------------------
    R_v2 = R_v1_stratified + length_pen + eos_bonus

    gold_est   = GOLD_EST_SLOPE x N_ops + GOLD_EST_BASE  (default 12.5x + 50)
    length_pen = - alpha * max(0, (gen_len - gold_est) / gold_est)   alpha=0.10
    eos_bonus  = + beta  if ended_with_eos and
                          0.5*gold_est <= gen_len <= 1.5*gold_est    beta=0.05

V2 collapsed at step ~210 in the original LLaMA-rsLoRA run (the worst survival
of any version); preserved here for reproducibility of the V1-V6 ablation.
"""
from grpo_jssp.config import V1_WEIGHTS, GOLD_EST_SLOPE, GOLD_EST_BASE


def _coverage(violations: dict) -> float:
    emitted = violations.get("ops_emitted", 0) or 0
    expected = violations.get("ops_expected", 0) or 0
    if expected <= 0:
        return 0.0
    return min(emitted / expected, 1.0)


def _parseable(violations: dict) -> bool:
    """Parseable = output contains >=1 JSSP-structured token `Jx-My: s+d->e`
    (timing-consistent ones land in ops_emitted, timing-broken ones in
    timing_consistency_violations)."""
    emitted = violations.get("ops_emitted", 0) or 0
    timing_bad = violations.get("timing_consistency_violations", 0) or 0
    return (emitted + timing_bad) > 0


def _stratified_v1(violations: dict, n_ops: int, bks: int | None) -> float:
    if violations["feasible"]:
        cmax = violations.get("makespan")
        if bks is not None and cmax is not None and cmax > 0:
            return min(bks / cmax, 1.0)
        return 1.0
    penalty = 0.0
    for k, w in V1_WEIGHTS.items():
        penalty += w * (violations.get(k, 0) / n_ops)
    return -penalty


def compute_reward(violations: dict, n_ops: int, bks: int = None,
                   mode: str = "hybrid",
                   gen_len: int | None = None,
                   ended_with_eos: bool | None = None,
                   lp_alpha: float = 0.10,
                   eos_beta: float = 0.05) -> float:
    """Scalar reward for one generated schedule.

    Args:
        violations: dict from check_violations (per-category counts + makespan,
                    feasible, ops_emitted, ops_expected).
        n_ops: jobs x machines, the penalty normalizer N_ops.
        bks: best-known makespan for the instance (None if unavailable).
        mode: "hybrid" (V4) | "uniform" | "stratified" (V1) | "stratified_v2" (V2).
        gen_len: token length of the completion (required for V2 only).
        ended_with_eos: whether the completion ended on EOS (required for V2 only).
        lp_alpha: V2 length-penalty coefficient (default 0.10 as in original V2).
        eos_beta: V2 EOS-bonus coefficient (default 0.05 as in original V2).
    """
    if n_ops <= 0:
        n_ops = 1

    if mode == "uniform":
        if not _parseable(violations):
            return 0.0
        return 7.0 if violations["feasible"] else 1.0

    if mode == "stratified":
        # V1: weighted-per-category penalty for infeasible; BKS/Cmax for feasible.
        # No parseability floor and no coverage gate -- original V1 behavior.
        return _stratified_v1(violations, n_ops, bks)

    if mode == "stratified_v2":
        # V2: V1 stratified + soft length shaping. Requires gen_len / ended_with_eos.
        if gen_len is None or ended_with_eos is None:
            raise ValueError("stratified_v2 requires gen_len and ended_with_eos")
        base = _stratified_v1(violations, n_ops, bks)
        gold_est = GOLD_EST_SLOPE * n_ops + GOLD_EST_BASE
        length_pen = -lp_alpha * max(0.0, (gen_len - gold_est) / gold_est)
        in_band = (0.5 * gold_est) <= gen_len <= (1.5 * gold_est)
        bonus = eos_beta if (ended_with_eos and in_band) else 0.0
        return base + length_pen + bonus

    if mode not in ("hybrid", "hybrid_v7"):
        raise ValueError(f"unknown reward mode: {mode}")

    # Unparseable -> hard floor, below any partial attempt.
    if not _parseable(violations):
        return -1.0

    coverage = _coverage(violations)
    r_format = 1.0

    # Missing: always graded by count (missing count IS the coverage signal).
    missing = violations["missing_op_count"]
    r_m = 1.0 if missing == 0 else -(missing / n_ops)

    # Structural constraints: +1 only with coverage; penalty by count otherwise.
    def structural(n_k):
        if n_k == 0:
            return coverage          # vacuous-zero protection
        return -(n_k / n_ops)

    r_r = structural(violations["routing_order_violations"])
    r_c = structural(violations["machine_capacity_violations"])
    r_t = structural(violations["timing_consistency_violations"])
    r_p = structural(violations["precedence_violations"])

    # Quality gate: makespan reward only when fully feasible.
    r_quality = 0.0
    if violations["feasible"]:
        cmax = violations.get("makespan")
        if bks is not None and cmax is not None and cmax > 0:
            r_quality = min(bks / cmax, 1.0)
        else:
            r_quality = 1.0

    base = r_format + r_m + r_r + r_c + r_t + r_p + r_quality

    if mode == "hybrid":
        # V4/V5: preserved unchanged. Range [-1, 7].
        return base

    # V7: hybrid + over-emit penalty. Symmetric to r_m, no coverage gate.
    # Range [-1, 8].
    over = violations.get("over_op_count", 0)
    r_o = 1.0 if over == 0 else -(over / n_ops)
    return base + r_o
