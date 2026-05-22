"""V4 Hybrid P-GRPO reward for GRPO on JSSP.

    R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality

Per constraint:
    +1.0          if no violations (satisfied)
    -(n_k/N_ops)  if violations exist (penalty proportional to severity)

R_quality = BKS/Cmax, gated on full feasibility (no makespan hacking).
Range: [-1.0 (unparseable floor) .. 7.0 (feasible + optimal)].

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
"""


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


def compute_reward(violations: dict, n_ops: int, bks: int = None,
                   mode: str = "hybrid") -> float:
    """Scalar reward for one generated schedule.

    Args:
        violations: dict from check_violations (per-category counts + makespan,
                    feasible, ops_emitted, ops_expected).
        n_ops: jobs x machines, the penalty normalizer N_ops.
        bks: best-known makespan for the instance (None if unavailable).
        mode: "hybrid" (V4) | "uniform" (ablation on the V4 scale).
    """
    if n_ops <= 0:
        n_ops = 1

    if mode == "uniform":
        if not _parseable(violations):
            return 0.0
        return 7.0 if violations["feasible"] else 1.0

    if mode != "hybrid":
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

    return r_format + r_m + r_r + r_c + r_t + r_p + r_quality
