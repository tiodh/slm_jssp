"""CAP-GRPO reward functions for JSSP.

Four modes
----------
hybrid (V4/default)
    R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality
    Per constraint: +1 satisfied / -(n_k/N_ops) violated.
    Structural constraints (+1) gated by op coverage to prevent vacuous zeros.
    R_quality = BKS/Cmax, gated on full feasibility.
    Range: (-inf .. 7.0]
      - Unparseable (hard floor): exactly -1.0
      - Parseable, natural violations (n_k <= N_ops): approx [-3.8 .. 7.0]
      - Parseable, unbounded: -inf possible via excess timing_bad or
        machine_capacity_violations (both uncapped in checker.py)

hybrid_v7
    R = R_v4 + R_O  (over-emit penalty, no coverage gate)
    Range: (-inf .. 8.0]
      - Same floor behaviour as hybrid; max = 8.0 when all constraints met

stratified (V1)
    feasible  : R = min(BKS/Cmax, 1.0)   (or 1.0 if BKS unavailable)
    infeasible: R = -sum_k(w_k * n_k/N_ops)  (frozen V1 weights)
    No parseability floor, no coverage gate — preserved for ablation.

uniform
    unparseable: 0.0  |  infeasible: 1.0  |  feasible: 7.0

Coverage gate (hybrid modes)
    Structural constraints (routing/capacity/timing/precedence) return
    +coverage instead of +1.0 when there are no violations, where
    coverage = ops_emitted / ops_expected. Prevents an empty output from
    collecting free +1 rewards.
"""
from .config import V1_WEIGHTS, GOLD_EST_SLOPE, GOLD_EST_BASE


def _coverage(v: dict) -> float:
    emitted  = v.get("ops_emitted", 0) or 0
    expected = v.get("ops_expected", 0) or 0
    return min(emitted / expected, 1.0) if expected > 0 else 0.0


def _parseable(v: dict) -> bool:
    return (v.get("ops_emitted", 0) or 0) + (v.get("timing_consistency_violations", 0) or 0) > 0


def _stratified_v1(v: dict, n_ops: int, bks) -> float:
    if v["feasible"]:
        cmax = v.get("makespan")
        if bks is not None and cmax is not None and cmax > 0:
            return min(bks / cmax, 1.0)
        return 1.0
    penalty = sum(w * (v.get(k, 0) / n_ops) for k, w in V1_WEIGHTS.items())
    return -penalty


def compute_reward(
    v: dict,
    n_ops: int,
    bks=None,
    mode: str = "hybrid",
    gen_len: int | None = None,
    ended_with_eos: bool | None = None,
    lp_alpha: float = 0.10,
    eos_beta: float  = 0.05,
) -> float:
    """Scalar reward for one generated schedule.

    Args:
        v       : dict from checker.check_violations
        n_ops   : jobs * machines (penalty normaliser N_ops)
        bks     : best-known makespan (None if unavailable)
        mode    : reward mode (see module docstring)
        gen_len / ended_with_eos: required only for stratified_v2
    """
    if n_ops <= 0:
        n_ops = 1

    if mode == "uniform":
        if not _parseable(v):
            return 0.0
        return 7.0 if v["feasible"] else 1.0

    if mode == "stratified":
        return _stratified_v1(v, n_ops, bks)

    if mode == "stratified_v2":
        if gen_len is None or ended_with_eos is None:
            raise ValueError("stratified_v2 requires gen_len and ended_with_eos")
        base     = _stratified_v1(v, n_ops, bks)
        gold_est = GOLD_EST_SLOPE * n_ops + GOLD_EST_BASE
        length_pen = -lp_alpha * max(0.0, (gen_len - gold_est) / gold_est)
        in_band  = 0.5 * gold_est <= gen_len <= 1.5 * gold_est
        bonus    = eos_beta if (ended_with_eos and in_band) else 0.0
        return base + length_pen + bonus

    if mode not in ("hybrid", "hybrid_v7"):
        raise ValueError(f"unknown reward mode: {mode!r}")

    if not _parseable(v):
        return -1.0

    cov      = _coverage(v)
    r_format = 1.0

    missing  = v["missing_op_count"]
    r_m      = 1.0 if missing == 0 else -(missing / n_ops)

    def structural(n_k):
        return cov if n_k == 0 else -(n_k / n_ops)

    r_r = structural(v["routing_order_violations"])
    r_c = structural(v["machine_capacity_violations"])
    r_t = structural(v["timing_consistency_violations"])
    r_p = structural(v["precedence_violations"])

    r_quality = 0.0
    if v["feasible"]:
        cmax = v.get("makespan")
        if bks is not None and cmax is not None and cmax > 0:
            r_quality = min(bks / cmax, 1.0)
        else:
            r_quality = 1.0

    base = r_format + r_m + r_r + r_c + r_t + r_p + r_quality

    if mode == "hybrid":
        return base

    over = v.get("over_op_count", 0)
    r_o  = 1.0 if over == 0 else -(over / n_ops)
    return base + r_o
