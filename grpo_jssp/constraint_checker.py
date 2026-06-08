"""Parse generated JSSP schedule and count 5 violation types.

Thin wrapper over the repo-level feasibility.py so behavior matches the existing
eval pipelines (eval_rslora.py, eval_rslora_benchmarks.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feasibility import (
    extract_makespan,
    parse_schedule_ops_strict,
    validate_feasibility,
)


def check_violations(schedule_str: str, jobs_spec: list) -> dict:
    """Run the 5-category checker on a generated schedule.

    Args:
        schedule_str: model output containing `Jx-My: s+d->e` tokens.
        jobs_spec: list of jobs, each a list of (machine, duration) tuples
                   (the required routing for the instance).

    Returns:
        dict with feasibility flag, per-category counts, totals, makespan.
    """
    ops, timing_bad = parse_schedule_ops_strict(schedule_str)
    feasible, info = validate_feasibility(ops, jobs_spec, timing_bad)
    makespan = extract_makespan(schedule_str) if ops else None

    total = (
        info["missing_op_count"]
        + info["over_op_count"]
        + info["routing_order_violations"]
        + info["machine_capacity_violations"]
        + info["timing_consistency_violations"]
        + info["precedence_violations"]
    )
    return {
        "feasible": feasible,
        "missing_op_count": info["missing_op_count"],
        "over_op_count": info["over_op_count"],
        "routing_order_violations": info["routing_order_violations"],
        "machine_capacity_violations": info["machine_capacity_violations"],
        "timing_consistency_violations": info["timing_consistency_violations"],
        "precedence_violations": info["precedence_violations"],
        "total_violations": total,
        "makespan": makespan if feasible else (makespan if ops else None),
        "ops_emitted": info["ops_emitted"],
        "ops_expected": info["ops_expected"],
    }
