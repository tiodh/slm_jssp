"""5-category JSSP feasibility checker.

Categories
----------
- precedence_violations       : within a job, op[i] starts before op[i-1] ends
- routing_order_violations    : wrong machine or duration at routing position i
- timing_consistency_violations: parsed op where start + duration != end
- machine_capacity_violations : two ops on the same machine overlap in time
- missing_op_count            : expected operations never emitted
- over_op_count               : more ops emitted for a job than expected
"""
import re

OP_PATTERN = re.compile(r'J(\d+)-M(\d+):\s*(\d+)\s*\+\s*(\d+)\s*->\s*(\d+)')


def extract_makespan(text: str):
    times = re.findall(r'->\s*(\d+)', text)
    return max(int(t) for t in times) if times else None


def parse_schedule_ops_strict(text: str):
    """Return (valid_ops, timing_inconsistent_count).

    valid_ops : list of (job, machine, start, dur, end) with s+d==e
    """
    valid, timing_bad = [], 0
    for m in OP_PATTERN.finditer(text):
        j, mc, s, d, e = map(int, m.groups())
        if s + d == e:
            valid.append((j, mc, s, d, e))
        else:
            timing_bad += 1
    return valid, timing_bad


def validate_feasibility(ops, jobs, timing_inconsistent_count: int = 0):
    """Return (feasible, info_dict) for a parsed schedule.

    Args:
        ops   : valid ops from parse_schedule_ops_strict
        jobs  : list of jobs, each a list of (machine, duration) tuples
        timing_inconsistent_count: from parse_schedule_ops_strict
    """
    n = len(jobs)
    by_job = {j: [] for j in range(n)}
    extra_ops = 0
    for o in ops:
        if o[0] < n:
            by_job[o[0]].append(o)
        else:
            extra_ops += 1

    precedence_violations   = 0
    routing_order_violations = 0
    missing_op_count         = 0
    over_op_count            = 0

    for j, ops_j in by_job.items():
        ops_j.sort(key=lambda x: x[2])
        expected = jobs[j]
        if len(ops_j) > len(expected):
            over_op_count += len(ops_j) - len(expected)
        if len(ops_j) < len(expected):
            missing_op_count += len(expected) - len(ops_j)
        last_end = 0
        for i, op in enumerate(ops_j[: len(expected)]):
            mc, du = expected[i]
            if op[1] != mc or op[3] != du:
                routing_order_violations += 1
            if op[2] < last_end:
                precedence_violations += 1
            last_end = op[4]

    by_mc = {}
    for o in ops:
        by_mc.setdefault(o[1], []).append(o)
    machine_capacity_violations = 0
    for lst in by_mc.values():
        lst.sort(key=lambda x: x[2])
        for i in range(1, len(lst)):
            if lst[i][2] < lst[i - 1][4]:
                machine_capacity_violations += 1

    feasible = (
        precedence_violations    == 0
        and routing_order_violations == 0
        and timing_inconsistent_count == 0
        and machine_capacity_violations == 0
        and missing_op_count     == 0
        and over_op_count        == 0
    )
    total_expected = sum(len(j) for j in jobs)
    info = {
        "ops_emitted":                   len(ops),
        "ops_expected":                  total_expected,
        "extra_ops":                     extra_ops,
        "precedence_violations":         precedence_violations,
        "routing_order_violations":      routing_order_violations,
        "timing_consistency_violations": timing_inconsistent_count,
        "machine_capacity_violations":   machine_capacity_violations,
        "missing_op_count":              missing_op_count,
        "over_op_count":                 over_op_count,
    }
    return feasible, info


def check_violations(schedule_str: str, jobs_spec: list) -> dict:
    """Full pipeline: parse → validate → return dict with feasibility flag."""
    ops, timing_bad = parse_schedule_ops_strict(schedule_str)
    feasible, info  = validate_feasibility(ops, jobs_spec, timing_bad)
    makespan = extract_makespan(schedule_str)
    total = (
        info["missing_op_count"]
        + info["over_op_count"]
        + info["routing_order_violations"]
        + info["machine_capacity_violations"]
        + info["timing_consistency_violations"]
        + info["precedence_violations"]
    )
    return {
        "feasible":                      feasible,
        "missing_op_count":              info["missing_op_count"],
        "over_op_count":                 info["over_op_count"],
        "routing_order_violations":      info["routing_order_violations"],
        "machine_capacity_violations":   info["machine_capacity_violations"],
        "timing_consistency_violations": info["timing_consistency_violations"],
        "precedence_violations":         info["precedence_violations"],
        "total_violations":              total,
        "makespan":                      makespan,
        "ops_emitted":                   info["ops_emitted"],
        "ops_expected":                  info["ops_expected"],
    }
