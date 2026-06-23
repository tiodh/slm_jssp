"""Evaluate OpenAI API models on 18 OOD JSSP benchmarks.

Mirrors eval_benchmarks.py prompt format + uses shared feasibility.py checker so
output JSON is directly comparable to metrics_benchmarks_<model>.json from
fine-tuned baselines.

Usage:
    export OPENAI_API_KEY=sk-...
    python eval_openai_benchmarks.py --model gpt-5
    python eval_openai_benchmarks.py --model o3-mini --reasoning-effort medium
    python eval_openai_benchmarks.py --model gpt-4o --budget 0.50

Output: metrics_openai_<model>_benchmarks.json
"""
import os
import sys
import re
import json
import time
import argparse

from openai import OpenAI, OpenAIError, BadRequestError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feasibility import (
    extract_makespan,
    parse_schedule_ops_strict,
    validate_feasibility,
)

BENCH_DIR = "data/benchmarks"
JOBSHOP1 = os.path.join(BENCH_DIR, "jobshop1.txt")

BEST_KNOWN = {
    "ft06": 55, "ft10": 930, "ft20": 1165,
    "la01": 666, "la02": 655, "la03": 597, "la04": 590, "la05": 593,
    "la06": 926, "la07": 890, "la08": 863, "la09": 951, "la10": 958,
    "la16": 945, "la17": 784, "la18": 848, "la19": 842, "la20": 902,
}

WANTED_FT_LA = ["ft06", "ft10", "ft20",
                "la01", "la02", "la03", "la04", "la05",
                "la06", "la07", "la08", "la09", "la10",
                "la16", "la17", "la18", "la19", "la20"]

# Pricing per 1M tokens — VERIFY at platform.openai.com/docs/pricing
# Updated approximate as of Jan 2026 knowledge cutoff.
PRICING = {
    "gpt-5":         (1.25,  10.00),   # placeholder; check current
    "gpt-5-mini":    (0.25,   2.00),
    "gpt-4o":        (2.50,  10.00),
    "gpt-4o-mini":   (0.15,   0.60),
    "gpt-4.1":       (2.00,   8.00),
    "gpt-4.1-mini":  (0.40,   1.60),
    "o3":           (10.00,  40.00),
    "o3-mini":       (1.10,   4.40),
    "o4-mini":       (1.10,   4.40),
}


def parse_orlib_instance(lines):
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        ops = [(toks[2 * k], toks[2 * k + 1]) for k in range(len(toks) // 2)]
        jobs.append(ops)
    return n, m, jobs


def load_jobshop1(path):
    instances = {}
    with open(path) as f:
        text = f.read()
    blocks = re.split(r'\n\s*instance\s+(\w+)\s*\n', text)
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]
        body_lines = [l for l in body.splitlines() if l.strip()
                      and not l.lstrip().startswith('+')
                      and not re.match(r'^\s*[A-Za-z]', l)]
        instances[name] = body_lines
    return instances


def to_starjob_format(n, m, jobs):
    instruction = (f"Optimize schedule for {n} Jobs (denoted as J) across {m} "
                   "Machines (denoted as M) to minimize makespan. The makespan is "
                   "the completion time of the last operation in the schedule. "
                   "Each M can process only one J at a time, and once started, J "
                   "cannot be interrupted.\n\n")
    parts = []
    for j, ops in enumerate(jobs):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    return instruction, "\n".join(parts) + "\n"


def is_reasoning_model(model):
    return model.startswith("o")


def uses_new_completions_api(model):
    """Models that require max_completion_tokens (not max_tokens) and reject temperature."""
    return model.startswith("o") or model.startswith("gpt-5")


# One-shot example: 3x3 JSSP (feasible, makespan=15)
# J0: M1:3 M0:2 M2:2 | J1: M0:2 M2:3 M1:4 | J2: M1:2 M2:4 M0:3
ONE_SHOT_EXAMPLE = {
    "instruction": (
        "Optimize schedule for 3 Jobs (denoted as J) across 3 "
        "Machines (denoted as M) to minimize makespan. The makespan is "
        "the completion time of the last operation in the schedule. "
        "Each M can process only one J at a time, and once started, J "
        "cannot be interrupted.\n\n"
    ),
    "input": "J0:\nM1:3 M0:2 M2:2 \nJ1:\nM0:2 M2:3 M1:4 \nJ2:\nM1:2 M2:4 M0:3 \n",
    "output": (
        "J0-M1: 2+3->5, J0-M0: 5+2->7, J0-M2: 10+2->12\n"
        "J1-M0: 0+2->2, J1-M2: 5+3->8, J1-M1: 8+4->12\n"
        "J2-M1: 0+2->2, J2-M2: 8+4->12, J2-M0: 12+3->15\n"
        "Makespan: 15"
    ),
}


def call_openai(client, model, instruction, input_text, max_tokens,
                reasoning_effort=None, n_shots=0):
    """Single API call, returns response object."""
    format_hint = (
        "Output ONLY the schedule (no explanation) in this exact format:\n"
        "  J{job}-M{machine}: {start}+{duration}->{end}, ...\n"
        "After the last operation, append: Makespan: <value>\n"
    )
    messages = []
    if n_shots >= 1:
        ex = ONE_SHOT_EXAMPLE
        messages.append({"role": "user", "content": (
            f"{ex['instruction']}{ex['input']}\n{format_hint}"
        )})
        messages.append({"role": "assistant", "content": ex["output"]})

    user_msg = f"{instruction}\n{input_text}\n\n{format_hint}"
    messages.append({"role": "user", "content": user_msg})
    kwargs = {"model": model, "messages": messages}
    kwargs = {"model": model, "messages": messages}

    if uses_new_completions_api(model):
        kwargs["max_completion_tokens"] = max_tokens
        # Both gpt-5 and o-series accept reasoning_effort.
        # gpt-5: minimal | low | medium | high  (default medium)
        # o3/o3-mini: low | medium | high
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0

    return client.chat.completions.create(**kwargs)


def compute_cost(usage, model):
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    pin = usage.prompt_tokens
    pout = usage.completion_tokens  # includes reasoning_tokens for o-series
    return (pin * in_price + pout * out_price) / 1_000_000


def family(name):
    if name.startswith("ft"): return "FT"
    if name.startswith("la"): return "LA"
    if name.startswith("ta"): return "TAI"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5",
                    help=f"Choices include: {', '.join(PRICING.keys())}")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="For gpt-5 and o-series. gpt-5 supports 'minimal' (faster, less reasoning).")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="Max output tokens (incl reasoning for o-series)")
    ap.add_argument("--budget", type=float, default=2.0,
                    help="Hard budget cap (USD). Stops eval if exceeded.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON (default: metrics_openai_<model>_benchmarks.json)")
    ap.add_argument("--n-shots", type=int, default=0,
                    help="Number of few-shot examples to prepend (0=zero-shot, 1=single-shot)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set", file=sys.stderr)
        print("Run: export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    if args.model not in PRICING:
        print(f"WARNING: pricing for '{args.model}' unknown, cost tracking disabled")

    client = OpenAI()
    out_path = args.out or f"metrics_openai_{args.model}_benchmarks.json"

    print(f"=== OpenAI OOD eval ===")
    print(f"Model:            {args.model}")
    if is_reasoning_model(args.model):
        print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Max tokens:       {args.max_tokens}")
    print(f"N-shots:          {args.n_shots}")
    print(f"Budget cap:       ${args.budget:.2f}")
    print(f"Output:           {out_path}")
    print()

    js1 = load_jobshop1(JOBSHOP1)

    results = []
    total_cost = 0.0
    t_start = time.time()

    for name in WANTED_FT_LA:
        if total_cost > args.budget:
            print(f"\nBUDGET EXCEEDED ${total_cost:.3f} > ${args.budget:.2f} — stopping")
            break

        if name not in js1:
            print(f"  SKIP {name}: not in jobshop1.txt")
            continue

        n, m, jobs = parse_orlib_instance(js1[name])
        size = f"{n}x{m}"
        bks = BEST_KNOWN.get(name)
        instruction, input_text = to_starjob_format(n, m, jobs)

        t0 = time.time()
        try:
            resp = call_openai(client, args.model, instruction, input_text,
                               args.max_tokens, args.reasoning_effort,
                               n_shots=args.n_shots)
        except (OpenAIError, BadRequestError) as e:
            print(f"  ERROR {name}: {e}")
            continue
        elapsed = time.time() - t0

        cost = compute_cost(resp.usage, args.model)
        total_cost += cost

        raw = resp.choices[0].message.content or ""

        ops, timing_bad = parse_schedule_ops_strict(raw)
        feasible, info = validate_feasibility(ops, jobs, timing_bad)
        makespan = extract_makespan(raw) if ops else None
        gap_pct = None
        if bks and makespan and makespan > 0:
            gap_pct = 100.0 * (makespan - bks) / bks

        rec = {
            "name": name,
            "size": size,
            "best_known": bks,
            "pred": makespan,
            "gap_pct": gap_pct,
            "feasible": feasible,
            "input_tokens": resp.usage.prompt_tokens,
            "gen_tokens": resp.usage.completion_tokens,
            "time_s": round(elapsed, 1),
            "cost_usd": round(cost, 5),
            "raw_output": raw,
            **info,
        }
        results.append(rec)

        gap_s = f"{gap_pct:+.2f}%" if gap_pct is not None else "  NA  "
        pred_s = f"{makespan:>5}" if makespan else " NA  "
        print(f"  {name:6s} {size:>6s}  "
              f"feas={'Y' if feasible else 'N'}  "
              f"pred={pred_s}  "
              f"gap={gap_s}  "
              f"toks={resp.usage.prompt_tokens:>4}+{resp.usage.completion_tokens:>5}  "
              f"t={elapsed:>5.1f}s  "
              f"cost=${cost:.4f}  total=${total_cost:.3f}")

    # Aggregate by family
    by_family = {}
    for fam in ["FT", "LA", "TAI"]:
        recs = [r for r in results if family(r["name"]) == fam]
        if not recs:
            continue
        feas_with_gap = [r for r in recs if r["feasible"] and r["gap_pct"] is not None]
        all_with_gap = [r for r in recs if r["gap_pct"] is not None]
        by_family[fam] = {
            "n": len(recs),
            "valid_parse": sum(1 for r in recs if r["ops_emitted"] > 0),
            "feasible": sum(1 for r in recs if r["feasible"]),
            "violation_totals": {
                k: sum(r[k] for r in recs) for k in
                ["precedence_violations", "routing_order_violations",
                 "timing_consistency_violations", "machine_capacity_violations",
                 "missing_op_count", "over_op_count"]
            },
            "all_mean_gap_pct": round(sum(r["gap_pct"] for r in all_with_gap) / len(all_with_gap), 2) if all_with_gap else None,
            "feas_mean_gap_pct": round(sum(r["gap_pct"] for r in feas_with_gap) / len(feas_with_gap), 2) if feas_with_gap else None,
        }

    overall = {
        "model": f"openai/{args.model}",
        "reasoning_effort": args.reasoning_effort if is_reasoning_model(args.model) else None,
        "max_tokens": args.max_tokens,
        "total_time_min": round((time.time() - t_start) / 60, 2),
        "total_cost_usd": round(total_cost, 4),
        "n_instances_evaluated": len(results),
        "by_family": by_family,
        "results": results,
        "overall_violation_totals": {
            k: sum(r[k] for r in results) for k in
            ["precedence_violations", "routing_order_violations",
             "timing_consistency_violations", "machine_capacity_violations",
             "missing_op_count", "over_op_count"]
        },
    }

    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print(f"=== DONE ===")
    print(f"Total cost:    ${total_cost:.4f}")
    print(f"Instances:     {len(results)}/{len(WANTED_FT_LA)}")
    print(f"Feasible:      {sum(1 for r in results if r['feasible'])}/{len(results)}")
    print(f"Output:        {out_path}")


if __name__ == "__main__":
    main()
