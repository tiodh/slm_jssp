#!/usr/bin/env python3
"""Aggregate ALL evaluation results into paper-ready CSV + markdown tables.

Categories:
  A. GRPO checkpoint evals (eval_results/*.json) — OOD + SM
  B. SFT baseline metrics on Starjob test set (metrics_<model>.json + metrics_lora_/rslora_)
  C. OOD benchmarks on FT+LA (metrics_*_benchmarks.json incl. frontier/OpenAI)

Output:
  reports/paper_grpo_evals.csv
  reports/paper_sft_baselines.csv
  reports/paper_ood_benchmarks.csv
  reports/paper_summary.md
"""

import csv
import glob
import json
import os
import re

ROOT = "/home/tio/Documents/Starjob"
EVAL_DIR = os.path.join(ROOT, "grpo_jssp", "eval_results")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# A. GRPO checkpoint evals
# -----------------------------------------------------------------------------
def collect_grpo_evals():
    """Read all eval_results/*.json with schema {summary, results}.

    Returns rows: model, split (ood/sm), n, feasible, feasibility_pct,
    mean_gap_pct, median_gap_pct, total violations breakdown.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        # parse "<base>_<split>" where split ∈ {ood, sm}
        m = re.match(r"^(.*)_(ood|sm)$", name)
        if not m:
            continue
        base, split = m.group(1), m.group(2)
        try:
            d = json.load(open(path))
        except Exception:
            continue
        s = d.get("summary")
        if not s:
            continue
        row = {
            "model": base,
            "split": split,
            "n": s.get("n"),
            "feasible": s.get("n_feasible"),
            "feasibility_pct": round(s.get("feasibility_rate", 0) * 100, 2),
            "mean_gap_pct": round(s.get("mean_gap_to_bks", 0) * 100, 2),
            "median_gap_pct": round(s.get("median_gap_to_bks", 0) * 100, 2),
            "miss_op_sum": s.get("missing_op_count_sum"),
            "route_viol_sum": s.get("routing_order_violations_sum"),
            "mc_viol_sum": s.get("machine_capacity_violations_sum"),
            "time_viol_sum": s.get("timing_consistency_violations_sum"),
            "prec_viol_sum": s.get("precedence_violations_sum"),
        }
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# B. SFT baseline metrics on Starjob test
# -----------------------------------------------------------------------------
def collect_sft_baselines():
    """Files like metrics_<model>.json (SFT eval on starjob test, schema A),
    metrics_lora_<model>.json / metrics_rslora_<model>.json (schema B: overall+by_size list).

    Both produce: model, variant (sft/lora/rslora), split, n, exact%, gap_le_*,
    mean_gap_pct, median_gap_pct, mae, feasibility_pct, eval_loss.
    For schema B, exact% derived from exact_makespan/feasible; gap_le_* unavailable (None).
    """
    rows = []
    # NOTE on naming:
    #  - All baselines in this project are LoRA-or-rsLoRA fine-tuned via TRL SFTTrainer.
    #    No pure full-parameter SFT exists in the repo.
    #  - metrics_<model>_<size>.json (lora_n50): produced by training-script eval at end
    #    of training, n=50 subset of Starjob test, includes eval_loss.
    #  - metrics_lora_<model>.json (lora_n200): re-run by eval_lora.py on full n=200 test,
    #    explicit method="LoRA" in metadata.
    #  - metrics_rslora_<model>.json (rslora_n200): re-run by eval_rslora.py, method="rsLoRA".
    patterns = [
        ("lora_n50",   os.path.join(ROOT, "metrics_*.json")),
        ("lora",       os.path.join(ROOT, "metrics_lora_*.json")),
        ("rslora",     os.path.join(ROOT, "metrics_rslora_*.json")),
    ]
    seen = set()
    for variant, pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            base = os.path.basename(path)
            if "benchmarks" in base:
                continue
            if variant == "lora_n50" and ("_lora_" in base or "_rslora_" in base):
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                d = json.load(open(path))
            except Exception:
                continue
            model_name = d.get("model_name") or d.get("model") or base.replace("metrics_", "").replace(".json", "")

            # Schema A (SFT) has `by_group`; Schema B (LoRA/rsLoRA) has only `by_size`.
            is_schema_b = ("by_size" in d) and ("by_group" not in d)

            if not is_schema_b:
                # Schema A (SFT): overall + by_group(small/medium/large) with exact_match.pct etc.
                for split_key in ("overall", "small", "medium", "large"):
                    section = d.get(split_key) if split_key == "overall" else d.get("by_group", {}).get(split_key)
                    if not section:
                        continue
                    row = {
                        "model": model_name,
                        "variant": variant,
                        "split": split_key,
                        "n": section.get("n"),
                        "exact_match_pct": section.get("exact_match", {}).get("pct") if isinstance(section.get("exact_match"), dict) else None,
                        "gap_le_5_pct": section.get("gap_le_5pct", {}).get("pct") if isinstance(section.get("gap_le_5pct"), dict) else None,
                        "gap_le_10_pct": section.get("gap_le_10pct", {}).get("pct") if isinstance(section.get("gap_le_10pct"), dict) else None,
                        "gap_le_20_pct": section.get("gap_le_20pct", {}).get("pct") if isinstance(section.get("gap_le_20pct"), dict) else None,
                        "mean_gap_pct": section.get("mean_gap_pct"),
                        "median_gap_pct": section.get("median_gap_pct"),
                        "mae": section.get("mae"),
                        "feasibility_pct": section.get("feasibility_rate", {}).get("pct") if isinstance(section.get("feasibility_rate"), dict) else None,
                        "eval_loss": section.get("eval_loss"),
                    }
                    rows.append(row)
            else:
                # Schema B (LoRA/rsLoRA): overall + by_size [{size, n, ...}]
                o = d.get("overall", {})
                n_total = d.get("total_samples") or o.get("valid")
                feas = o.get("feasible")
                exact = o.get("exact_makespan")
                feas_pct = (feas / n_total * 100) if (feas is not None and n_total) else None
                exact_pct = (exact / n_total * 100) if (exact is not None and n_total) else None
                rows.append({
                    "model": model_name,
                    "variant": variant,
                    "split": "overall",
                    "n": n_total,
                    "exact_match_pct": round(exact_pct, 2) if exact_pct is not None else None,
                    "gap_le_5_pct": None,
                    "gap_le_10_pct": None,
                    "gap_le_20_pct": None,
                    "mean_gap_pct": o.get("mean_gap_pct"),
                    "median_gap_pct": o.get("median_gap_pct"),
                    "mae": None,
                    "feasibility_pct": round(feas_pct, 2) if feas_pct is not None else None,
                    "eval_loss": None,
                })
                # Aggregate by_size into small/medium/large buckets based on instance dims
                buckets = {"small": [], "medium": [], "large": []}
                for s in d.get("by_size", []):
                    size = s.get("size", "")
                    m = re.match(r"(\d+)x(\d+)", size)
                    if not m:
                        continue
                    a, b = int(m.group(1)), int(m.group(2))
                    dim = max(a, b)
                    if dim <= 5:
                        bkt = "small"
                    elif dim <= 10:
                        bkt = "medium"
                    else:
                        bkt = "large"
                    buckets[bkt].append(s)
                for bkt, items in buckets.items():
                    if not items:
                        continue
                    n_b = sum(i.get("n", 0) for i in items)
                    feas_b = sum(i.get("feasible", 0) for i in items)
                    exact_b = sum(i.get("exact_makespan", 0) for i in items)
                    gap_w = sum(i.get("mean_gap_pct", 0) * i.get("n", 0) for i in items)
                    rows.append({
                        "model": model_name,
                        "variant": variant,
                        "split": bkt,
                        "n": n_b,
                        "exact_match_pct": round(exact_b / n_b * 100, 2) if n_b else None,
                        "gap_le_5_pct": None,
                        "gap_le_10_pct": None,
                        "gap_le_20_pct": None,
                        "mean_gap_pct": round(gap_w / n_b, 2) if n_b else None,
                        "median_gap_pct": None,
                        "mae": None,
                        "feasibility_pct": round(feas_b / n_b * 100, 2) if n_b else None,
                        "eval_loss": None,
                    })
    return rows


# -----------------------------------------------------------------------------
# C. OOD benchmarks on FT+LA (metrics_*_benchmarks.json incl OpenAI)
# -----------------------------------------------------------------------------
def collect_ood_benchmarks():
    """Files like metrics_benchmarks_<model>.json (LLM baselines + LoRA + rsLoRA variants),
    metrics_openai_*_benchmarks.json (frontier).

    Returns rows per (model, family in {FT, LA, combined}) with feasibility, violations, gap, cost.
    """
    rows = []
    patterns = [
        "metrics_benchmarks_*.json",
        "metrics_lora_benchmarks_*.json",
        "metrics_rslora_benchmarks_*.json",
        "metrics_openai_*.json",
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            base = os.path.basename(path)
            if path in seen:
                continue
            if "BROKEN" in base:
                continue
            seen.add(path)
            try:
                d = json.load(open(path))
            except Exception:
                continue
            # Determine model display name; disambiguate OpenAI by reasoning_effort
            if "openai" in base:
                m = d.get("model", base.replace(".json", ""))
                eff = d.get("reasoning_effort")
                model_name = f"{m} ({eff})" if eff else m
            elif "rslora_benchmarks" in base:
                model_name = "rslora_" + base.replace("metrics_rslora_benchmarks_", "").replace(".json", "")
            elif "lora_benchmarks" in base:
                model_name = "lora_" + base.replace("metrics_lora_benchmarks_", "").replace(".json", "")
            elif "benchmarks_" in base:
                # LoRA-tuned model benchmark (eval_benchmarks.py loads output_<model>_alpha32_r32_*).
                # File doesn't carry a `method` field but the loaded checkpoint is LoRA.
                model_name = "lora_" + base.replace("metrics_benchmarks_", "").replace(".json", "")
            else:
                continue  # skip uncategorized

            by_family = d.get("by_family", {})
            total_n = sum(by_family.get(f, {}).get("n", 0) for f in ("FT", "LA"))
            total_feas = sum(by_family.get(f, {}).get("feasible", 0) for f in ("FT", "LA"))
            total_cost = d.get("total_cost_usd")
            total_time = d.get("total_time_min")

            for family in ("FT", "LA", "combined"):
                if family == "combined":
                    if total_n == 0:
                        continue
                    # Weighted mean gap
                    gaps_all = []
                    gaps_feas = []
                    for f in ("FT", "LA"):
                        fs = by_family.get(f, {})
                        n_f = fs.get("n", 0)
                        if n_f and fs.get("all_mean_gap_pct") is not None:
                            gaps_all.append((n_f, fs["all_mean_gap_pct"]))
                        if fs.get("feasible", 0) and fs.get("feas_mean_gap_pct") is not None:
                            gaps_feas.append((fs.get("feasible", 0), fs["feas_mean_gap_pct"]))
                    all_gap = (sum(n*g for n,g in gaps_all) / sum(n for n,_ in gaps_all)) if gaps_all else None
                    feas_gap = (sum(n*g for n,g in gaps_feas) / sum(n for n,_ in gaps_feas)) if gaps_feas else None
                    row = {
                        "model": model_name,
                        "family": "combined",
                        "n": total_n,
                        "feasible": total_feas,
                        "feasibility_pct": round(total_feas / total_n * 100, 2),
                        "all_mean_gap_pct": round(all_gap, 2) if all_gap is not None else None,
                        "feas_mean_gap_pct": round(feas_gap, 2) if feas_gap is not None else None,
                        "prec_viol": sum(by_family.get(f, {}).get("violation_totals", {}).get("precedence_violations", 0) for f in ("FT", "LA")),
                        "route_viol": sum(by_family.get(f, {}).get("violation_totals", {}).get("routing_order_violations", 0) for f in ("FT", "LA")),
                        "time_viol": sum(by_family.get(f, {}).get("violation_totals", {}).get("timing_consistency_violations", 0) for f in ("FT", "LA")),
                        "mc_viol": sum(by_family.get(f, {}).get("violation_totals", {}).get("machine_capacity_violations", 0) for f in ("FT", "LA")),
                        "miss_op": sum(by_family.get(f, {}).get("violation_totals", {}).get("missing_op_count", 0) for f in ("FT", "LA")),
                        "time_min": total_time,
                        "cost_usd": total_cost,
                    }
                else:
                    fs = by_family.get(family, {})
                    if not fs:
                        continue
                    n_f = fs.get("n", 0)
                    row = {
                        "model": model_name,
                        "family": family,
                        "n": n_f,
                        "feasible": fs.get("feasible"),
                        "feasibility_pct": round(fs.get("feasible", 0) / n_f * 100, 2) if n_f else None,
                        "all_mean_gap_pct": fs.get("all_mean_gap_pct"),
                        "feas_mean_gap_pct": fs.get("feas_mean_gap_pct"),
                        "prec_viol": fs.get("violation_totals", {}).get("precedence_violations"),
                        "route_viol": fs.get("violation_totals", {}).get("routing_order_violations"),
                        "time_viol": fs.get("violation_totals", {}).get("timing_consistency_violations"),
                        "mc_viol": fs.get("violation_totals", {}).get("machine_capacity_violations"),
                        "miss_op": fs.get("violation_totals", {}).get("missing_op_count"),
                        "time_min": None,
                        "cost_usd": None,
                    }
                rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Write CSV
# -----------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {path})")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


# -----------------------------------------------------------------------------
# Markdown summary
# -----------------------------------------------------------------------------
def build_markdown_summary(grpo_rows, sft_rows, ood_rows):
    lines = []
    lines.append("# Evaluation Metrics — Paper Summary\n")
    lines.append("Generated from `scripts/build_paper_metrics.py`. Raw data in `reports/paper_*.csv`.\n\n")
    lines.append("**Splits:** **OOD** = 18 instances FT+LA (out-of-distribution real benchmarks); **SM** = 200 instances Starjob test (held-out 2%); **Starjob test** = full test split (n=50 subset or n=200 full).\n\n")
    lines.append("**Variant labels (Starjob test, cat B):**\n")
    lines.append("- `lora_n50`: LoRA fine-tune evaluated by training script at end of training (n=50 subset, includes `eval_loss`).\n")
    lines.append("- `lora`: same LoRA family re-evaluated by `eval_lora.py` on full n=200 split.\n")
    lines.append("- `rslora`: rsLoRA fine-tune (`use_rslora=True`) re-evaluated by `eval_rslora.py` on n=200.\n\n")
    lines.append("**Variant labels (OOD bench, cat C):**\n")
    lines.append("- `lora_<model>`: LoRA fine-tune on FT+LA OOD (from `eval_benchmarks.py`).\n")
    lines.append("- `rslora_<model>`: rsLoRA fine-tune on FT+LA OOD (from `eval_rslora_benchmarks.py`).\n\n")
    lines.append("All baselines in this repo are adapter-based (LoRA or rsLoRA) trained via TRL `SFTTrainer`; no pure full-parameter SFT model exists.\n\n---\n")

    # A. GRPO checkpoint evals — group by model base, average across ckpts where applicable
    lines.append("## A. GRPO Evaluations (OOD = FT+LA 18 inst, SM = Starjob test 200 inst)\n")
    lines.append("Per checkpoint where evaluated; final/single-eval models shown as single row.\n\n")
    by_model = {}
    for r in grpo_rows:
        by_model.setdefault(r["model"], {})[r["split"]] = r
    lines.append("| Model | n | OOD feas | OOD feas% | OOD gap% (mean/med) | SM feas | SM feas% | SM gap% (mean/med) |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for model in sorted(by_model.keys()):
        ood = by_model[model].get("ood", {})
        sm = by_model[model].get("sm", {})
        n_ood = ood.get("n", "-")
        ood_f = f"{ood.get('feasible','-')}/{n_ood}" if n_ood != "-" else "-"
        ood_p = f"{ood.get('feasibility_pct','-')}" if ood else "-"
        ood_g = f"{ood.get('mean_gap_pct','-')} / {ood.get('median_gap_pct','-')}" if ood else "-"
        n_sm = sm.get("n", "-")
        sm_f = f"{sm.get('feasible','-')}/{n_sm}" if n_sm != "-" else "-"
        sm_p = f"{sm.get('feasibility_pct','-')}" if sm else "-"
        sm_g = f"{sm.get('mean_gap_pct','-')} / {sm.get('median_gap_pct','-')}" if sm else "-"
        lines.append(f"| {model} | {n_ood} | {ood_f} | {ood_p} | {ood_g} | {sm_f} | {sm_p} | {sm_g} |\n")

    # B. SFT baselines on Starjob test
    lines.append("\n## B. Adapter-tuned Baselines on Starjob Test (LoRA / rsLoRA via TRL SFTTrainer)\n")
    lines.append("\n### B.1 Overall (across all sizes)\n")
    lines.append("| Model | Variant | n | exact% | gap≤5% | gap≤10% | gap≤20% | mean_gap% | median_gap% | feas% | eval_loss |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(sft_rows, key=lambda x: (x["variant"], x["model"], x["split"])):
        if r["split"] != "overall":
            continue
        lines.append(f"| {r['model']} | {r['variant']} | {r.get('n','-')} | {r.get('exact_match_pct','-')} | {r.get('gap_le_5_pct','-')} | {r.get('gap_le_10_pct','-')} | {r.get('gap_le_20_pct','-')} | {r.get('mean_gap_pct','-')} | {r.get('median_gap_pct','-')} | {r.get('feasibility_pct','-')} | {r.get('eval_loss','-')} |\n")

    lines.append("\n### B.2 Per-size breakdown (small/medium/large)\n")
    lines.append("| Model | Variant | Size | n | exact% | mean_gap% | feas% |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in sorted(sft_rows, key=lambda x: (x["variant"], x["model"], ["small","medium","large"].index(x["split"]) if x["split"] in ("small","medium","large") else 99)):
        if r["split"] == "overall":
            continue
        lines.append(f"| {r['model']} | {r['variant']} | {r['split']} | {r.get('n','-')} | {r.get('exact_match_pct','-')} | {r.get('mean_gap_pct','-')} | {r.get('feasibility_pct','-')} |\n")

    # C. OOD benchmarks
    lines.append("\n## C. OOD Benchmarks (FT+LA real instances, 18 total)\n")
    lines.append("| Model | n | feasible | feas% | all_gap% | feas_gap% | prec | route | time | mc | miss_op | cost ($) |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(ood_rows, key=lambda x: (x["model"], 0 if x["family"]=="combined" else 1)):
        if r["family"] != "combined":
            continue
        lines.append(f"| {r['model']} | {r['n']} | {r.get('feasible','-')} | {r.get('feasibility_pct','-')} | {r.get('all_mean_gap_pct','-')} | {r.get('feas_mean_gap_pct','-')} | {r.get('prec_viol','-')} | {r.get('route_viol','-')} | {r.get('time_viol','-')} | {r.get('mc_viol','-')} | {r.get('miss_op','-')} | {r.get('cost_usd','-')} |\n")

    return "".join(lines)


# -----------------------------------------------------------------------------
def main():
    print("== category A: GRPO checkpoint evals ==")
    grpo = collect_grpo_evals()
    write_csv(os.path.join(REPORT_DIR, "paper_grpo_evals.csv"), grpo)

    print("== category B: SFT baselines on Starjob test ==")
    sft = collect_sft_baselines()
    write_csv(os.path.join(REPORT_DIR, "paper_sft_baselines.csv"), sft)

    print("== category C: OOD benchmarks (FT+LA, incl. frontier) ==")
    ood = collect_ood_benchmarks()
    write_csv(os.path.join(REPORT_DIR, "paper_ood_benchmarks.csv"), ood)

    print("== markdown summary ==")
    md = build_markdown_summary(grpo, sft, ood)
    md_path = os.path.join(REPORT_DIR, "paper_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  wrote {md_path}")

    print("\nDONE. Files in reports/:")
    for n in ("paper_grpo_evals.csv", "paper_sft_baselines.csv", "paper_ood_benchmarks.csv", "paper_summary.md"):
        p = os.path.join(REPORT_DIR, n)
        if os.path.exists(p):
            print(f"  {p}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
