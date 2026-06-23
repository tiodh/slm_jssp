"""Aggregate 5-category violation breakdown across 4 models x 2 methods.

Inputs (overwritten by re-run):
  metrics_lora_<model>.json            -- Starjob SM, LoRA
  metrics_rslora_<model>.json          -- Starjob SM, rsLoRA
  metrics_benchmarks_<model>.json      -- OOD FT+LA, LoRA
  metrics_rslora_benchmarks_<model>.json -- OOD FT+LA, rsLoRA

Output:
  VIOLATION_REPORT.md                 -- Markdown report
  violation_summary.json              -- machine-readable aggregate
  violations_per_instance_starjob.csv -- 200 x 8 variants per-instance rows
  violations_per_instance_ood.csv     -- 18 x 8 variants per-instance rows
"""
import csv
import json
import os
import sys

MODELS = ["llama", "granite", "ministral", "qwen2"]
MODEL_LABELS = {
    "llama":     "LLaMA 3.1 8B",
    "granite":   "Granite 3.2 8B",
    "ministral": "Ministral 8B",
    "qwen2":     "Qwen2 7B",
}
METHODS = ["LoRA", "rsLoRA"]

VIOL_KEYS = [
    "precedence_violations",
    "routing_order_violations",
    "timing_consistency_violations",
    "machine_capacity_violations",
    "missing_op_count",
]
VIOL_LABELS = {
    "precedence_violations":          "Precedence",
    "routing_order_violations":       "Routing-order",
    "timing_consistency_violations":  "Timing-consistency",
    "machine_capacity_violations":    "Machine-capacity",
    "missing_op_count":               "Missing-op",
}


def load(path):
    """Load metrics file. Treat pre-5-category-validator files as missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    has_new_schema = (
        ("overall" in d and "violation_instance_counts" in d["overall"])
        or ("overall_violation_instance_counts" in d)
    )
    if not has_new_schema:
        return None
    return d


def starjob_path(model, method):
    if method == "LoRA":
        return f"metrics_lora_{model}.json"
    return f"metrics_rslora_{model}.json"


def ood_path(model, method):
    if method == "LoRA":
        return f"metrics_benchmarks_{model}.json"
    return f"metrics_rslora_benchmarks_{model}.json"


def size_bucket_starjob(size_str):
    j, m = (int(x) for x in size_str.split("x"))
    if j <= 5 and m <= 5:
        return "S (<=5x5)"
    return "M (6-10)"


def size_bucket_ood(size_str):
    """Group OOD instances by JxM."""
    return size_str  # 6x6 / 10x5 / 10x10 / 15x5 etc — small enough to leave per-shape


def aggregate_starjob(data):
    """Sum violation counts and feasibility by size bucket from per-instance results."""
    buckets = {}
    for r in data["results"]:
        b = size_bucket_starjob(r["size"])
        bucket = buckets.setdefault(b, {
            "n": 0, "feasible": 0,
            "violation_totals": {k: 0 for k in VIOL_KEYS},
            "violation_instance_counts": {k: 0 for k in VIOL_KEYS},
        })
        bucket["n"] += 1
        if r["feasible"]:
            bucket["feasible"] += 1
        for k in VIOL_KEYS:
            v = r.get(k, 0)
            bucket["violation_totals"][k] += v
            if v > 0:
                bucket["violation_instance_counts"][k] += 1
    return buckets


def aggregate_ood(data):
    """Bucket OOD instances by family (FT, LA) and shape."""
    by_family = {}
    by_shape = {}
    for r in data["results"]:
        fam = r["name"][:2].upper()
        bf = by_family.setdefault(fam, {
            "n": 0, "feasible": 0,
            "violation_totals": {k: 0 for k in VIOL_KEYS},
            "violation_instance_counts": {k: 0 for k in VIOL_KEYS},
        })
        bf["n"] += 1
        if r["feasible"]:
            bf["feasible"] += 1
        for k in VIOL_KEYS:
            v = r.get(k, 0)
            bf["violation_totals"][k] += v
            if v > 0:
                bf["violation_instance_counts"][k] += 1

        sh = r["size"]
        bs = by_shape.setdefault(sh, {
            "n": 0, "feasible": 0,
            "violation_totals": {k: 0 for k in VIOL_KEYS},
            "violation_instance_counts": {k: 0 for k in VIOL_KEYS},
        })
        bs["n"] += 1
        if r["feasible"]:
            bs["feasible"] += 1
        for k in VIOL_KEYS:
            v = r.get(k, 0)
            bs["violation_totals"][k] += v
            if v > 0:
                bs["violation_instance_counts"][k] += 1
    return by_family, by_shape


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt_pct(num, den):
    if den == 0:
        return "-"
    return f"{num}/{den} ({num/den*100:.1f}%)"


def build_report():
    out = []
    out.append("# Violation Breakdown Report — 5 Kategori\n")
    out.append("Re-evaluasi 4 model fine-tuned (LoRA + rsLoRA) pada Starjob SM (200 sampel) "
               "dan OOD FT+LA (18 instance: ft06/10/20, la01-10, la16-20).\n")
    out.append("**Kategori pelanggaran:**")
    out.append("- **Precedence**: dalam satu job, op[i].start < op[i-1].end")
    out.append("- **Routing-order**: op pada posisi rute i memakai mesin / durasi yang salah")
    out.append("- **Timing-consistency**: parsed `s + d != e`")
    out.append("- **Machine-capacity**: dua op pada mesin sama overlap")
    out.append("- **Missing-op**: jumlah operasi yang dijadwalkan kurang dari yang diperlukan\n")

    out.append("---\n")

    # ---- 1. STARJOB SM ----
    out.append("## 1. Starjob SM (in-distribution, 200 sampel)\n")
    out.append("### 1.1 Feasibility rate\n")
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(starjob_path(model, method))
            if d is None:
                rows.append([MODEL_LABELS[model], method, "—", "—", "—"])
                continue
            o = d["overall"]
            valid = o["valid"]
            feas = o["feasible"]
            rows.append([
                MODEL_LABELS[model], method,
                f"{valid}/{d['total_samples']}",
                fmt_pct(feas, valid),
                f"{o.get('mean_gap_pct','-')}%",
            ])
    out.append(md_table(
        ["Model", "Metode", "Valid parse", "Feasible / valid", "Mean gap"],
        rows
    ))
    out.append("")

    # ---- 1.2 Violation totals (instance count where viol > 0) ----
    out.append("### 1.2 Violation incidence (% instance dengan ≥1 pelanggaran tipe X)\n")
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(starjob_path(model, method))
            if d is None:
                rows.append([MODEL_LABELS[model], method] + ["—"] * 5)
                continue
            ic = d["overall"]["violation_instance_counts"]
            n = d["total_samples"]
            row = [MODEL_LABELS[model], method]
            for k in VIOL_KEYS:
                row.append(f"{ic[k]}/{n} ({ic[k]/n*100:.1f}%)" if n else "—")
            rows.append(row)
    out.append(md_table(
        ["Model", "Metode"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
        rows
    ))
    out.append("")

    # ---- 1.3 Violation totals (raw counts) ----
    out.append("### 1.3 Total raw count per kategori (jumlah pelanggaran absolut)\n")
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(starjob_path(model, method))
            if d is None:
                rows.append([MODEL_LABELS[model], method] + ["—"] * 5)
                continue
            tot = d["overall"]["violation_totals"]
            row = [MODEL_LABELS[model], method] + [tot[k] for k in VIOL_KEYS]
            rows.append(row)
    out.append(md_table(
        ["Model", "Metode"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
        rows
    ))
    out.append("")

    # ---- 1.4 By size bucket: S vs M ----
    out.append("### 1.4 Distribusi pelanggaran per ukuran instance (S=≤5x5, M=6-10)\n")
    out.append("Format: `instance_count_with_viol / n_in_bucket`. (Total raw counts dalam tanda kurung.)\n")
    for bucket in ["S (<=5x5)", "M (6-10)"]:
        out.append(f"#### Bucket {bucket}\n")
        rows = []
        for model in MODELS:
            for method in METHODS:
                d = load(starjob_path(model, method))
                if d is None:
                    rows.append([MODEL_LABELS[model], method, "—"] + ["—"] * 5)
                    continue
                buckets = aggregate_starjob(d)
                b = buckets.get(bucket, {"n": 0, "feasible": 0,
                                         "violation_totals": {k: 0 for k in VIOL_KEYS},
                                         "violation_instance_counts": {k: 0 for k in VIOL_KEYS}})
                row = [MODEL_LABELS[model], method,
                       fmt_pct(b["feasible"], b["n"])]
                for k in VIOL_KEYS:
                    ic = b["violation_instance_counts"][k]
                    tot = b["violation_totals"][k]
                    row.append(f"{ic}/{b['n']} ({tot})" if b["n"] else "—")
                rows.append(row)
        out.append(md_table(
            ["Model", "Metode", "Feasible"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
            rows
        ))
        out.append("")

    # ---- 1.5 Per exact size ----
    out.append("### 1.5 Per exact size (count + total raw violation per kategori)\n")
    for model in MODELS:
        for method in METHODS:
            d = load(starjob_path(model, method))
            if d is None:
                continue
            out.append(f"#### {MODEL_LABELS[model]} ({method})\n")
            rows = []
            for s in d["by_size"]:
                row = [s["size"], s["n"],
                       fmt_pct(s["feasible"], s["n"])]
                for k in VIOL_KEYS:
                    ic = s["violation_instance_counts"][k]
                    tot = s["violation_totals"][k]
                    row.append(f"{ic} ({tot})")
                rows.append(row)
            out.append(md_table(
                ["Size", "n", "Feasible"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
                rows
            ))
            out.append("")

    out.append("---\n")

    # ---- 2. OOD ----
    out.append("## 2. OOD: FT + LA (18 instance, max_new_tokens=7000)\n")
    out.append("### 2.1 Feasibility rate\n")
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(ood_path(model, method))
            if d is None:
                rows.append([MODEL_LABELS[model], method, "—", "—"])
                continue
            results = d["results"]
            n = len(results)
            feas = sum(1 for r in results if r["feasible"])
            valid = sum(1 for r in results if r["pred"] is not None)
            rows.append([
                MODEL_LABELS[model], method,
                f"{valid}/{n}",
                fmt_pct(feas, n),
            ])
    out.append(md_table(
        ["Model", "Metode", "Valid parse", "Feasible / total"],
        rows
    ))
    out.append("")

    # ---- 2.2 Violation incidence overall (FT+LA) ----
    out.append("### 2.2 Violation incidence (% instance dengan pelanggaran tipe X)\n")
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(ood_path(model, method))
            if d is None:
                rows.append([MODEL_LABELS[model], method] + ["—"] * 5)
                continue
            n = len(d["results"])
            ic = d.get("overall_violation_instance_counts",
                       {k: sum(1 for r in d["results"] if r.get(k, 0) > 0) for k in VIOL_KEYS})
            row = [MODEL_LABELS[model], method]
            for k in VIOL_KEYS:
                row.append(f"{ic[k]}/{n} ({ic[k]/n*100:.1f}%)")
            rows.append(row)
    out.append(md_table(
        ["Model", "Metode"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
        rows
    ))
    out.append("")

    # ---- 2.3 Per family ----
    out.append("### 2.3 Per family (FT vs LA): feasibility + incidence\n")
    for fam in ["FT", "LA"]:
        out.append(f"#### {fam} family\n")
        rows = []
        for model in MODELS:
            for method in METHODS:
                d = load(ood_path(model, method))
                if d is None:
                    rows.append([MODEL_LABELS[model], method, "—"] + ["—"] * 5)
                    continue
                by_family, _ = aggregate_ood(d)
                b = by_family.get(fam, {"n": 0, "feasible": 0,
                                         "violation_totals": {k: 0 for k in VIOL_KEYS},
                                         "violation_instance_counts": {k: 0 for k in VIOL_KEYS}})
                row = [MODEL_LABELS[model], method, fmt_pct(b["feasible"], b["n"])]
                for k in VIOL_KEYS:
                    ic = b["violation_instance_counts"][k]
                    tot = b["violation_totals"][k]
                    row.append(f"{ic}/{b['n']} ({tot})")
                rows.append(row)
        out.append(md_table(
            ["Model", "Metode", "Feasible"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
            rows
        ))
        out.append("")

    # ---- 2.4 Per shape (JxM) ----
    out.append("### 2.4 Per ukuran (JxM) untuk OOD\n")
    out.append("Format: `inst dengan viol / n_di_shape (total raw count)`.\n")
    # Collect all shapes seen
    all_shapes = set()
    for model in MODELS:
        for method in METHODS:
            d = load(ood_path(model, method))
            if d is None:
                continue
            for r in d["results"]:
                all_shapes.add(r["size"])
    shapes_sorted = sorted(all_shapes,
                           key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])))
    for shape in shapes_sorted:
        out.append(f"#### Shape {shape}\n")
        rows = []
        for model in MODELS:
            for method in METHODS:
                d = load(ood_path(model, method))
                if d is None:
                    rows.append([MODEL_LABELS[model], method, "—"] + ["—"] * 5)
                    continue
                _, by_shape = aggregate_ood(d)
                b = by_shape.get(shape, {"n": 0, "feasible": 0,
                                          "violation_totals": {k: 0 for k in VIOL_KEYS},
                                          "violation_instance_counts": {k: 0 for k in VIOL_KEYS}})
                row = [MODEL_LABELS[model], method, fmt_pct(b["feasible"], b["n"])]
                for k in VIOL_KEYS:
                    ic = b["violation_instance_counts"][k]
                    tot = b["violation_totals"][k]
                    row.append(f"{ic}/{b['n']} ({tot})" if b["n"] else "—")
                rows.append(row)
        out.append(md_table(
            ["Model", "Metode", "Feasible"] + [VIOL_LABELS[k] for k in VIOL_KEYS],
            rows
        ))
        out.append("")

    # ---- 2.5 Per instance ----
    out.append("### 2.5 Per instance (FT+LA) — feasibility per (model, method)\n")
    instances = []
    for model in MODELS:
        for method in METHODS:
            d = load(ood_path(model, method))
            if d:
                for r in d["results"]:
                    instances.append(r["name"])
                break
        if instances:
            break
    inst_order = sorted(set(instances),
                        key=lambda x: (x[:2], int(x[2:])))
    headers = ["Instance", "Size"]
    for model in MODELS:
        for method in METHODS:
            headers.append(f"{model[:3]}/{method[:1].lower()}")
    rows = []
    for inst in inst_order:
        size = ""
        row = [inst]
        for model in MODELS:
            for method in METHODS:
                d = load(ood_path(model, method))
                if d is None:
                    row.append("—")
                    continue
                rec = next((r for r in d["results"] if r["name"] == inst), None)
                if rec is None:
                    row.append("—")
                    continue
                if not size:
                    size = rec["size"]
                if rec["feasible"]:
                    row.append("F")
                else:
                    flags = []
                    if rec.get("precedence_violations", 0) > 0: flags.append("P")
                    if rec.get("routing_order_violations", 0) > 0: flags.append("R")
                    if rec.get("timing_consistency_violations", 0) > 0: flags.append("T")
                    if rec.get("machine_capacity_violations", 0) > 0: flags.append("C")
                    if rec.get("missing_op_count", 0) > 0: flags.append("M")
                    row.append("".join(flags) if flags else "?")
        rows.append([inst, size] + row[1:])
    out.append(md_table(headers, rows))
    out.append("")
    out.append("Legend: F=Feasible. Pelanggaran: P=Precedence, R=Routing-order, T=Timing-consistency, C=Machine-capacity, M=Missing-op.\n")

    return "\n".join(out)


def build_summary_json():
    summary = {"starjob_sm": {}, "ood_ftla": {}}
    for model in MODELS:
        summary["starjob_sm"][model] = {}
        summary["ood_ftla"][model] = {}
        for method in METHODS:
            sj = load(starjob_path(model, method))
            ood = load(ood_path(model, method))
            if sj:
                summary["starjob_sm"][model][method] = {
                    "valid": sj["overall"]["valid"],
                    "feasible": sj["overall"]["feasible"],
                    "total_samples": sj["total_samples"],
                    "violation_totals": sj["overall"].get("violation_totals", {}),
                    "violation_instance_counts": sj["overall"].get("violation_instance_counts", {}),
                    "by_size_bucket": aggregate_starjob(sj),
                }
            if ood:
                by_fam, by_shape = aggregate_ood(ood)
                feas = sum(1 for r in ood["results"] if r["feasible"])
                summary["ood_ftla"][model][method] = {
                    "n": len(ood["results"]),
                    "feasible": feas,
                    "violation_totals": ood.get("overall_violation_totals", {}),
                    "violation_instance_counts": ood.get("overall_violation_instance_counts", {}),
                    "by_family": by_fam,
                    "by_shape": by_shape,
                }
    return summary


CSV_COLS_STARJOB = [
    "model", "method", "idx", "size", "jobs", "machines",
    "feasible", "true_makespan", "pred_makespan", "gap_pct", "exact_makespan",
    "ops_emitted", "ops_expected", "extra_ops",
    "precedence_violations", "routing_order_violations",
    "timing_consistency_violations", "machine_capacity_violations",
    "missing_op_count", "total_violations",
    "gen_tokens", "time_s",
]
CSV_COLS_OOD = [
    "model", "method", "name", "size",
    "feasible", "best_known", "pred", "gap_pct",
    "input_tokens", "gen_tokens", "time_s",
    "precedence_violations", "routing_order_violations",
    "timing_consistency_violations", "machine_capacity_violations",
    "missing_op_count", "total_violations", "extra_ops",
]


def _row_total_viol(rec):
    return sum(rec.get(k, 0) for k in VIOL_KEYS)


def write_csv_starjob(path):
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(starjob_path(model, method))
            if d is None:
                continue
            for r in d["results"]:
                rows.append({
                    "model": model,
                    "method": method,
                    "idx": r.get("idx"),
                    "size": r.get("size"),
                    "jobs": r.get("jobs"),
                    "machines": r.get("machines"),
                    "feasible": int(bool(r.get("feasible"))),
                    "true_makespan": r.get("true_makespan"),
                    "pred_makespan": r.get("pred_makespan"),
                    "gap_pct": r.get("gap_pct"),
                    "exact_makespan": int(bool(r.get("exact_makespan"))),
                    "ops_emitted": r.get("ops_emitted"),
                    "ops_expected": r.get("ops_expected"),
                    "extra_ops": r.get("extra_ops", 0),
                    "precedence_violations": r.get("precedence_violations", 0),
                    "routing_order_violations": r.get("routing_order_violations", 0),
                    "timing_consistency_violations": r.get("timing_consistency_violations", 0),
                    "machine_capacity_violations": r.get("machine_capacity_violations", 0),
                    "missing_op_count": r.get("missing_op_count", 0),
                    "total_violations": _row_total_viol(r),
                    "gen_tokens": r.get("gen_tokens"),
                    "time_s": r.get("time_s"),
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS_STARJOB)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_csv_ood(path):
    rows = []
    for model in MODELS:
        for method in METHODS:
            d = load(ood_path(model, method))
            if d is None:
                continue
            for r in d["results"]:
                rows.append({
                    "model": model,
                    "method": method,
                    "name": r.get("name"),
                    "size": r.get("size"),
                    "feasible": int(bool(r.get("feasible"))),
                    "best_known": r.get("best_known"),
                    "pred": r.get("pred"),
                    "gap_pct": r.get("gap_pct"),
                    "input_tokens": r.get("input_tokens"),
                    "gen_tokens": r.get("gen_tokens"),
                    "time_s": r.get("time_s"),
                    "precedence_violations": r.get("precedence_violations", 0),
                    "routing_order_violations": r.get("routing_order_violations", 0),
                    "timing_consistency_violations": r.get("timing_consistency_violations", 0),
                    "machine_capacity_violations": r.get("machine_capacity_violations", 0),
                    "missing_op_count": r.get("missing_op_count", 0),
                    "total_violations": _row_total_viol(r),
                    "extra_ops": r.get("extra_ops", 0),
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS_OOD)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    md = build_report()
    with open("VIOLATION_REPORT.md", "w") as f:
        f.write(md)
    summary = build_summary_json()
    with open("violation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    n_sj = write_csv_starjob("violations_per_instance_starjob.csv")
    n_ood = write_csv_ood("violations_per_instance_ood.csv")
    print("Written: VIOLATION_REPORT.md")
    print("Written: violation_summary.json")
    print(f"Written: violations_per_instance_starjob.csv ({n_sj} rows)")
    print(f"Written: violations_per_instance_ood.csv ({n_ood} rows)")
