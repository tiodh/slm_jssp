"""Build strict-checker apples-to-apples comparison table for OOD eval.

Per-model fields (13):
  strict_feas, trim_ms, trim_gap, emit, exp, over_jobs           -- existing
  missing, routing_v, machine_v, timing_v, precedence_v, total_v -- violation breakdown
  gen_time                                                       -- latency

Models (15):
  Local LoRA (9):   SFT-LoRA, LoRA-LLaMA, rsLoRA-LLaMA, GRPO-V1, GRPO-V3, GRPO-V4,
                    GRPO-V5, GRPO-V6 ck400, GRPO-V7
  OpenAI API (6):   GPT-4o-mini, GPT-4o, GPT-5 (minimal), o3-mini (medium),
                    o3-mini (high), o3 (medium)
(GRPO-V2 omitted: training collapsed at step 700, no final_adapter.)

LoRA-LLaMA: no granular per-instance source available; only the original 6 fields
are populated from the legacy strict_recheck CSV; granular columns left blank.

Output: reports/strict_recheck_all_models_full.csv
"""
import csv
import json
from pathlib import Path

REPO = Path("/home/tio/Documents/Starjob")
REPORTS = REPO / "reports"
EVAL_DIR = REPO / "grpo_jssp/eval_results"
LEGACY_CSV = REPORTS / "strict_recheck_ood_all_models.csv"
OUT_CSV = REPORTS / "strict_recheck_all_models_full.csv"

PER_MODEL_FIELDS = [
    "strict_feas", "trim_ms", "trim_gap", "emit", "exp", "over_jobs",
    "missing", "routing_v", "machine_v", "timing_v", "precedence_v",
    "total_v", "gen_time",
]


def _gap(ms, bks, feas):
    if not feas or ms is None or ms == "":
        return ""
    try:
        return round((float(ms) - float(bks)) / float(bks) * 100, 4)
    except (TypeError, ValueError):
        return ""


def _empty():
    return {f: "" for f in PER_MODEL_FIELDS}


def load_from_violations_csv(path: Path) -> dict:
    """SFT/V5/V6 schema: idx,name,n_ops,bks,feasible,makespan,missing_op_count,
    routing_order_violations,machine_capacity_violations,timing_consistency_violations,
    precedence_violations,total_violations,ops_emitted,ops_expected,gen_time_s
    (no over_op_count column)."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["name"]] = {
                "strict_feas": row["feasible"],
                "trim_ms": row["makespan"],
                "emit": row["ops_emitted"],
                "exp": row["ops_expected"],
                "over_jobs": "",  # filled from legacy strict CSV downstream
                "missing": row["missing_op_count"],
                "routing_v": row["routing_order_violations"],
                "machine_v": row["machine_capacity_violations"],
                "timing_v": row["timing_consistency_violations"],
                "precedence_v": row["precedence_violations"],
                "total_v": row["total_violations"],
                "gen_time": row["gen_time_s"],
            }
    return out


def load_from_rslora_csv(path: Path) -> dict:
    """rsLoRA schema: name,size,n_ops_expected,bks,pred,gap_pct,feasible,
    ops_emitted,extra_ops,missing_op_count,precedence,routing,machine_cap,timing."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            tv = sum(int(row[k]) for k in ("precedence","routing","machine_cap","timing"))
            tv += int(row["missing_op_count"])
            out[row["name"]] = {
                "strict_feas": row["feasible"],
                "trim_ms": row["pred"],
                "emit": row["ops_emitted"],
                "exp": row["n_ops_expected"],
                "over_jobs": row["extra_ops"],
                "missing": row["missing_op_count"],
                "routing_v": row["routing"],
                "machine_v": row["machine_cap"],
                "timing_v": row["timing"],
                "precedence_v": row["precedence"],
                "total_v": str(tv),
                "gen_time": "",
            }
    return out


def load_from_json(path: Path) -> dict:
    """GRPO-V1/V3/V4/V7 eval json schema (full per_instance fields)."""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d["per_instance"]:
        out[r["name"]] = {
            "strict_feas": str(r["feasible"]),
            "trim_ms": r.get("makespan", ""),
            "emit": r.get("ops_emitted", ""),
            "exp": r.get("ops_expected", ""),
            "over_jobs": r.get("over_op_count", 0),
            "missing": r.get("missing_op_count", ""),
            "routing_v": r.get("routing_order_violations", ""),
            "machine_v": r.get("machine_capacity_violations", ""),
            "timing_v": r.get("timing_consistency_violations", ""),
            "precedence_v": r.get("precedence_violations", ""),
            "total_v": r.get("total_violations", ""),
            "gen_time": r.get("gen_time_s", ""),
        }
    return out


def load_from_openai_json(path: Path) -> dict:
    """OpenAI benchmark JSON schema. Per-instance under 'results' key with fields:
    name, size, best_known, pred (=makespan), gap_pct, feasible, ops_emitted,
    ops_expected, extra_ops, over_op_count, missing_op_count,
    precedence_violations, routing_order_violations, timing_consistency_violations,
    machine_capacity_violations, time_s, cost_usd, etc."""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d["results"]:
        tv = (
            int(r.get("missing_op_count", 0) or 0)
            + int(r.get("over_op_count", 0) or 0)
            + int(r.get("precedence_violations", 0) or 0)
            + int(r.get("routing_order_violations", 0) or 0)
            + int(r.get("timing_consistency_violations", 0) or 0)
            + int(r.get("machine_capacity_violations", 0) or 0)
        )
        out[r["name"]] = {
            "strict_feas": str(r["feasible"]),
            "trim_ms": r.get("pred", "") if r.get("pred") is not None else "",
            "emit": r.get("ops_emitted", ""),
            "exp": r.get("ops_expected", ""),
            "over_jobs": r.get("over_op_count", 0),
            "missing": r.get("missing_op_count", ""),
            "routing_v": r.get("routing_order_violations", ""),
            "machine_v": r.get("machine_capacity_violations", ""),
            "timing_v": r.get("timing_consistency_violations", ""),
            "precedence_v": r.get("precedence_violations", ""),
            "total_v": tv,
            "gen_time": r.get("time_s", ""),
        }
    return out


def load_legacy_strict(path: Path) -> dict:
    """The pre-existing CSV — used only for LoRA-LLaMA's 6 base fields (no
    granular per-instance source elsewhere) and to fill `over_jobs` for SFT/V5/V6
    whose per-instance CSVs lack that column."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["name"]] = row
    return out


def main():
    legacy = load_legacy_strict(LEGACY_CSV)

    sft_pi = load_from_violations_csv(REPORTS / "sft_per_instance_violations_ood.csv")
    rslora_pi = load_from_rslora_csv(REPORTS / "rslora_llama_per_instance_ood.csv")
    v5_pi = load_from_violations_csv(REPORTS / "grpo_v5_per_instance_violations_ood.csv")
    v6_pi = load_from_violations_csv(REPORTS / "grpo_v6_ck400_per_instance_violations_ood.csv")
    v1_pi = load_from_json(EVAL_DIR / "full_lora_stratified_2000_v1_ood.json")
    v3_pi = load_from_json(EVAL_DIR / "full_lora_stratified_n2000_v3_ood.json")
    v4_pi = load_from_json(EVAL_DIR / "full_lora_hybrid_n2000_v4_ood.json")
    v7_pi = load_from_json(EVAL_DIR / "full_lora_hybrid_lc_over_n2000_v7_ood.json")

    # For SFT/V5/V6/rsLoRA the post-patch ("strict") checker disagrees with the
    # pre-patch per-instance CSVs on feasibility/makespan/over_jobs (it's tighter
    # — e.g. caught over-emit, machine_cap violations). Use LEGACY strict CSV as
    # canonical for: strict_feas, trim_ms, trim_gap, emit, exp, over_jobs.
    # Use per_instance CSV ONLY for the granular violation breakdown
    # (missing/routing/machine/timing/precedence/total/gen_time). These rows
    # reflect what the model actually emitted; strict feasibility is the
    # post-patch verdict on those same outputs.
    for lbl, store in [("SFT-LoRA", sft_pi), ("rsLoRA-LLaMA", rslora_pi),
                       ("GRPO-V5", v5_pi), ("GRPO-V6 ck400", v6_pi)]:
        for name, pi in store.items():
            lr = legacy.get(name, {})
            pi["strict_feas"] = lr.get(f"{lbl}_strict_feas", "")
            pi["trim_ms"] = lr.get(f"{lbl}_trim_ms", "")
            pi["trim_gap"] = lr.get(f"{lbl}_trim_gap", "")
            pi["emit"] = lr.get(f"{lbl}_emit", "")
            pi["exp"] = lr.get(f"{lbl}_exp", "")
            pi["over_jobs"] = lr.get(f"{lbl}_over_jobs", "")

    # OpenAI API baselines (post-patch checker schema already)
    gpt4o_mini_pi = load_from_openai_json(REPO / "metrics_openai_gpt-4o-mini_benchmarks.json")
    gpt4o_pi = load_from_openai_json(REPO / "metrics_openai_gpt-4o_benchmarks.json")
    gpt5_pi = load_from_openai_json(REPO / "metrics_openai_gpt-5_minimal_benchmarks.json")
    o3mini_med_pi = load_from_openai_json(REPO / "metrics_openai_o3-mini_medium_benchmarks.json")
    o3mini_high_pi = load_from_openai_json(REPO / "metrics_openai_o3-mini_high_benchmarks.json")
    o3_med_pi = load_from_openai_json(REPO / "metrics_openai_o3_medium_benchmarks.json")

    # V1/V3/V4/V7 + OpenAI stores: trim_gap computed from BKS (OpenAI's native
    # gap_pct exists too but recompute for consistency with LoRA models).
    for store in [v1_pi, v3_pi, v4_pi, v7_pi,
                  gpt4o_mini_pi, gpt4o_pi, gpt5_pi,
                  o3mini_med_pi, o3mini_high_pi, o3_med_pi]:
        for name, pi in store.items():
            lr = legacy.get(name, {})
            pi["trim_gap"] = _gap(pi["trim_ms"], lr.get("bks", 0), pi["strict_feas"] == "True")

    # LoRA-LLaMA: only 6 base fields from legacy
    lora_pi = {}
    for name, lr in legacy.items():
        d = _empty()
        d["strict_feas"] = lr.get("LoRA-LLaMA_strict_feas", "")
        d["trim_ms"] = lr.get("LoRA-LLaMA_trim_ms", "")
        d["trim_gap"] = lr.get("LoRA-LLaMA_trim_gap", "")
        d["emit"] = lr.get("LoRA-LLaMA_emit", "")
        d["exp"] = lr.get("LoRA-LLaMA_exp", "")
        d["over_jobs"] = lr.get("LoRA-LLaMA_over_jobs", "")
        lora_pi[name] = d

    MODELS = [
        ("SFT-LoRA", sft_pi),
        ("LoRA-LLaMA", lora_pi),
        ("rsLoRA-LLaMA", rslora_pi),
        ("GRPO-V1", v1_pi),
        ("GRPO-V3", v3_pi),
        ("GRPO-V4", v4_pi),
        ("GRPO-V5", v5_pi),
        ("GRPO-V6 ck400", v6_pi),
        ("GRPO-V7", v7_pi),
        ("GPT-4o-mini", gpt4o_mini_pi),
        ("GPT-4o", gpt4o_pi),
        ("GPT-5 minimal", gpt5_pi),
        ("o3-mini medium", o3mini_med_pi),
        ("o3-mini high", o3mini_high_pi),
        ("o3 medium", o3_med_pi),
    ]

    fieldnames = ["name", "bks"]
    for label, _ in MODELS:
        for f in PER_MODEL_FIELDS:
            fieldnames.append(f"{label}_{f}")

    out_rows = []
    instance_names = list(legacy.keys())
    for name in instance_names:
        bks = legacy[name]["bks"]
        row = {"name": name, "bks": bks}
        for label, store in MODELS:
            pi = store.get(name, _empty())
            for f in PER_MODEL_FIELDS:
                row[f"{label}_{f}"] = pi.get(f, "")
        out_rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {OUT_CSV} ({len(out_rows)} instances x {len(MODELS)} models x {len(PER_MODEL_FIELDS)} fields = {len(out_rows)*len(MODELS)*len(PER_MODEL_FIELDS)} datapoints)")
    print(f"Note: GRPO-V2 omitted — collapsed at step 700, no final_adapter (see V2_COLLAPSE_SIGSEGV_NOTES.md).")
    print(f"Note: LoRA-LLaMA — only 6 base fields available; granular violation breakdown not in source data.")
    print()
    print("=== Strict OOD feasibility summary ===")
    print(f"{'Model':<22}{'Strict':<10}{'Mean gap':<12}{'Mean missing':<14}{'Mean total_v':<14}{'Mean gen_time':<14}")
    print("-" * 86)
    for label, _ in MODELS:
        feas_rows = [r for r in out_rows if r[f"{label}_strict_feas"] == "True"]
        n_feas = len(feas_rows)

        def fmean(col, only_feas=False):
            src = feas_rows if only_feas else out_rows
            vals = []
            for r in src:
                v = r[f"{label}_{col}"]
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            return sum(vals)/len(vals) if vals else float("nan")

        mg = fmean("trim_gap", only_feas=True)
        mm = fmean("missing")
        mt = fmean("total_v")
        mgt = fmean("gen_time")
        def fmt(x): return f"{x:.2f}" if x == x else "  n/a"
        print(f"  {label:<20}{n_feas:>2}/18    {fmt(mg):<12}{fmt(mm):<14}{fmt(mt):<14}{fmt(mgt):<14}")


if __name__ == "__main__":
    main()
