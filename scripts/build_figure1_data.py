"""Consolidate per-checkpoint eval_loss + feasibility into figure1_data.csv.

Eval loss: every 200 steps from loss_curves/<model>_eval.csv (one row per step).
Feasibility: ONLY at the saved checkpoint listed in metrics_rslora_<model>.json
             (in-distribution SM eval) — sparse, 1 point per model.
"""
import csv
import json
from pathlib import Path

REPO = Path("/home/tio/Documents/Starjob")

MODELS = {
    "llama":   {
        "loss_csv": "loss_curves/llama_3_1_8b_eval.csv",
        "sm":  "metrics_rslora_llama.json",
        "ood": "metrics_rslora_benchmarks_llama.json",
    },
    "mistral": {  # repo calls it 'ministral' but user's label is 'mistral'
        "loss_csv": "loss_curves/ministral_8b_eval.csv",
        "sm":  "metrics_rslora_ministral.json",
        "ood": "metrics_rslora_benchmarks_ministral.json",
    },
    "granite": {
        "loss_csv": "loss_curves/granite_3_2_8b_eval.csv",
        "sm":  "metrics_rslora_granite.json",
        "ood": "metrics_rslora_benchmarks_granite.json",
    },
    "qwen2":   {
        "loss_csv": "loss_curves/qwen2_7b_eval.csv",
        "sm":  "metrics_rslora_qwen2.json",
        "ood": "metrics_rslora_benchmarks_qwen2.json",
    },
}


def parse_ckpt_step(path_str: str) -> int | None:
    if not path_str:
        return None
    for part in Path(path_str).parts:
        if part.startswith("checkpoint-"):
            try:
                return int(part.split("-")[1])
            except ValueError:
                return None
    return None


def feas_from_sm(metrics_path: Path):
    """Return (step, feas_rate, feas, total) for SM format."""
    with open(metrics_path) as f:
        d = json.load(f)
    step = parse_ckpt_step(d.get("checkpoint", ""))
    o = d.get("overall", {})
    total = d.get("total_samples", o.get("valid"))
    feas = o.get("feasible")
    if total and feas is not None:
        return step, feas / total, feas, total
    return step, None, None, None


def feas_from_ood(metrics_path: Path):
    """Return (step, feas_rate, feas, total) for OOD by_family format."""
    with open(metrics_path) as f:
        d = json.load(f)
    step = parse_ckpt_step(d.get("checkpoint", ""))
    fams = d.get("by_family", {})
    total = sum(v["n"] for v in fams.values())
    feas = sum(v["feasible"] for v in fams.values())
    if total > 0:
        return step, feas / total, feas, total
    return step, None, None, None


def main():
    rows = []
    gaps = []  # collect data-availability notes

    for model, paths in MODELS.items():
        loss_path = REPO / paths["loss_csv"]
        sm_path = REPO / paths["sm"]
        ood_path = REPO / paths["ood"]

        if not loss_path.exists():
            gaps.append(f"{model}: loss CSV missing: {loss_path}")
            continue
        if not sm_path.exists():
            gaps.append(f"{model}: SM metrics missing: {sm_path}")
            sm_step = sm_rate = sm_feas = sm_total = None
        else:
            sm_step, sm_rate, sm_feas, sm_total = feas_from_sm(sm_path)
        if not ood_path.exists():
            gaps.append(f"{model}: OOD metrics missing: {ood_path}")
            ood_step = ood_rate = ood_feas = ood_total = None
        else:
            ood_step, ood_rate, ood_feas, ood_total = feas_from_ood(ood_path)

        # Read eval loss CSV
        with open(loss_path) as f:
            steps = list(csv.DictReader(f))

        # Mark which step(s) the SM feasibility came from
        eval_steps = {int(r["step"]) for r in steps}
        sm_step_in_eval = sm_step in eval_steps if sm_step else False
        ood_step_in_eval = ood_step in eval_steps if ood_step else False

        gaps.append(
            f"{model}: {len(steps)} eval-loss points (steps {steps[0]['step']}..{steps[-1]['step']}); "
            f"SM feas at step={sm_step} ({'in eval grid' if sm_step_in_eval else 'NOT in eval grid'}); "
            f"OOD feas at step={ood_step} ({'in eval grid' if ood_step_in_eval else 'NOT in eval grid'})"
        )

        for r in steps:
            step = int(r["step"])
            eval_loss = float(r["eval_loss"])
            row = {
                "model": model,
                "method": "rsLoRA",
                "step": step,
                "eval_loss": round(eval_loss, 6),
                "feasibility_rate": "",
                "feasible_count": "",
                "total_count": "",
            }
            # Attach SM feasibility if this is the saved-ckpt step
            if step == sm_step and sm_rate is not None:
                row["feasibility_rate"] = round(sm_rate, 4)
                row["feasible_count"] = sm_feas
                row["total_count"] = sm_total
            rows.append(row)

    # Write CSV
    out = REPO / "figure1_data.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "method", "step", "eval_loss",
                        "feasibility_rate", "feasible_count", "total_count"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out} ({len(rows)} rows)")
    print()
    print("=== Data availability ===")
    for g in gaps:
        print(f"  {g}")


if __name__ == "__main__":
    main()
