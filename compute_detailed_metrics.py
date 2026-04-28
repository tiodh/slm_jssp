import re
import json
import sys
import statistics

def parse_eval_log(log_path):
    """Parse eval log and extract all sample results."""
    samples = []
    with open(log_path, 'r') as f:
        for line in f:
            # Match: [1/50] 10x2 | True=1602 Pred=1687 | err=+85 (5.3%)
            # or:    [24/50] 2x2 | True=219 Pred=219 | EXACT
            m = re.search(
                r'\[(\d+)/(\d+)\]\s+(\d+)x(\d+)\s+\|\s+True=(\d+)\s+Pred=(\d+)\s+\|\s+(.*)',
                line
            )
            if m:
                idx = int(m.group(1))
                total = int(m.group(2))
                jobs = int(m.group(3))
                machines = int(m.group(4))
                true_val = int(m.group(5))
                pred_val = int(m.group(6))
                rest = m.group(7).strip()

                if 'EXACT' in rest:
                    gap_pct = 0.0
                else:
                    gm = re.search(r'\(([\d.]+)%\)', rest)
                    if gm:
                        gap_pct = float(gm.group(1))
                    else:
                        gap_pct = abs(pred_val - true_val) / true_val * 100 if true_val > 0 else 0.0

                signed_err = pred_val - true_val
                size_str = f"{jobs}x{machines}"

                # Group classification
                if jobs <= 5 and machines <= 5:
                    group = "small"
                elif jobs <= 10:
                    group = "medium"
                else:
                    group = "large"

                samples.append({
                    "idx": idx,
                    "size": size_str,
                    "jobs": jobs,
                    "machines": machines,
                    "true": true_val,
                    "pred": pred_val,
                    "gap_pct": gap_pct,
                    "signed_err": signed_err,
                    "group": group,
                    "exact": pred_val == true_val,
                    "feasible": pred_val >= true_val,
                })
    return samples


def compute_metrics(samples, label="all"):
    """Compute all metrics for a list of samples."""
    n = len(samples)
    if n == 0:
        return {"n": 0}

    gaps = [s["gap_pct"] for s in samples]
    signed_errs = [s["signed_err"] for s in samples]
    abs_errs = [abs(s["signed_err"]) for s in samples]

    exact_count = sum(1 for s in samples if s["exact"])
    pred_le_true = sum(1 for s in samples if s["pred"] <= s["true"])
    infeasible = sum(1 for s in samples if s["pred"] < s["true"])

    gap_le_5 = sum(1 for g in gaps if g <= 5.0)
    gap_le_10 = sum(1 for g in gaps if g <= 10.0)
    gap_le_20 = sum(1 for g in gaps if g <= 20.0)
    gap_gt_50 = sum(1 for g in gaps if g > 50.0)

    mean_gap = statistics.mean(gaps)
    median_gap = statistics.median(gaps)
    min_gap = min(gaps)
    max_gap = max(gaps)
    mae = statistics.mean(abs_errs)
    mean_signed = statistics.mean(signed_errs)

    return {
        "n": n,
        "exact_match": {"count": exact_count, "pct": round(exact_count / n * 100, 1)},
        "gap_le_5pct": {"count": gap_le_5, "pct": round(gap_le_5 / n * 100, 1)},
        "gap_le_10pct": {"count": gap_le_10, "pct": round(gap_le_10 / n * 100, 1)},
        "gap_le_20pct": {"count": gap_le_20, "pct": round(gap_le_20 / n * 100, 1)},
        "gap_gt_50pct": {"count": gap_gt_50, "pct": round(gap_gt_50 / n * 100, 1)},
        "mean_gap_pct": round(mean_gap, 2),
        "median_gap_pct": round(median_gap, 2),
        "min_gap_pct": round(min_gap, 2),
        "max_gap_pct": round(max_gap, 2),
        "mae": round(mae, 1),
        "mean_signed_error": round(mean_signed, 1),
        "pred_le_true": {"count": pred_le_true, "pct": round(pred_le_true / n * 100, 1)},
        "infeasible_pred_lt_true": {"count": infeasible, "pct": round(infeasible / n * 100, 1)},
    }


def compute_by_size(samples):
    """Compute per-size metrics."""
    sizes = {}
    for s in samples:
        key = s["size"]
        if key not in sizes:
            sizes[key] = []
        sizes[key].append(s)

    result = []
    for size_str in sorted(sizes.keys(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
        ss = sizes[size_str]
        gaps = [s["gap_pct"] for s in ss]
        exact_count = sum(1 for s in ss if s["exact"])
        gap_le_5 = sum(1 for g in gaps if g <= 5.0)
        gap_le_10 = sum(1 for g in gaps if g <= 10.0)
        gap_le_20 = sum(1 for g in gaps if g <= 20.0)
        n = len(ss)
        result.append({
            "size": size_str,
            "n": n,
            "mean_gap_pct": round(statistics.mean(gaps), 2),
            "min_gap_pct": round(min(gaps), 2),
            "max_gap_pct": round(max(gaps), 2),
            "exact_match": exact_count,
        })
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python compute_detailed_metrics.py <log_path> <model_name> <eval_loss> [train_loss_avg]")
        sys.exit(1)

    log_path = sys.argv[1]
    model_name = sys.argv[2]
    eval_loss = float(sys.argv[3])
    train_loss_avg = float(sys.argv[4]) if len(sys.argv) > 4 else None

    samples = parse_eval_log(log_path)
    print(f"Parsed {len(samples)} samples from {log_path}")

    # Overall
    overall = compute_metrics(samples)
    overall["feasibility_rate"] = {"valid": len(samples), "total": len(samples), "pct": 100.0}
    overall["eval_loss"] = eval_loss
    if train_loss_avg is not None:
        overall["train_loss_avg"] = train_loss_avg

    # By group
    groups = {"small": [], "medium": [], "large": []}
    for s in samples:
        groups[s["group"]].append(s)

    by_group = {}
    for gname in ["small", "medium", "large"]:
        by_group[gname] = compute_metrics(groups[gname], gname)

    # By size
    by_size = compute_by_size(samples)

    output = {
        "model_name": model_name,
        "overall": overall,
        "by_group": by_group,
        "by_size": by_size,
    }

    out_path = f"metrics_{model_name.lower().replace('-', '_').replace('.', '_')}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
