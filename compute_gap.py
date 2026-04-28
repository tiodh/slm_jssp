"""Compute makespan gap statistics from eval results."""
import re
from collections import defaultdict

results = []
with open("eval.log") as f:
    for line in f:
        m = re.match(r'\s+\[(\d+)/50\]\s+(\d+x\d+)\s+\|\s+True=(\d+)\s+Pred=(\d+)', line)
        if m:
            idx = int(m.group(1))
            size = m.group(2)
            true_ms = int(m.group(3))
            pred_ms = int(m.group(4))
            gap = (pred_ms - true_ms) / true_ms * 100
            results.append({"idx": idx, "size": size, "true": true_ms, "pred": pred_ms, "gap": gap})

# Overall stats
gaps = [r["gap"] for r in results]
print(f"{'='*70}")
print(f"MAKESPAN GAP ANALYSIS ({len(results)} samples)")
print(f"{'='*70}")
print(f"{'Metric':<30} {'Value':>10}")
print(f"{'-'*40}")
print(f"{'Mean Gap':<30} {sum(gaps)/len(gaps):>9.1f}%")
print(f"{'Median Gap':<30} {sorted(gaps)[len(gaps)//2]:>9.1f}%")
print(f"{'Min Gap':<30} {min(gaps):>9.1f}%")
print(f"{'Max Gap':<30} {max(gaps):>9.1f}%")
print(f"{'Exact (0% gap)':<30} {sum(1 for g in gaps if g == 0):>9d}")
print(f"{'Gap <= 5%':<30} {sum(1 for g in gaps if g <= 5):>9d}")
print(f"{'Gap <= 10%':<30} {sum(1 for g in gaps if g <= 10):>9d}")
print(f"{'Gap <= 20%':<30} {sum(1 for g in gaps if g <= 20):>9d}")
print(f"{'Gap > 50%':<30} {sum(1 for g in gaps if g > 50):>9d}")

# By size group
size_groups = defaultdict(list)
for r in results:
    j, m = map(int, r["size"].split("x"))
    if j <= 5 and m <= 5:
        group = "Small (≤5x5)"
    elif j <= 10 and m <= 10:
        group = "Medium (6-10 jobs)"
    else:
        group = "Large (>10 jobs)"
    size_groups[group].append(r)

print(f"\n{'='*70}")
print(f"GAP BY PROBLEM SIZE GROUP")
print(f"{'='*70}")
print(f"{'Group':<25} {'N':>4} {'Mean Gap':>10} {'Med Gap':>10} {'Min':>8} {'Max':>8} {'Exact':>6}")
print(f"{'-'*70}")
for group in ["Small (≤5x5)", "Medium (6-10 jobs)", "Large (>10 jobs)"]:
    if group in size_groups:
        g = [r["gap"] for r in size_groups[group]]
        exact = sum(1 for x in g if x == 0)
        sg = sorted(g)
        print(f"{group:<25} {len(g):>4} {sum(g)/len(g):>9.1f}% {sg[len(sg)//2]:>9.1f}% {min(g):>7.1f}% {max(g):>7.1f}% {exact:>5}")

# Detailed by exact size
print(f"\n{'='*70}")
print(f"GAP BY EXACT PROBLEM SIZE")
print(f"{'='*70}")
size_detail = defaultdict(list)
for r in results:
    size_detail[r["size"]].append(r["gap"])

print(f"{'Size':<10} {'N':>4} {'Mean Gap':>10} {'Min':>8} {'Max':>8} {'Exact':>6}")
print(f"{'-'*50}")
for size in sorted(size_detail.keys(), key=lambda s: (int(s.split('x')[0]), int(s.split('x')[1]))):
    g = size_detail[size]
    exact = sum(1 for x in g if x == 0)
    print(f"{size:<10} {len(g):>4} {sum(g)/len(g):>9.1f}% {min(g):>7.1f}% {max(g):>7.1f}% {exact:>5}")

# Individual results table
print(f"\n{'='*70}")
print(f"ALL SAMPLES")
print(f"{'='*70}")
print(f"{'#':<4} {'Size':<8} {'True':>8} {'Pred':>8} {'Gap':>8}")
print(f"{'-'*40}")
for r in sorted(results, key=lambda x: x["gap"]):
    print(f"{r['idx']:<4} {r['size']:<8} {r['true']:>8} {r['pred']:>8} {r['gap']:>7.1f}%")
