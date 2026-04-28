"""Plot benchmark eval results: gap distribution per size for all 4 models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "llama":     ("Llama-3.1-8B",   "#2E6FDB"),
    "ministral": ("Ministral-8B",   "#E67E22"),
    "qwen2":     ("Qwen2-7B",       "#27AE60"),
    "granite":   ("Granite-3.2-8B", "#8E44AD"),
}

def n_ops(size_str):
    n, m = map(int, size_str.split("x"))
    return n * m

data = {}
for k in MODELS:
    with open(f"metrics_benchmarks_{k}.json") as f:
        data[k] = json.load(f)

# --- Figure 1: scatter gap vs n_ops, colored by feasibility ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

ax = axes[0]
for key, (label, color) in MODELS.items():
    xs_f, ys_f, xs_i, ys_i = [], [], [], []
    for r in data[key]["results"]:
        if r["pred"] is None: continue
        nop = n_ops(r["size"])
        g = r["gap_pct"]
        if r.get("feasible"):
            xs_f.append(nop); ys_f.append(g)
        else:
            xs_i.append(nop); ys_i.append(g)
    ax.scatter(xs_f, ys_f, color=color, marker="o", s=50,
               label=f"{label} (feasible)", edgecolors="black", linewidths=0.5)
    ax.scatter(xs_i, ys_i, color=color, marker="x", s=50, alpha=0.6,
               label=f"{label} (infeasible)")
ax.set_xlabel("Problem size (n_jobs × n_machines = total ops)")
ax.set_ylabel("Gap vs best-known (%)")
ax.set_title("Gap distribution across benchmark instances")
ax.set_yscale("symlog", linthresh=10)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper left")

# --- Figure 2: feasibility rate per family as grouped bars ---
ax = axes[1]
families = ["FT", "LA", "TA"]
x = np.arange(len(families))
width = 0.2
for i, (key, (label, color)) in enumerate(MODELS.items()):
    rates = []
    for fam in families:
        by = data[key].get("by_family", {}).get(fam)
        if by is None:
            rates.append(0)
        else:
            rates.append(100.0 * by["feasible"] / by["n"])
    ax.bar(x + (i - 1.5) * width, rates, width, label=label, color=color)

ax.set_xticks(x); ax.set_xticklabels(["FT (n=3)", "LA (n=15)", "TA (n=10)"])
ax.set_ylabel("Feasibility rate (%)")
ax.set_title("Feasibility rate by benchmark family")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("benchmark_results.png", bbox_inches="tight", facecolor="white")
print("Saved: benchmark_results.png")

# --- Summary table ---
print("\n" + "="*75)
print(f"{'Model':<18} {'Family':<8} {'Feas/Total':<12} {'Median Gap (feas)':<20}")
print("="*75)
for key, (label, _) in MODELS.items():
    for fam in families:
        by = data[key].get("by_family", {}).get(fam)
        if by is None:
            print(f"{label:<18} {fam:<8} {'-':<12} {'-':<20}")
            continue
        feas_str = f"{by['feasible']}/{by['n']}"
        med = by.get("feas_median_gap_pct")
        med_str = f"{med}%" if med is not None else "N/A"
        print(f"{label:<18} {fam:<8} {feas_str:<12} {med_str:<20}")
