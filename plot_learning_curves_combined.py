"""Merge LoRA + rsLoRA learning curves into one figure and one long-form CSV."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("loss_curves/all_losses.json") as f:
    lora = json.load(f)
with open("loss_curves/all_losses_rslora.json") as f:
    rslora = json.load(f)

LABELS = {
    "llama_3_1_8b":   ("Llama-3.1-8B",    "#2E6FDB"),
    "ministral_8b":   ("Ministral-8B",    "#E67E22"),
    "qwen2_7b":       ("Qwen2-7B",        "#27AE60"),
    "granite_3_2_8b": ("Granite-3.2-8B",  "#8E44AD"),
}


def lora_train(d):
    return [(p[0], p[1]) for p in d["train"]]


def lora_eval(d):
    return [(p[0], p[1]) for p in d["eval"]]


def rs_train(d):
    return [(p["step"], p["loss"]) for p in d["train"]]


def rs_eval(d):
    return [(p["step"], p["eval_loss"]) for p in d["eval"]]


# --- long-form CSV ---
csv_path = "loss_curves/all_losses_long.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "method", "phase", "step", "loss"])
    for key in LABELS:
        for s, v in lora_train(lora[key]):
            w.writerow([key, "lora", "train", s, v])
        for s, v in lora_eval(lora[key]):
            w.writerow([key, "lora", "eval", s, v])
        for s, v in rs_train(rslora[key]):
            w.writerow([key, "rslora", "train", s, v])
        for s, v in rs_eval(rslora[key]):
            w.writerow([key, "rslora", "eval", s, v])
print(f"Saved: {csv_path}")


def smooth(y, k=50):
    y = np.array(y)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)

# --- Train loss (smoothed) ---
ax = axes[0]
k = 50
for key, (label, color) in LABELS.items():
    s, v = zip(*lora_train(lora[key]))
    sm = smooth(v, k)
    ax.plot(s[k - 1:], sm, color=color, linewidth=1.6,
            linestyle="-", label=f"{label} (LoRA)")
    s, v = zip(*rs_train(rslora[key]))
    sm = smooth(v, k)
    ax.plot(s[k - 1:], sm, color=color, linewidth=1.6,
            linestyle="--", label=f"{label} (rsLoRA)")
ax.set_title("Training Loss — LoRA (solid) vs rsLoRA (dashed) — 50-step MA",
             fontsize=12)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, ncol=2)

# --- Eval loss ---
ax = axes[1]
for key, (label, color) in LABELS.items():
    s, v = zip(*lora_eval(lora[key]))
    ax.plot(s, v, color=color, linewidth=1.6, linestyle="-",
            marker="o", markersize=3, label=f"{label} (LoRA)")
    s, v = zip(*rs_eval(rslora[key]))
    ax.plot(s, v, color=color, linewidth=1.6, linestyle="--",
            marker="s", markersize=3, label=f"{label} (rsLoRA)")
ax.set_title("Evaluation Loss — LoRA (solid/circle) vs rsLoRA (dashed/square)",
             fontsize=12)
ax.set_xlabel("Step")
ax.set_ylabel("Eval Loss")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
out = "loss_curves/learning_curves_combined.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
