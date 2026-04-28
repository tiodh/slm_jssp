"""Plot learning curves for all 4 models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("loss_curves/all_losses.json") as f:
    losses = json.load(f)

LABELS = {
    "llama_3_1_8b":   ("Llama-3.1-8B",    "#2E6FDB"),
    "ministral_8b":   ("Ministral-8B",    "#E67E22"),
    "qwen2_7b":       ("Qwen2-7B",        "#27AE60"),
    "granite_3_2_8b": ("Granite-3.2-8B",  "#8E44AD"),
}

def smooth(y, k=50):
    y = np.array(y)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

# --- Train loss ---
ax = axes[0]
for key, (label, color) in LABELS.items():
    d = losses[key]["train"]
    steps = [p[0] for p in d]
    vals  = [p[1] for p in d]
    k = 50
    sm = smooth(vals, k)
    sm_steps = steps[k-1:]
    ax.plot(sm_steps, sm, label=label, color=color, linewidth=1.6)
ax.set_title("Training Loss  (50-step moving avg)", fontsize=13)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.grid(alpha=0.3)
ax.legend()
ax.set_ylim(bottom=0)

# --- Eval loss ---
ax = axes[1]
for key, (label, color) in LABELS.items():
    d = losses[key]["eval"]
    steps = [p[0] for p in d]
    vals  = [p[1] for p in d]
    ax.plot(steps, vals, label=label, color=color, marker="o",
            markersize=3, linewidth=1.6)
ax.set_title("Evaluation Loss", fontsize=13)
ax.set_xlabel("Step")
ax.set_ylabel("Eval Loss")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("loss_curves/learning_curves.png", bbox_inches="tight", facecolor="white")
print("Saved: loss_curves/learning_curves.png")
