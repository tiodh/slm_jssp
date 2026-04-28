"""Extract train_loss and eval_loss per step from each model's trainer_state.json."""
import json
import csv
import os

models = {
    "llama_3_1_8b":   "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406/trainer_state.json",
    "ministral_8b":   "output_ministral8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406/trainer_state.json",
    "qwen2_7b":       "output_qwen2_7b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406/trainer_state.json",
    "granite_3_2_8b": "output_granite8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14406/trainer_state.json",
}

losses = {}
for name, path in models.items():
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    with open(path) as f:
        state = json.load(f)

    train = []   # (step, loss)
    evals = []   # (step, eval_loss)
    for e in state["log_history"]:
        step = e.get("step")
        if "loss" in e and "eval_loss" not in e:
            train.append((step, e["loss"]))
        if "eval_loss" in e:
            evals.append((step, e["eval_loss"]))
    losses[name] = {"train": train, "eval": evals}
    print(f"{name}: train={len(train)} eval={len(evals)} best_eval={state.get('best_metric')}")

# Save per-model CSVs
os.makedirs("loss_curves", exist_ok=True)
for name, d in losses.items():
    with open(f"loss_curves/{name}_train.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "train_loss"])
        w.writerows(d["train"])
    with open(f"loss_curves/{name}_eval.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "eval_loss"])
        w.writerows(d["eval"])

# Also save combined JSON
with open("loss_curves/all_losses.json", "w") as f:
    json.dump(losses, f, indent=2)

print("\nSaved to loss_curves/")
print("  <model>_train.csv  — step, train_loss")
print("  <model>_eval.csv   — step, eval_loss")
print("  all_losses.json    — combined")
