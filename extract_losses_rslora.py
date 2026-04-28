"""Extract train_loss and eval_loss per step for rslora runs from each model's trainer_state.json."""
import json
import os

models = {
    "llama_3_1_8b":   "output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230/trainer_state.json",
    "ministral_8b":   "output_ministral8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230/trainer_state.json",
    "qwen2_7b":       "output_qwen2_7b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230/trainer_state.json",
    "granite_3_2_8b": "output_granite8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-13230/trainer_state.json",
}

losses = {}
for name, path in models.items():
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    with open(path) as f:
        state = json.load(f)

    train = []
    evals = []
    for e in state["log_history"]:
        step = e.get("step")
        if "loss" in e and "eval_loss" not in e:
            train.append({"step": step, "loss": e["loss"]})
        if "eval_loss" in e:
            evals.append({"step": step, "eval_loss": e["eval_loss"]})
    losses[name] = {
        "train": train,
        "eval": evals,
        "best_metric": state.get("best_metric"),
    }
    print(f"{name}: train={len(train)} eval={len(evals)} best_eval={state.get('best_metric')}")

os.makedirs("loss_curves", exist_ok=True)
out_path = "loss_curves/all_losses_rslora.json"
with open(out_path, "w") as f:
    json.dump(losses, f, indent=2)

print(f"\nSaved to {out_path}")
