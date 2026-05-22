"""Evaluate GRPO-tuned adapter on StarJob SM and OOD FT+LA benchmarks."""
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import json
import time
import statistics
from pathlib import Path

import torch
from transformers import GenerationConfig

from grpo_jssp.config import (
    SFT_CHECKPOINT, STARJOB_SM_PATH, JOBSHOP1_PATH,
    OOD_INSTANCES, BEST_KNOWN,
    MAX_SEQ_LENGTH, MAX_NEW_TOKENS, SEED,
)
from grpo_jssp.data_utils import load_starjob_sm, load_ood_benchmarks
from grpo_jssp.constraint_checker import check_violations


def _load(adapter_path: Path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """Match existing eval_rslora{,_benchmarks}.py: T=0.1, do_sample=True."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=MAX_SEQ_LENGTH - max_new_tokens).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), dt


def _aggregate(results: list) -> dict:
    feas = [r for r in results if r["feasible"]]
    summary = {
        "n": len(results),
        "n_feasible": len(feas),
        "feasibility_rate": len(feas) / max(len(results), 1),
        "missing_op_count_sum":           sum(r["missing_op_count"] for r in results),
        "routing_order_violations_sum":   sum(r["routing_order_violations"] for r in results),
        "machine_capacity_violations_sum": sum(r["machine_capacity_violations"] for r in results),
        "timing_consistency_violations_sum": sum(r["timing_consistency_violations"] for r in results),
        "precedence_violations_sum":      sum(r["precedence_violations"] for r in results),
    }
    gaps = [(r["makespan"] - r["bks"]) / r["bks"]
            for r in feas if r.get("bks") and r.get("makespan")]
    if gaps:
        summary["mean_gap_to_bks"] = statistics.mean(gaps)
        summary["median_gap_to_bks"] = statistics.median(gaps)
    return summary


def eval_starjob_sm(adapter_path: Path, num_samples: int = 200,
                    out_path: Path | None = None) -> dict:
    model, tokenizer = _load(adapter_path)
    # Use held-out 2% test split (seed=42) to keep eval truly held-out from training.
    records = load_starjob_sm(STARJOB_SM_PATH, limit=num_samples, split="test")
    results = []
    for i, r in enumerate(records):
        gen, dt = _generate(model, tokenizer, r["prompt"])
        v = check_violations(gen, r["jobs_spec"])
        v.update({
            "idx": i,
            "bks": r["bks"],
            "n_ops": r["n_ops"],
            "gen_time_s": round(dt, 2),
            "generation": gen,
        })
        results.append(v)
        if (i + 1) % 10 == 0:
            print(f"  [SM {i+1}/{len(records)}] feas_so_far="
                  f"{sum(x['feasible'] for x in results)}/{i+1}")
    summary = _aggregate(results)
    out = {"summary": summary, "per_instance": results}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def eval_ood(adapter_path: Path, out_path: Path | None = None) -> dict:
    model, tokenizer = _load(adapter_path)
    records = load_ood_benchmarks(JOBSHOP1_PATH, OOD_INSTANCES)
    results = []
    for r in records:
        gen, dt = _generate(model, tokenizer, r["prompt"])
        v = check_violations(gen, r["jobs_spec"])
        v.update({
            "name": r["name"],
            "bks": r["bks"],
            "n_ops": r["n_ops"],
            "gen_time_s": round(dt, 2),
            "generation": gen,
        })
        results.append(v)
        marker = "OK" if v["feasible"] else f"viol={v['total_violations']}"
        ms = v["makespan"] if v["makespan"] else "-"
        print(f"  [OOD {r['name']}] {marker} cmax={ms} bks={r['bks']}")
    summary = _aggregate(results)
    out = {"summary": summary, "per_instance": results}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    out_sm  = Path("grpo_jssp/runs/eval_sft_sm.json")
    out_ood = Path("grpo_jssp/runs/eval_sft_ood.json")
    print("=== eval SM (baseline = SFT) ===")
    s = eval_starjob_sm(SFT_CHECKPOINT, num_samples=20, out_path=out_sm)
    print(json.dumps(s["summary"], indent=2))
    print("=== eval OOD (baseline = SFT) ===")
    o = eval_ood(SFT_CHECKPOINT, out_path=out_ood)
    print(json.dumps(o["summary"], indent=2))
