"""Evaluator: generate jadwal, validasi feasibility, hitung gap ke BKS."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from .config import MAX_NEW_TOKENS_EVAL
from .data import build_eval_prompt, load_eval_instances, load_sm_eval_sample
from .jssp_checker import (
    extract_makespan,
    parse_schedule_ops_strict,
    validate_feasibility,
)


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS_EVAL,
    temperature: float = 0.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):] if full.startswith(prompt) else full


def evaluate_instance(
    model,
    tokenizer,
    instance: dict,
    max_new_tokens: int = MAX_NEW_TOKENS_EVAL,
    temperature: float = 0.0,
) -> dict:
    """Generate schedule untuk satu instance, validasi, hitung makespan + gap."""
    prompt = build_eval_prompt(instance)
    t0 = time.time()
    response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    elapsed = time.time() - t0

    ops, timing_bad = parse_schedule_ops_strict(response)
    feasible, info = validate_feasibility(ops, instance["jobs"], timing_bad)
    makespan = extract_makespan(response)

    bks = instance.get("bks")
    gap = None
    if feasible and makespan is not None and bks:
        gap = (makespan - bks) / bks

    return {
        "name": instance["name"],
        "n_jobs": instance["n"],
        "n_machines": instance["m"],
        "bks": bks,
        "feasible": feasible,
        "makespan": makespan,
        "gap_to_bks": gap,
        "violations": info,
        "elapsed_sec": round(elapsed, 2),
        "response_chars": len(response),
    }


def evaluate_dataset(
    model,
    tokenizer,
    dataset: str,
    max_new_tokens: int = MAX_NEW_TOKENS_EVAL,
    temperature: float = 0.0,
    n_sm_samples: int = 20,
    output_json: Path | None = None,
) -> dict:
    """Evaluasi pada dataset {sm, ft, la, all}.

    SM-eval bersifat dianggap "in-distribution sanity check": ambil n_sm_samples
    instance pertama dari starjob_train_sm.jsonl (bukan eval test resmi).
    """
    dataset = dataset.lower()
    instances: list[dict] = []

    if dataset == "sm":
        instances = load_sm_eval_sample(n_sm_samples)
    elif dataset in {"ft", "la", "all"}:
        instances = load_eval_instances(dataset)
    else:
        raise ValueError(f"dataset harus 'sm' | 'ft' | 'la' | 'all', dapat {dataset!r}")

    FastLanguageModel.for_inference(model)

    print(f"[eval] {len(instances)} instances pada dataset={dataset}")
    results = []
    for i, inst in enumerate(instances, 1):
        print(f"[eval] [{i}/{len(instances)}] {inst['name']} ", end="", flush=True)
        r = evaluate_instance(
            model, tokenizer, inst,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        results.append(r)
        flag = "FEAS" if r["feasible"] else "infeas"
        gap_s = f"{r['gap_to_bks']*100:.2f}%" if r["gap_to_bks"] is not None else "n/a"
        print(f"-> {flag} makespan={r['makespan']} gap={gap_s} ({r['elapsed_sec']:.1f}s)")

    n_feas = sum(1 for r in results if r["feasible"])
    feas_gaps = [r["gap_to_bks"] for r in results if r["feasible"] and r["gap_to_bks"] is not None]
    summary = {
        "dataset": dataset,
        "n_total": len(results),
        "n_feasible": n_feas,
        "feasibility_rate": n_feas / max(1, len(results)),
        "mean_gap_to_bks": (sum(feas_gaps) / len(feas_gaps)) if feas_gaps else None,
    }
    out = {"summary": summary, "results": results}

    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(out, indent=2))
        print(f"[eval] Disimpan ke {output_json}")
    return out
