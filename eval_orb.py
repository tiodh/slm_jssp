"""Evaluate SFT + GRPO V1–V6 on ORB01–ORB10 benchmarks using checker 553237d."""
import os
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import sys
import json
import re
import time
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feasibility import parse_schedule_ops_strict, validate_feasibility, extract_makespan

BASE = Path(__file__).resolve().parent
JOBSHOP1 = BASE / "data" / "benchmarks" / "jobshop1.txt"
OUT_DIR  = BASE / "grpo_jssp" / "eval_results"

ORB_INSTANCES = [f"orb{i:02d}" for i in range(1, 11)]

BEST_KNOWN = {
    "orb01": 1059, "orb02": 888,  "orb03": 1005, "orb04": 1005, "orb05": 887,
    "orb06": 1010, "orb07": 397,  "orb08": 899,  "orb09": 934,  "orb10": 944,
}

MODELS = {
    "SFT": BASE / "output_alpha32_r32_seq8192_b1_ga8_ep1" / "checkpoint-9800",
    "V1":  BASE / "grpo_jssp/runs/full_lora_stratified_2000_v1/final_adapter",
    "V2":  BASE / "grpo_jssp/runs/full_lora_stratified_v2_2000_v2/checkpoint-500",
    "V3":  BASE / "grpo_jssp/runs/full_lora_stratified_n2000_v3/final_adapter",
    "V4":  BASE / "grpo_jssp/runs/full_lora_hybrid_n2000_v4/final_adapter",
    "V5":  BASE / "grpo_jssp/runs/full_lora_hybrid_lc_n2000_v5/final_adapter",
    "V6":  BASE / "grpo_jssp/runs/full_stratified_lc_n2000_v6/checkpoint-400",
}

MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 4096
SEED = 42

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """


def load_orb_instances(path):
    text = path.read_text()
    blocks = re.split(r'\n\s*instance\s+(\w+)\s*\n', text)
    parsed = {}
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        if name not in ORB_INSTANCES:
            continue
        body_lines = [
            l for l in blocks[i + 1].splitlines()
            if l.strip()
            and not l.lstrip().startswith('+')
            and not re.match(r'^\s*[A-Za-z]', l)
        ]
        try:
            # Some instances (orb05, orb10) have text on first line, find n m
            nm_line = None
            for bl in body_lines:
                toks = bl.split()
                if len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():
                    nm_line = bl
                    break
            if nm_line is None:
                continue
            n, m = map(int, nm_line.split()[:2])
            nm_idx = body_lines.index(nm_line)
            jobs = []
            for j in range(nm_idx + 1, nm_idx + 1 + n):
                toks = list(map(int, body_lines[j].split()))
                ops = [(toks[2*k], toks[2*k+1]) for k in range(len(toks)//2)]
                jobs.append(ops)
            if len(jobs) == n and all(len(j) == m for j in jobs):
                parsed[name] = (n, m, jobs)
        except Exception:
            continue
    return parsed


def to_prompt(n, m, jobs):
    instruction = (
        f"Optimize schedule for {n} Jobs (denoted as J) across {m} "
        "Machines (denoted as M) to minimize makespan. The makespan is "
        "the completion time of the last operation in the schedule. "
        "Each M can process only one J at a time, and once started, J "
        "cannot be interrupted.\n\n"
    )
    parts = []
    for j, ops in enumerate(jobs):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    input_text = "\n".join(parts) + "\n"
    return ALPACA_PROMPT.format(instruction, input_text)


def run_model(label, adapter_path, instances):
    from unsloth import FastLanguageModel
    print(f"\n{'='*60}")
    print(f"Loading {label}: {adapter_path}")

    if not adapter_path.exists():
        print(f"  ERROR: path not found, skipping")
        return None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for name in ORB_INSTANCES:
        if name not in instances:
            print(f"  SKIP {name}: not in jobshop1.txt")
            continue
        n, m, jobs = instances[name]
        bks = BEST_KNOWN.get(name)
        prompt = to_prompt(n, m, jobs)

        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        dt = round(time.time() - t0, 2)
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        generation = tokenizer.decode(gen_ids, skip_special_tokens=True)

        ops, timing_bad = parse_schedule_ops_strict(generation)
        feasible, info = validate_feasibility(ops, jobs, timing_bad)
        makespan = extract_makespan(generation) if feasible or ops else None
        gap_pct = round(100.0 * (makespan - bks) / bks, 4) if (makespan and bks) else None

        miss = info['missing_op_count']
        over = info['over_op_count']
        rout = info['routing_order_violations']
        mcap = info['machine_capacity_violations']
        tcon = info['timing_consistency_violations']
        prec = info['precedence_violations']

        rec = {
            "name": name,
            "size": f"{n}x{m}",
            "bks": bks,
            "makespan": makespan,
            "gap_pct": gap_pct,
            "feasible": feasible,
            "missing_op_count": miss,
            "over_op_count": over,
            "routing_order_violations": rout,
            "machine_capacity_violations": mcap,
            "timing_consistency_violations": tcon,
            "precedence_violations": prec,
            "total_violations": miss+over+rout+mcap+tcon+prec,
            "ops_emitted": info['ops_emitted'],
            "ops_expected": info['ops_expected'],
            "gen_time_s": dt,
            "generation": generation,
        }
        results.append(rec)

        status = "OK" if feasible else f"viol={rec['total_violations']}"
        ms_s = str(makespan) if makespan else "-"
        gap_s = f"{gap_pct:+.1f}%" if gap_pct is not None else "-"
        print(f"  {name} [{n}x{m}] {status} makespan={ms_s} gap={gap_s} t={dt}s")

    # cleanup GPU
    del model
    torch.cuda.empty_cache()

    feas = sum(1 for r in results if r['feasible'])
    print(f"  => {feas}/{len(results)} feasible")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    help="Which models to run (default: all SFT V1-V6)")
    args = ap.parse_args()

    instances = load_orb_instances(JOBSHOP1)
    print(f"Loaded {len(instances)} ORB instances: {list(instances.keys())}")

    all_results = {}
    for label in args.models:
        if label not in MODELS:
            print(f"Unknown model: {label}")
            continue
        results = run_model(label, MODELS[label], instances)
        if results is None:
            continue
        all_results[label] = results

        # Save per-model JSON
        out_path = OUT_DIR / f"orb_{label.lower()}_ood.json"
        with open(out_path, "w") as f:
            json.dump({"model": label, "per_instance": results}, f, indent=2)
        print(f"  Saved: {out_path}")

    # Save combined CSV
    import csv
    csv_path = BASE / "reports" / "orb_eval_all_models.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","case","size","bks","pred","gap_pct",
                    "miss","over","rout","mcap","time","prec","total","feasible"])
        for label, results in all_results.items():
            for r in results:
                w.writerow([
                    label, r["name"], r["size"], r["bks"],
                    r["makespan"] or "", r["gap_pct"] or "",
                    r["missing_op_count"], r.get("over_op_count",0),
                    r["routing_order_violations"], r["machine_capacity_violations"],
                    r["timing_consistency_violations"], r["precedence_violations"],
                    r["total_violations"], r["feasible"],
                ])
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
