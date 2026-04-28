"""Evaluate rsLoRA fine-tuned models on small+medium JSSP instances (jobs<=10, machines<=10).
Checks: makespan accuracy, feasibility (routing + machine constraints), per-size breakdown."""
import os
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
import sys
import json
import re
import random
import argparse
import time
import statistics
import torch
from unsloth import FastLanguageModel

sys.stdout.reconfigure(line_buffering=True)

# Model registry: base_model, checkpoint_dir, needs_eager_attn
MODELS = {
    "llama":     ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                  "output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1",
                  False),
    "granite":   ("unsloth/granite-3.2-8b-instruct-bnb-4bit",
                  "output_granite8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1",
                  True),
    "ministral": ("mistralai/Ministral-8B-Instruct-2410",
                  "output_ministral8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1",
                  True),
    "qwen2":     ("unsloth/Qwen2-7B-Instruct-bnb-4bit",
                  "output_qwen2_7b_rslora_alpha32_r32_seq8192_b1_ga8_ep1",
                  False),
}

DATA_FILE = "./data/starjob_train_sm.jsonl"
MAX_SEQ_LENGTH = 8192
SEED = 42

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """


def extract_makespan(text):
    """Extract makespan (max end time) from schedule output."""
    times = re.findall(r'->\s*(\d+)', text)
    if not times:
        return None
    return max(int(t) for t in times)


def extract_size(instruction):
    """Extract (jobs, machines) from instruction text."""
    m = re.search(r'(\d+)\s*Jobs.*?(\d+)\s*Machines', instruction)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_input_jobs(input_text):
    """Parse job definitions from input to get expected routing.
    Returns list of jobs, each job is list of (machine, duration)."""
    jobs = []
    current_ops = []
    for line in input_text.strip().split('\n'):
        line = line.strip()
        if re.match(r'^J\d+:', line):
            if current_ops:
                jobs.append(current_ops)
            # Parse ops on the same line after "Jx:"
            rest = re.sub(r'^J\d+:\s*', '', line)
            ops = re.findall(r'M(\d+):(\d+)', rest)
            current_ops = [(int(m), int(d)) for m, d in ops]
        else:
            ops = re.findall(r'M(\d+):(\d+)', line)
            current_ops.extend([(int(m), int(d)) for m, d in ops])
    if current_ops:
        jobs.append(current_ops)
    return jobs


def parse_schedule_ops(text):
    """Parse generated schedule into list of (job, machine, start, duration, end)."""
    pat = re.compile(r'J(\d+)-M(\d+):\s*(\d+)\s*\+\s*(\d+)\s*->\s*(\d+)')
    ops = []
    for m in pat.finditer(text):
        j, mc, s, d, e = map(int, m.groups())
        if s + d == e:
            ops.append((j, mc, s, d, e))
    return ops


def validate_feasibility(ops, jobs):
    """Check routing order, durations, and no machine overlap.
    Returns (feasible, info_dict)."""
    n = len(jobs)
    by_job = {j: [] for j in range(n)}
    for o in ops:
        if o[0] < n:
            by_job[o[0]].append(o)

    routing_violations = 0
    machine_violations = 0
    missing_ops = 0

    for j, ops_j in by_job.items():
        ops_j.sort(key=lambda x: x[2])
        expected = jobs[j]
        if len(ops_j) < len(expected):
            missing_ops += len(expected) - len(ops_j)
        last_end = 0
        for i, op in enumerate(ops_j[:len(expected)]):
            mc, du = expected[i]
            if op[1] != mc or op[3] != du:
                routing_violations += 1
            if op[2] < last_end:
                routing_violations += 1
            last_end = op[4]

    # Machine overlap
    by_mc = {}
    for o in ops:
        by_mc.setdefault(o[1], []).append(o)
    for mc, lst in by_mc.items():
        lst.sort(key=lambda x: x[2])
        for i in range(1, len(lst)):
            if lst[i][2] < lst[i - 1][4]:
                machine_violations += 1

    total_expected = sum(len(j) for j in jobs)
    feasible = (routing_violations == 0 and machine_violations == 0 and missing_ops == 0)
    return feasible, {
        "ops_emitted": len(ops),
        "ops_expected": total_expected,
        "missing_ops": missing_ops,
        "routing_violations": routing_violations,
        "machine_violations": machine_violations,
    }


def find_best_checkpoint(output_dir):
    """Find the best checkpoint from trainer_state.json or fallback to last."""
    state_files = []
    if os.path.isdir(output_dir):
        for ckpt in sorted(os.listdir(output_dir)):
            state_path = os.path.join(output_dir, ckpt, "trainer_state.json")
            if os.path.exists(state_path):
                state_files.append((ckpt, state_path))

    if not state_files:
        # Try direct checkpoint dirs
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("-")[1]))
            return os.path.join(output_dir, ckpts[-1])
        return output_dir

    # Read trainer_state from last checkpoint to find best
    last_ckpt, last_state_path = state_files[-1]
    with open(last_state_path) as f:
        state = json.load(f)
    best_ckpt = state.get("best_model_checkpoint")
    if best_ckpt and os.path.isdir(best_ckpt):
        return best_ckpt

    return os.path.join(output_dir, last_ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of test samples to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    args = parser.parse_args()

    base_model, output_dir, eager = MODELS[args.model]

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_best_checkpoint(output_dir)

    print(f"=== Evaluating {args.model.upper()} (rsLoRA) ===")
    print(f"Base: {base_model}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {DATA_FILE}")

    # Load test split (same split as training: 2% test, seed=42)
    all_data = []
    with open(DATA_FILE) as f:
        for line in f:
            all_data.append(json.loads(line))

    random.seed(SEED)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    test_size = int(len(all_data) * 0.02)
    test_indices = indices[:test_size]
    test_data = [all_data[i] for i in test_indices]

    samples = random.sample(test_data, min(args.num_samples, len(test_data)))
    print(f"Test set: {len(test_data)} | Evaluating: {len(samples)} samples\n")

    # Load model
    kwargs = dict(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    if eager:
        kwargs["attn_implementation"] = "eager"

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    print(f"Loading LoRA adapter from {ckpt_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ckpt_path)
    FastLanguageModel.for_inference(model)

    # Evaluate
    results = []
    t_start = time.time()

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        input_text = sample["input"]
        true_output = sample["output"]
        nj, nm = extract_size(instruction)
        size_str = f"{nj}x{nm}" if nj else "unknown"
        true_makespan = extract_makespan(true_output)
        jobs = parse_input_jobs(input_text)

        if true_makespan is None:
            print(f"  [{i+1}/{len(samples)}] {size_str} | SKIP (can't parse true makespan)")
            continue

        prompt = alpaca_prompt.format(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
            )
        dt = time.time() - t0

        n_gen_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        generated = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)
        pred_makespan = extract_makespan(generated)
        pred_ops = parse_schedule_ops(generated)

        # Feasibility check
        if jobs:
            feasible, feas_info = validate_feasibility(pred_ops, jobs)
        else:
            feasible, feas_info = False, {"ops_emitted": 0, "ops_expected": 0,
                                          "missing_ops": 0, "routing_violations": 0,
                                          "machine_violations": 0}

        rec = {
            "idx": i + 1,
            "size": size_str,
            "jobs": nj,
            "machines": nm,
            "true_makespan": true_makespan,
            "pred_makespan": pred_makespan,
            "feasible": feasible,
            "gen_tokens": n_gen_tokens,
            "time_s": round(dt, 1),
            **feas_info,
        }

        if pred_makespan is not None:
            gap_pct = (pred_makespan - true_makespan) / true_makespan * 100
            rec["gap_pct"] = round(gap_pct, 2)
            rec["exact_makespan"] = pred_makespan == true_makespan

            if feasible:
                status = "FEASIBLE+EXACT" if rec["exact_makespan"] else f"FEASIBLE gap={gap_pct:+.1f}%"
            else:
                status = (f"INFEASIBLE gap={gap_pct:+.1f}% "
                          f"[route={feas_info['routing_violations']}, "
                          f"machine={feas_info['machine_violations']}, "
                          f"missing={feas_info['missing_ops']}]")
            print(f"  [{i+1}/{len(samples)}] {size_str} | True={true_makespan} Pred={pred_makespan} | {status}")
        else:
            rec["gap_pct"] = None
            rec["exact_makespan"] = False
            print(f"  [{i+1}/{len(samples)}] {size_str} | True={true_makespan} | INVALID (no schedule parsed)")

        results.append(rec)

    total_time = time.time() - t_start

    # ========== Compute metrics ==========
    valid = [r for r in results if r["pred_makespan"] is not None]
    feasible_results = [r for r in valid if r["feasible"]]

    print(f"\n{'='*70}")
    print(f"  {args.model.upper()} rsLoRA EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total samples:      {len(results)}")
    print(f"Valid predictions:   {len(valid)}/{len(results)} ({len(valid)/len(results)*100:.1f}%)")
    print(f"Feasible solutions:  {len(feasible_results)}/{len(valid)} ({len(feasible_results)/len(valid)*100:.1f}%)" if valid else "")
    print(f"Total eval time:     {total_time/60:.1f} min")

    # Overall metrics
    if valid:
        gaps = [r["gap_pct"] for r in valid]
        exact_count = sum(1 for r in valid if r["exact_makespan"])
        feas_exact = sum(1 for r in feasible_results if r["exact_makespan"])
        print(f"\n--- All valid predictions ---")
        print(f"Exact makespan match: {exact_count}/{len(valid)} ({exact_count/len(valid)*100:.1f}%)")
        print(f"Mean gap:    {statistics.mean(gaps):.2f}%")
        print(f"Median gap:  {statistics.median(gaps):.2f}%")
        print(f"Min gap:     {min(gaps):.2f}%")
        print(f"Max gap:     {max(gaps):.2f}%")

    if feasible_results:
        feas_gaps = [r["gap_pct"] for r in feasible_results]
        print(f"\n--- Feasible solutions only ---")
        print(f"Feasible + exact makespan: {feas_exact}/{len(feasible_results)} ({feas_exact/len(feasible_results)*100:.1f}%)")
        print(f"Mean gap:    {statistics.mean(feas_gaps):.2f}%")
        print(f"Median gap:  {statistics.median(feas_gaps):.2f}%")

    # By group (small vs medium)
    print(f"\n--- By group ---")
    for group_name, cond in [("Small (<=5x5)", lambda r: r["jobs"] <= 5 and r["machines"] <= 5),
                              ("Medium (6-10)", lambda r: r["jobs"] > 5 or r["machines"] > 5)]:
        grp = [r for r in valid if cond(r)]
        if not grp:
            continue
        grp_gaps = [r["gap_pct"] for r in grp]
        grp_exact = sum(1 for r in grp if r["exact_makespan"])
        grp_feas = sum(1 for r in grp if r["feasible"])
        grp_feas_exact = sum(1 for r in grp if r["feasible"] and r["exact_makespan"])
        print(f"  {group_name}: n={len(grp)}, exact={grp_exact}, feasible={grp_feas}, "
              f"feas+exact={grp_feas_exact}, mean_gap={statistics.mean(grp_gaps):.2f}%, "
              f"median_gap={statistics.median(grp_gaps):.2f}%")

    # By exact size
    print(f"\n--- By size ---")
    print(f"{'Size':<8} {'N':>4} {'Exact':>6} {'Feas':>6} {'F+Exact':>8} {'MeanGap':>9} {'MinGap':>8} {'MaxGap':>8}")
    print("-" * 62)
    size_groups = {}
    for r in valid:
        size_groups.setdefault(r["size"], []).append(r)
    for size_str in sorted(size_groups.keys(),
                           key=lambda s: (int(s.split('x')[0]), int(s.split('x')[1]))):
        grp = size_groups[size_str]
        grp_gaps = [r["gap_pct"] for r in grp]
        exact = sum(1 for r in grp if r["exact_makespan"])
        feas = sum(1 for r in grp if r["feasible"])
        feas_exact = sum(1 for r in grp if r["feasible"] and r["exact_makespan"])
        print(f"{size_str:<8} {len(grp):>4} {exact:>6} {feas:>6} {feas_exact:>8} "
              f"{statistics.mean(grp_gaps):>8.2f}% {min(grp_gaps):>7.2f}% {max(grp_gaps):>7.2f}%")

    # Save JSON
    output = {
        "model": args.model,
        "method": "rsLoRA",
        "checkpoint": ckpt_path,
        "dataset": DATA_FILE,
        "total_samples": len(results),
        "total_time_min": round(total_time / 60, 2),
        "overall": {
            "valid": len(valid),
            "feasible": len(feasible_results),
            "exact_makespan": sum(1 for r in valid if r["exact_makespan"]) if valid else 0,
            "feasible_exact": sum(1 for r in feasible_results if r["exact_makespan"]) if feasible_results else 0,
            "mean_gap_pct": round(statistics.mean([r["gap_pct"] for r in valid]), 2) if valid else None,
            "median_gap_pct": round(statistics.median([r["gap_pct"] for r in valid]), 2) if valid else None,
        },
        "by_size": [],
        "results": results,
    }
    for size_str in sorted(size_groups.keys(),
                           key=lambda s: (int(s.split('x')[0]), int(s.split('x')[1]))):
        grp = size_groups[size_str]
        grp_gaps = [r["gap_pct"] for r in grp]
        output["by_size"].append({
            "size": size_str,
            "n": len(grp),
            "exact_makespan": sum(1 for r in grp if r["exact_makespan"]),
            "feasible": sum(1 for r in grp if r["feasible"]),
            "feasible_exact": sum(1 for r in grp if r["feasible"] and r["exact_makespan"]),
            "mean_gap_pct": round(statistics.mean(grp_gaps), 2),
            "min_gap_pct": round(min(grp_gaps), 2),
            "max_gap_pct": round(max(grp_gaps), 2),
        })

    out_path = f"metrics_rslora_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
