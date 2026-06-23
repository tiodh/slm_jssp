"""Evaluate any of the 4 fine-tuned models on classic JSSP benchmarks
(FT, LA, TAI <=20x15) with feasibility validation and 7000 max_new_tokens."""
import os
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
import sys
import re
import json
import time
import argparse
import torch
from unsloth import FastLanguageModel
from feasibility import (
    extract_makespan,
    parse_schedule_ops_strict,
    validate_feasibility,
)

sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "llama":    ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                 "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400",
                 False),
    "ministral":("mistralai/Ministral-8B-Instruct-2410",
                 "output_ministral8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400",
                 True),
    "qwen2":    ("unsloth/Qwen2-7B-Instruct-bnb-4bit",
                 "output_qwen2_7b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400",
                 False),
    "granite":  ("unsloth/granite-3.2-8b-instruct-bnb-4bit",
                 "output_granite8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400",
                 True),
}

MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 7000

BENCH_DIR = "data/benchmarks"
JOBSHOP1 = os.path.join(BENCH_DIR, "jobshop1.txt")

BEST_KNOWN = {
    "ft06":  55, "ft10":  930, "ft20": 1165,
    "la01": 666, "la02":  655, "la03":  597, "la04": 590, "la05": 593,
    "la06": 926, "la07":  890, "la08":  863, "la09": 951, "la10": 958,
    "la16": 945, "la17":  784, "la18":  848, "la19": 842, "la20": 902,
    "ta01":1231, "ta02": 1244, "ta03": 1218, "ta04":1175, "ta05":1224,
    "ta06":1238, "ta07": 1227, "ta08": 1217, "ta09":1274, "ta10":1241,
}

WANTED_FT_LA = ["ft06","ft10","ft20",
                "la01","la02","la03","la04","la05",
                "la06","la07","la08","la09","la10",
                "la16","la17","la18","la19","la20"]
WANTED_TAI = [f"ta{i:02d}" for i in range(1, 11)]

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """

def parse_orlib_instance(lines):
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        ops = [(toks[2*k], toks[2*k+1]) for k in range(len(toks)//2)]
        jobs.append(ops)
    return n, m, jobs

def load_jobshop1(path):
    instances = {}
    with open(path) as f:
        text = f.read()
    blocks = re.split(r'\n\s*instance\s+(\w+)\s*\n', text)
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i+1]
        body_lines = [l for l in body.splitlines() if l.strip()
                      and not l.lstrip().startswith('+')
                      and not re.match(r'^\s*[A-Za-z]', l)]
        try:
            n, m, jobs = parse_orlib_instance(body_lines)
            if len(jobs) == n and all(len(j) == m for j in jobs):
                instances[name] = (n, m, jobs)
        except Exception:
            pass
    return instances

def load_tai_files(names):
    instances = {}
    for name in names:
        path = os.path.join(BENCH_DIR, f"{name}.txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        n, m, jobs = parse_orlib_instance(lines)
        instances[name] = (n, m, jobs)
    return instances

def to_starjob_format(n, m, jobs):
    instruction = (f"Optimize schedule for {n} Jobs (denoted as J) across {m} "
                   "Machines (denoted as M) to minimize makespan. The makespan is "
                   "the completion time of the last operation in the schedule. "
                   "Each M can process only one J at a time, and once started, J "
                   "cannot be interrupted.\n\n")
    parts = []
    for j, ops in enumerate(jobs):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    return instruction, "\n".join(parts) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--skip-tai", action="store_true",
                    help="Skip TAI 15x15 instances (cut eval time)")
    args = ap.parse_args()

    base, ckpt, eager = MODELS[args.model]
    print(f"=== Evaluating {args.model.upper()} ===")
    print(f"Base: {base}\nCheckpoint: {ckpt}\nEager attention: {eager}")

    js1 = load_jobshop1(JOBSHOP1)
    selected = []
    for name in WANTED_FT_LA:
        if name in js1: selected.append((name, *js1[name]))
    if not args.skip_tai:
        tai = load_tai_files(WANTED_TAI)
        for name in WANTED_TAI:
            if name in tai: selected.append((name, *tai[name]))
    else:
        print("Skipping TAI 15x15 instances (--skip-tai)")
    print(f"Total instances: {len(selected)}")

    kwargs = dict(
        model_name=base,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    if eager:
        kwargs["attn_implementation"] = "eager"

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    print(f"Loading LoRA from {ckpt}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ckpt)
    FastLanguageModel.for_inference(model)

    results = []
    t_start = time.time()
    for i, (name, n, m, jobs) in enumerate(selected, 1):
        bk = BEST_KNOWN.get(name)
        instr, inp = to_starjob_format(n, m, jobs)
        prompt = alpaca_prompt.format(instr, inp)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        in_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
            )
        dt = time.time() - t0
        gen = tokenizer.decode(out_ids[0][in_len:], skip_special_tokens=True)
        pred = extract_makespan(gen)
        ops, timing_bad = parse_schedule_ops_strict(gen)
        feas, info = validate_feasibility(ops, jobs, timing_bad)
        gap = (pred - bk) / bk * 100 if (pred is not None and bk) else None

        rec = {"name": name, "size": f"{n}x{m}", "best_known": bk,
               "pred": pred, "gap_pct": gap, "feasible": feas,
               "input_tokens": in_len,
               "gen_tokens": int(out_ids.shape[1] - in_len),
               "time_s": round(dt, 1),
               "raw_output": gen,
               **info}
        results.append(rec)

        gap_s = f"gap={gap:+.1f}%" if gap is not None else "gap=UNPARSEABLE"
        if feas:
            feas_s = "status=FEASIBLE"
        else:
            feas_s = (f"status=INFEASIBLE ["
                      f"prec={info['precedence_violations']}, "
                      f"route={info['routing_order_violations']}, "
                      f"timing={info['timing_consistency_violations']}, "
                      f"mcap={info['machine_capacity_violations']}, "
                      f"miss={info['missing_op_count']}]")
        print(f"  [{i:2d}/{len(selected)}] {name} size={n}x{m} | "
              f"bk={bk} pred={pred} | {gap_s} | {feas_s} | "
              f"time={dt:.1f}s")

    total = time.time() - t_start
    print(f"\nTotal eval time: {total/60:.1f} min")

    families = {"ft": [], "la": [], "ta": []}
    for r in results:
        families[r["name"][:2]].append(r)

    viol_keys = [
        "precedence_violations",
        "routing_order_violations",
        "timing_consistency_violations",
        "machine_capacity_violations",
        "missing_op_count",
    ]

    def viol_totals(recs):
        return {k: sum(r.get(k, 0) for r in recs) for k in viol_keys}

    def viol_instance_counts(recs):
        return {k: sum(1 for r in recs if r.get(k, 0) > 0) for k in viol_keys}

    summary = {"model": args.model, "total_time_min": round(total/60, 2),
               "max_new_tokens": MAX_NEW_TOKENS, "by_family": {}, "results": results}
    summary["overall_violation_totals"] = viol_totals(results)
    summary["overall_violation_instance_counts"] = viol_instance_counts(results)
    for fam, recs in families.items():
        valid = [r for r in recs if r["pred"] is not None]
        feas_recs = [r for r in valid if r["feasible"]]
        if not recs:
            continue
        d = {
            "n": len(recs),
            "valid_parse": len(valid),
            "feasible": len(feas_recs),
            "violation_totals": viol_totals(recs),
            "violation_instance_counts": viol_instance_counts(recs),
        }
        if valid:
            all_gaps = [r["gap_pct"] for r in valid]
            d["exact_match_feasible"] = sum(1 for r in feas_recs if r["pred"] == r["best_known"])
            d["all_mean_gap_pct"] = round(sum(all_gaps)/len(all_gaps), 2)
            d["all_median_gap_pct"] = round(sorted(all_gaps)[len(all_gaps)//2], 2)
        if feas_recs:
            feas_gaps = [r["gap_pct"] for r in feas_recs]
            d["feas_mean_gap_pct"] = round(sum(feas_gaps)/len(feas_gaps), 2)
            d["feas_median_gap_pct"] = round(sorted(feas_gaps)[len(feas_gaps)//2], 2)
        summary["by_family"][fam.upper()] = d

    out_path = f"metrics_benchmarks_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"{args.model.upper()} BENCHMARK RESULTS")
    print("="*60)
    for fam, s in summary["by_family"].items():
        print(f"{fam}: jumlah_layak={s['feasible']}/{s['n']}, "
              f"prediksi_tepat_optimal={s['exact_match_feasible']}, "
              f"median_gap_semua={s['all_median_gap_pct']}%, "
              f"median_gap_yang_layak={s.get('feas_median_gap_pct','-')}%")
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
