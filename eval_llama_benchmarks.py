"""Evaluate fine-tuned Llama-3.1-8B on classic JSSP benchmarks (FT, LA, TAI)
constrained to sizes within the fine-tuning distribution (<=20x15)."""
import os
import sys
import re
import json
import time
import torch
from unsloth import FastLanguageModel

sys.stdout.reconfigure(line_buffering=True)

MODEL_DIR = "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400"
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 4096

BENCH_DIR = "data/benchmarks"
JOBSHOP1 = os.path.join(BENCH_DIR, "jobshop1.txt")

# Best-known / optimal makespans (literature standard values)
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
    """Parse one instance: first line 'n m', then n lines of (machine duration) pairs."""
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        ops = [(toks[2*k], toks[2*k+1]) for k in range(len(toks)//2)]
        jobs.append(ops)
    return n, m, jobs

def load_jobshop1(path):
    """Parse OR-Library jobshop1.txt into {name: (n, m, jobs)}."""
    instances = {}
    with open(path) as f:
        text = f.read()
    blocks = re.split(r'\n\s*instance\s+(\w+)\s*\n', text)
    # blocks[0]=preamble, then alternating name, body, name, body...
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i+1]
        body_lines = [l for l in body.splitlines() if l.strip()
                      and not l.lstrip().startswith('+')
                      and not re.match(r'^\s*[A-Za-z]', l)]
        # First numeric line should be "n m"
        try:
            n, m, jobs = parse_orlib_instance(body_lines)
            if len(jobs) == n and all(len(j) == m for j in jobs):
                instances[name] = (n, m, jobs)
        except Exception as e:
            pass
    return instances

def load_tai_files(names):
    """Load Taillard files downloaded from JSPLIB into {name: (n, m, jobs)}."""
    instances = {}
    for name in names:
        path = os.path.join(BENCH_DIR, f"{name}.txt")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        n, m, jobs = parse_orlib_instance(lines)
        instances[name] = (n, m, jobs)
    return instances

def to_starjob_format(n, m, jobs):
    """Convert (n, m, jobs) to Starjob instruction + input strings."""
    instruction = (f"Optimize schedule for {n} Jobs (denoted as J) across {m} "
                   "Machines (denoted as M) to minimize makespan. The makespan is "
                   "the completion time of the last operation in the schedule. "
                   "Each M can process only one J at a time, and once started, J "
                   "cannot be interrupted.\n\n")
    parts = []
    for j, ops in enumerate(jobs):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    input_text = "\n".join(parts) + "\n"
    return instruction, input_text

def extract_makespan(text):
    times = re.findall(r'->\s*(\d+)', text)
    if not times:
        return None
    return max(int(t) for t in times)

def main():
    print("Loading benchmark instances...")
    js1 = load_jobshop1(JOBSHOP1)
    print(f"  Parsed {len(js1)} instances from jobshop1.txt")
    tai = load_tai_files(WANTED_TAI)
    print(f"  Loaded {len(tai)} TAI instances")

    selected = []
    for name in WANTED_FT_LA:
        if name in js1:
            selected.append((name, *js1[name]))
        else:
            print(f"  WARNING: {name} not found")
    for name in WANTED_TAI:
        if name in tai:
            selected.append((name, *tai[name]))
        else:
            print(f"  WARNING: {name} not found")

    print(f"\nTotal instances to evaluate: {len(selected)}")
    for name, n, m, _ in selected:
        bk = BEST_KNOWN.get(name, "?")
        print(f"  {name}: {n}x{m}  best-known={bk}")

    print(f"\nLoading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    print(f"Loading LoRA adapter from {MODEL_DIR}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, MODEL_DIR)
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
        gap = (pred - bk) / bk * 100 if (pred is not None and bk) else None

        rec = {"name": name, "size": f"{n}x{m}", "best_known": bk,
               "pred": pred, "gap_pct": gap,
               "input_tokens": in_len, "gen_tokens": out_ids.shape[1] - in_len,
               "time_s": round(dt, 1)}
        results.append(rec)

        gap_s = f"{gap:+.1f}%" if gap is not None else "INVALID"
        print(f"  [{i}/{len(selected)}] {name} ({n}x{m}) | BK={bk} Pred={pred} | gap={gap_s} | {dt:.1f}s")

    total = time.time() - t_start
    print(f"\nTotal eval time: {total/60:.1f} min")

    # Aggregate by benchmark family
    families = {"ft": [], "la": [], "ta": []}
    for r in results:
        fam = r["name"][:2]
        if fam in families:
            families[fam].append(r)

    summary = {"total_time_min": round(total/60, 2), "by_family": {}, "results": results}
    for fam, recs in families.items():
        valid = [r for r in recs if r["pred"] is not None]
        if not valid:
            continue
        gaps = [r["gap_pct"] for r in valid]
        exact = sum(1 for r in valid if r["pred"] == r["best_known"])
        summary["by_family"][fam.upper()] = {
            "n": len(recs),
            "valid": len(valid),
            "exact_match": exact,
            "mean_gap_pct": round(sum(gaps)/len(gaps), 2),
            "median_gap_pct": round(sorted(gaps)[len(gaps)//2], 2),
            "min_gap_pct": round(min(gaps), 2),
            "max_gap_pct": round(max(gaps), 2),
        }

    with open("metrics_llama_benchmarks.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("LLAMA-3.1-8B BENCHMARK RESULTS (vs best-known)")
    print("="*60)
    for fam, s in summary["by_family"].items():
        print(f"{fam}: valid {s['valid']}/{s['n']}, exact {s['exact_match']}, "
              f"mean gap {s['mean_gap_pct']}%, median {s['median_gap_pct']}%")
    print("\nSaved: metrics_llama_benchmarks.json")

if __name__ == "__main__":
    main()
