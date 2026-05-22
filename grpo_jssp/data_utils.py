"""Data loaders for StarJob SM (train + in-distribution test) and OOD FT+LA.

Each record carries jobs_spec (required routing) and bks (best-known makespan)
so the trainer can call check_violations + compute_reward without re-parsing.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feasibility import extract_makespan, parse_schedule_ops_strict
from grpo_jssp.config import BEST_KNOWN, OOD_INSTANCES, SPLIT_SEED, TEST_FRAC

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """


def parse_input_jobs(input_text: str) -> list:
    """Parse `J0: M2:8 M0:5 ...` block into [[(machine, dur), ...], ...]."""
    jobs = []
    current = []
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if re.match(r"^J\d+:", line):
            if current:
                jobs.append(current)
            rest = re.sub(r"^J\d+:\s*", "", line)
            current = [(int(m), int(d)) for m, d in re.findall(r"M(\d+):(\d+)", rest)]
        else:
            current.extend(
                (int(m), int(d)) for m, d in re.findall(r"M(\d+):(\d+)", line)
            )
    if current:
        jobs.append(current)
    return jobs


def _bks_from_gold(output_text: str) -> int | None:
    """Derive BKS from gold-output schedule = max end time of all ops."""
    m = extract_makespan(output_text)
    if m is not None:
        return m
    ops, _ = parse_schedule_ops_strict(output_text)
    return max((e for *_, e in ops), default=None)


def format_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_PROMPT.format(instruction, input_text)


def _record_from_raw(r: dict) -> dict:
    jobs_spec = parse_input_jobs(r["input"])
    return {
        "instruction": r["instruction"],
        "input": r["input"],
        "output": r["output"],
        "prompt": format_prompt(r["instruction"], r["input"]),
        "jobs_spec": jobs_spec,
        "n_ops": sum(len(j) for j in jobs_spec),
        "bks": _bks_from_gold(r["output"]),
    }


def load_starjob_sm(path: Path | str, limit: int = None,
                    split: str = "train") -> list:
    """Load StarJob SM JSONL, then apply the 2% test split (seed=42) used by
    the SFT training/eval scripts. `split` ∈ {'train','test','all'}.

    Default 'train' is what GRPO should iterate over. 'test' matches the held-out
    set used in metrics_rslora_llama.json so eval numbers are apples-to-apples.
    """
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line))

    if split == "all":
        chosen = raw
    else:
        rng = random.Random(SPLIT_SEED)
        indices = list(range(len(raw)))
        rng.shuffle(indices)
        test_size = int(len(raw) * TEST_FRAC)
        test_idx = set(indices[:test_size])
        if split == "test":
            chosen_idx = indices[:test_size]
            # match eval_rslora.py: it then random.sample's from this list.
            # We preserve order from the shuffled-prefix to keep determinism.
        elif split == "train":
            chosen_idx = [i for i in range(len(raw)) if i not in test_idx]
        else:
            raise ValueError(f"split must be train|test|all, got {split!r}")
        chosen = [raw[i] for i in chosen_idx]

    if limit is not None:
        chosen = chosen[:limit]
    return [_record_from_raw(r) for r in chosen]


def _parse_orlib_block(lines: list) -> tuple:
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        ops = [(toks[2 * k], toks[2 * k + 1]) for k in range(len(toks) // 2)]
        jobs.append(ops)
    return n, m, jobs


def _to_starjob_input(jobs: list) -> str:
    parts = []
    for j, ops in enumerate(jobs):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    return "\n".join(parts)


def _instruction_for(n: int, m: int) -> str:
    return (
        f"Optimize schedule for {n} Jobs (denoted as J) across {m} Machines "
        "(denoted as M) to minimize makespan. The makespan is the completion "
        "time of the last operation in the schedule. Each M can process only "
        "one J at a time, and once started, J cannot be interrupted.\n\n"
    )


def load_ood_benchmarks(jobshop1_path: Path | str,
                        names: list = None) -> list:
    """Parse jobshop1.txt into [{name, prompt, jobs_spec, n_ops, bks}, ...]."""
    names = names or OOD_INSTANCES
    with open(jobshop1_path) as f:
        text = f.read()
    blocks = re.split(r"\n\s*instance\s+(\w+)\s*\n", text)
    parsed = {}
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body_lines = [
            l for l in blocks[i + 1].splitlines()
            if l.strip()
            and not l.lstrip().startswith("+")
            and not re.match(r"^\s*[A-Za-z]", l)
        ]
        try:
            n, m, jobs = _parse_orlib_block(body_lines)
            if len(jobs) == n and all(len(j) == m for j in jobs):
                parsed[name] = (n, m, jobs)
        except Exception:
            continue

    out = []
    for name in names:
        if name not in parsed:
            continue
        n, m, jobs = parsed[name]
        input_text = _to_starjob_input(jobs)
        out.append({
            "name": name,
            "instruction": _instruction_for(n, m),
            "input": input_text,
            "prompt": format_prompt(_instruction_for(n, m), input_text),
            "jobs_spec": jobs,
            "n_ops": sum(len(j) for j in jobs),
            "bks": BEST_KNOWN.get(name),
        })
    return out
