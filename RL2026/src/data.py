"""Data loaders for StarJob SM (training) and OOD FT+LA benchmarks (eval).

Each record includes jobs_spec (required routing) and bks (best-known makespan)
so the reward function can be applied without re-parsing the instance.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .checker import extract_makespan, parse_schedule_ops_strict
from .config import (
    ALPACA_PROMPT, BEST_KNOWN, JOBSHOP1, OOD_INSTANCES,
    SM_TRAIN_FILE, SPLIT_SEED, TEST_FRAC,
)


def _format_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_PROMPT.format(instruction, input_text)


def _parse_input_jobs(input_text: str) -> list:
    """Parse `J0: M2:8 M0:5 ...` block into [[(machine, dur), ...], ...]."""
    jobs, current = [], []
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if re.match(r"^J\d+:", line):
            if current:
                jobs.append(current)
            rest    = re.sub(r"^J\d+:\s*", "", line)
            current = [(int(m), int(d)) for m, d in re.findall(r"M(\d+):(\d+)", rest)]
        else:
            current.extend((int(m), int(d)) for m, d in re.findall(r"M(\d+):(\d+)", line))
    if current:
        jobs.append(current)
    return jobs


def _bks_from_gold(output_text: str):
    m = extract_makespan(output_text)
    if m is not None:
        return m
    ops, _ = parse_schedule_ops_strict(output_text)
    return max((e for *_, e in ops), default=None)


def _record_from_raw(r: dict) -> dict:
    jobs_spec = _parse_input_jobs(r["input"])
    return {
        "instruction": r["instruction"],
        "input":       r["input"],
        "output":      r["output"],
        "prompt":      _format_prompt(r["instruction"], r["input"]),
        "jobs_spec":   jobs_spec,
        "n_ops":       sum(len(j) for j in jobs_spec),
        "bks":         _bks_from_gold(r["output"]),
    }


def load_starjob_sm(
    path: Path | str = SM_TRAIN_FILE,
    limit: int | None = None,
    split: str = "train",
) -> list:
    """Load StarJob SM JSONL with 2% held-out test split (seed=42).

    split: 'train' (98%) | 'test' (2%) | 'all'
    """
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line))

    if split == "all":
        chosen = raw
    else:
        rng     = random.Random(SPLIT_SEED)
        indices = list(range(len(raw)))
        rng.shuffle(indices)
        test_size = int(len(raw) * TEST_FRAC)
        test_idx  = set(indices[:test_size])
        if split == "test":
            chosen = [raw[i] for i in indices[:test_size]]
        elif split == "train":
            chosen = [raw[i] for i in range(len(raw)) if i not in test_idx]
        else:
            raise ValueError(f"split must be train|test|all, got {split!r}")

    if limit is not None:
        chosen = chosen[:limit]
    return [_record_from_raw(r) for r in chosen]


def _parse_orlib_block(lines: list):
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        jobs.append([(toks[2 * k], toks[2 * k + 1]) for k in range(len(toks) // 2)])
    return n, m, jobs


def _instruction_for(n: int, m: int) -> str:
    return (
        f"Optimize schedule for {n} Jobs (denoted as J) across {m} Machines "
        "(denoted as M) to minimize makespan. The makespan is the completion "
        "time of the last operation in the schedule. Each M can process only "
        "one J at a time, and once started, J cannot be interrupted.\n\n"
    )


def _starjob_input(jobs: list) -> str:
    parts = []
    for j, ops in enumerate(jobs):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    return "\n".join(parts)


def load_ood_benchmarks(
    path: Path | str = JOBSHOP1,
    names: list | None = None,
) -> list:
    """Parse jobshop1.txt → list of eval records for OOD benchmarks."""
    names = names or OOD_INSTANCES
    with open(path) as f:
        text = f.read()
    blocks = re.split(r"\n\s*instance\s+(\w+)\s*\n", text)
    parsed = {}
    for i in range(1, len(blocks), 2):
        name       = blocks[i].strip()
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
        n, m, jobs    = parsed[name]
        input_text    = _starjob_input(jobs)
        out.append({
            "name":        name,
            "n":           n,
            "m":           m,
            "instruction": _instruction_for(n, m),
            "input":       input_text,
            "prompt":      _format_prompt(_instruction_for(n, m), input_text),
            "jobs_spec":   jobs,
            "n_ops":       sum(len(j) for j in jobs),
            "bks":         BEST_KNOWN.get(name),
        })
    return out
