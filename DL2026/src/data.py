"""Loader untuk dataset training (SM) dan evaluasi OOD (FT, LA).

Format prompt: Alpaca (instruction + input + response).

- SM (`starjob_train_sm.jsonl`): 108k contoh JSSP, dipakai untuk fine-tuning.
- FT (`ft06/ft10/ft20`) dan LA (`la01-la10`, `la16-la20`): instance benchmark
  klasik OR-Library, dibaca dari `data/benchmarks/jobshop1.txt`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset

from .config import (
    ALPACA_PROMPT,
    BEST_KNOWN_SOLUTION,
    JOBSHOP1,
    LA_FILE,
    SM_TRAIN_FILE,
    WANTED_FT,
    WANTED_LA,
)


def _format_alpaca(instruction: str, input_text: str, output: str, eos_token: str) -> str:
    return ALPACA_PROMPT.format(instruction, input_text, output) + eos_token


def load_sm_training_dataset(tokenizer) -> Dataset:
    """Load Starjob SM training set sebagai HF Dataset dengan kolom `text`."""
    if not SM_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"SM training file tidak ditemukan: {SM_TRAIN_FILE}. "
            "Pastikan dataset sudah disalin ke data/starjob_train_sm.jsonl."
        )

    eos = tokenizer.eos_token

    def formatting(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = [
            _format_alpaca(ins, inp, out, eos)
            for ins, inp, out in zip(instructions, inputs, outputs)
        ]
        return {"text": texts}

    ds = load_dataset("json", data_files=str(SM_TRAIN_FILE), split="train")
    ds = ds.map(formatting, batched=True, num_proc=1)
    return ds


def _parse_orlib_instance(lines: list[str]):
    """Parse satu instance OR-Library: baris pertama 'n m', n baris berikut tiap baris job ops."""
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        ops = [(toks[2 * k], toks[2 * k + 1]) for k in range(len(toks) // 2)]
        jobs.append(ops)
    return n, m, jobs


def load_jobshop1(path: Path = JOBSHOP1) -> dict[str, tuple[int, int, list]]:
    """Parse OR-Library `jobshop1.txt` → dict {name: (n_jobs, n_machines, jobs)}."""
    instances: dict[str, tuple[int, int, list]] = {}
    text = Path(path).read_text()
    blocks = re.split(r"\n\s*instance\s+(\w+)\s*\n", text)
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]
        body_lines = [
            l for l in body.splitlines() if l.strip()
            and not l.lstrip().startswith("+")
            and not re.match(r"^\s*[A-Za-z]", l)
        ]
        try:
            n, m, jobs = _parse_orlib_instance(body_lines)
            if len(jobs) == n and all(len(j) == m for j in jobs):
                instances[name] = (n, m, jobs)
        except Exception:
            pass
    return instances


def _to_starjob_format(n: int, m: int, jobs: list) -> tuple[str, str]:
    """Konversi (n, m, jobs) ke (instruction, input) format Starjob."""
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
    return instruction, "\n".join(parts) + "\n"


def load_eval_instances(dataset: str) -> list[dict]:
    """Load eval instances.

    Args:
        dataset: salah satu dari "ft", "la", "all".

    Returns:
        list of {name, n, m, jobs, instruction, input, bks}.
    """
    dataset = dataset.lower()
    if dataset not in {"ft", "la", "all"}:
        raise ValueError(f"dataset harus 'ft' | 'la' | 'all', dapat {dataset!r}")

    js1 = load_jobshop1()

    wanted: list[str] = []
    if dataset in {"ft", "all"}:
        wanted.extend(WANTED_FT)
    if dataset in {"la", "all"}:
        wanted.extend(WANTED_LA)

    out = []
    for name in wanted:
        if name not in js1:
            continue
        n, m, jobs = js1[name]
        instruction, inp = _to_starjob_format(n, m, jobs)
        out.append({
            "name": name,
            "n": n,
            "m": m,
            "jobs": jobs,
            "instruction": instruction,
            "input": inp,
            "bks": BEST_KNOWN_SOLUTION.get(name),
        })
    return out


def load_la_jsonl() -> list[dict]:
    """Alternative loader: LA01-LA40 dari lawrence_prompt_style.jsonl (sudah Alpaca)."""
    out = []
    with open(LA_FILE) as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj)
    return out


def load_sm_eval_sample(n: int = 20) -> list[dict]:
    """Ambil n instance pertama dari SM untuk sanity-check (bukan held-out test).

    Catatan: ini IN-DISTRIBUTION (overlap dengan training). Hanya untuk demo
    bahwa model berhasil fit pattern. Untuk evaluasi OOD pakai FT/LA.
    """
    inst_re = re.compile(r"(\d+)\s+Jobs.*?(\d+)\s+Machines", re.DOTALL)
    out: list[dict] = []
    with open(SM_TRAIN_FILE) as f:
        for line in f:
            if len(out) >= n:
                break
            obj = json.loads(line)
            instruction = obj["instruction"]
            inp = obj["input"]
            m = inst_re.search(instruction)
            if not m:
                continue
            n_jobs, n_machines = int(m.group(1)), int(m.group(2))
            jobs = []
            current = None
            for ln in inp.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if ln.startswith("J") and ":" in ln and not ln.startswith("J0:M"):
                    if current is not None:
                        jobs.append(current)
                    current = []
                else:
                    ops = []
                    for tok in ln.split():
                        if ":" in tok and tok.startswith("M"):
                            mc, du = tok[1:].split(":")
                            ops.append((int(mc), int(du)))
                    if current is not None:
                        current.extend(ops)
            if current is not None:
                jobs.append(current)
            if len(jobs) != n_jobs:
                continue
            out.append({
                "name": f"sm_{len(out):03d}",
                "n": n_jobs,
                "m": n_machines,
                "jobs": jobs,
                "instruction": instruction,
                "input": inp,
                "bks": None,
            })
    return out


def build_eval_prompt(instance: dict, eos_token: str = "") -> str:
    """Susun prompt eval (tanpa response — model lengkapi)."""
    return ALPACA_PROMPT.format(instance["instruction"], instance["input"], "")
