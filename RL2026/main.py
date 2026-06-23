"""CAP-GRPO CLI — Constraint-Aware Policy with Group Relative Policy Optimization.

Applies GRPO to fine-tune an LLM on the Job-Shop Scheduling Problem (JSSP),
using a 7-component constraint-aware reward with coverage gate.

Subcommands:
    train   GRPO fine-tuning from a base or SFT adapter.
    eval    Evaluate an adapter on SM (in-dist) or OOD benchmarks (FT/LA).
    infer   Single-instance inference + feasibility check (debug).

Examples:
    # Train from local SFT adapter (recommended):
    python main.py train --model-path /path/to/sft_adapter --reward-mode hybrid

    # Train from downloaded base model:
    python main.py train --model llama --reward-mode hybrid

    # Smoke test (10 steps, 50 records):
    python main.py train --model-path /path/to/sft_adapter --max-steps 10 --max-records 50

    # Evaluate:
    python main.py eval --model-path outputs/my_run/final_adapter --dataset all

    # Debug single instance:
    python main.py infer --model-path outputs/my_run/final_adapter --instance ft06
"""
from __future__ import annotations

import argparse
import sys

from src.config import (
    DEFAULT_REWARD_MODE, GRAD_ACCUM_STEPS, KL_COEF,
    LEARNING_RATE, MAX_NEW_TOKENS, MAX_SEQ_LENGTH,
    MODEL_REGISTRY, NUM_TRAIN_STEPS, SAVE_EVERY, TEMPERATURE,
)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        help="Registered model key (auto-downloads from HF Hub if not cached). "
             + " | ".join(f"{k}: {v['label']}" for k, v in MODEL_REGISTRY.items()),
    )
    g.add_argument(
        "--model-path",
        metavar="PATH",
        help="Path to a local HF model or SFT adapter directory.",
    )


def cmd_train(args: argparse.Namespace) -> int:
    import torch
    from src.data import load_starjob_sm
    from src.model import load_model
    from src.trainer import run_training

    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(
        model_key=args.model,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        dtype=dtype,
        for_training=True,
    )

    records = load_starjob_sm(split="train", limit=args.max_records)
    print(f"[train] SM training records: {len(records)}")

    if args.model:
        label = args.model
    else:
        # use only the last two path components to keep the run name short
        from pathlib import Path as _P
        parts = _P(args.model_path).parts
        label = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    run_name = args.run_name or f"capgrpo_{label}_{args.reward_mode}"

    final_dir = run_training(
        model, tokenizer, records,
        run_name=run_name,
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
        length_control=args.length_control,
        resume_from=args.resume_from,
        kl_coef=args.kl_coef,
        grad_accum=args.grad_accum,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        lp_alpha=args.lp_alpha,
        eos_beta=args.eos_beta,
    )
    print(f"[train] DONE — adapter: {final_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from pathlib import Path
    from src.evaluator import eval_ood, eval_sm
    from src.model import load_model

    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(
        model_key=args.model,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        dtype=dtype,
        for_training=False,
    )

    out_path = Path(args.out) if args.out else None
    dataset  = args.dataset.lower()

    if dataset == "sm":
        result = eval_sm(model, tokenizer,
                         num_samples=args.n_sm_samples,
                         max_new_tokens=args.max_new_tokens,
                         output_json=out_path)
    else:
        result = eval_ood(model, tokenizer,
                          dataset=dataset,
                          max_new_tokens=args.max_new_tokens,
                          output_json=out_path)

    print("\n[eval] Summary:")
    for k, v in result["summary"].items():
        print(f"  {k:>35}: {v}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    import json
    import torch
    from src.checker import check_violations
    from src.data import load_ood_benchmarks
    from src.evaluator import generate_one
    from src.model import load_model

    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(
        model_key=args.model,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        dtype=dtype,
        for_training=False,
    )

    records  = {r["name"]: r for r in load_ood_benchmarks()}
    if args.instance not in records:
        choices = list(records)[:10]
        print(f"Instance {args.instance!r} not found. Available: {choices} ...",
              file=sys.stderr)
        return 2
    r = records[args.instance]

    response, dt = generate_one(model, tokenizer, r["prompt"], args.max_new_tokens)
    v            = check_violations(response, r["jobs_spec"])
    bks          = r.get("bks")
    gap = (v["makespan"] - bks) / bks if (v["feasible"] and v["makespan"] and bks) else None

    print("\n----- RESPONSE (first 2000 chars) -----")
    print(response[:2000])
    print("\n----- VERDICT -----")
    print(json.dumps({
        "instance":    args.instance,
        "bks":         bks,
        "feasible":    v["feasible"],
        "makespan":    v["makespan"],
        "gap_to_bks":  gap,
        "gen_time_s":  round(dt, 2),
        "violations":  {k: v[k] for k in (
            "missing_op_count", "over_op_count", "routing_order_violations",
            "machine_capacity_violations", "timing_consistency_violations",
            "precedence_violations",
        )},
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CAP-GRPO: Constraint-Aware Policy with GRPO for JSSP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    pt = sub.add_parser("train", help="GRPO fine-tuning")
    _add_model_args(pt)
    pt.add_argument("--reward-mode", choices=["hybrid", "hybrid_v7", "stratified", "uniform"],
                    default=DEFAULT_REWARD_MODE)
    pt.add_argument("--length-control", action="store_true",
                    help="Zero advantages for over-length completions (V5/V6 technique)")
    pt.add_argument("--max-steps",   type=int,   default=NUM_TRAIN_STEPS)
    pt.add_argument("--max-records", type=int,   default=None,
                    help="Limit training records (None = full 98%% SM train split)")
    pt.add_argument("--save-every",  type=int,   default=SAVE_EVERY)
    pt.add_argument("--kl-coef",     type=float, default=KL_COEF)
    pt.add_argument("--grad-accum",  type=int,   default=GRAD_ACCUM_STEPS)
    pt.add_argument("--temperature", type=float, default=TEMPERATURE)
    pt.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    pt.add_argument("--lp-alpha",    type=float, default=0.10,
                    help="Length-penalty coefficient (stratified_v2 only)")
    pt.add_argument("--eos-beta",    type=float, default=0.05,
                    help="EOS bonus coefficient (stratified_v2 only)")
    pt.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    pt.add_argument("--dtype",       choices=["bfloat16", "float16"], default="bfloat16")
    pt.add_argument("--no-4bit",     action="store_true")
    pt.add_argument("--run-name",    default=None, help="Override output directory name")
    pt.add_argument("--resume-from", default=None, metavar="CKPT_DIR",
                    help="Resume training from a checkpoint directory")
    pt.set_defaults(func=cmd_train)

    # ---- eval ----
    pe = sub.add_parser("eval", help="Evaluate adapter on SM / FT / LA / all")
    _add_model_args(pe)
    pe.add_argument("--dataset", choices=["sm", "ft", "la", "all"], default="all")
    pe.add_argument("--n-sm-samples",   type=int, default=20)
    pe.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    pe.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    pe.add_argument("--dtype",   choices=["bfloat16", "float16"], default="bfloat16")
    pe.add_argument("--no-4bit", action="store_true")
    pe.add_argument("--out",     default=None, help="Output JSON path")
    pe.set_defaults(func=cmd_eval)

    # ---- infer ----
    pi = sub.add_parser("infer", help="Single-instance inference + feasibility check")
    _add_model_args(pi)
    pi.add_argument("--instance", required=True, help="OOD instance name (ft06, la01, ...)")
    pi.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    pi.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    pi.add_argument("--dtype",   choices=["bfloat16", "float16"], default="bfloat16")
    pi.add_argument("--no-4bit", action="store_true")
    pi.set_defaults(func=cmd_infer)

    return p


def main() -> int:
    p    = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
