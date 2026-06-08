"""CLI entry point.

Usage:
    python -m grpo_jssp.run train [--reward-mode hybrid|uniform] [--max-steps N] [--max-records N]
    python -m grpo_jssp.run eval-sm  --adapter PATH [--num-samples N] [--out PATH]
    python -m grpo_jssp.run eval-ood --adapter PATH [--out PATH]
"""
import argparse
import json
import os
from pathlib import Path

# Force wandb offline by default; override with WANDB_MODE=online if needed.
os.environ.setdefault("WANDB_MODE", "offline")


def cmd_train(args):
    from grpo_jssp.grpo_trainer import train
    train(
        reward_mode=args.reward_mode,
        max_records=args.max_records,
        run_name=args.run_name,
        max_steps=args.max_steps,
        length_control=args.length_control,
        resume_from=args.resume_from,
        sft_checkpoint=args.sft_checkpoint,
        kl_coef=args.kl_coef,
        grad_accum=args.grad_accum,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        lp_alpha=args.lp_alpha,
        eos_beta=args.eos_beta,
        save_every=args.save_every,
    )


def cmd_eval_sm(args):
    from grpo_jssp.evaluate import eval_starjob_sm
    out = eval_starjob_sm(
        Path(args.adapter),
        num_samples=args.num_samples,
        out_path=Path(args.out) if args.out else None,
    )
    print(json.dumps(out["summary"], indent=2))


def cmd_eval_ood(args):
    from grpo_jssp.evaluate import eval_ood
    out = eval_ood(
        Path(args.adapter),
        out_path=Path(args.out) if args.out else None,
    )
    print(json.dumps(out["summary"], indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--reward-mode",
                   choices=["hybrid", "hybrid_v7", "uniform", "stratified", "stratified_v2"],
                   default="hybrid")
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--max-records", type=int, default=None)
    t.add_argument("--run-name", type=str, default=None)
    t.add_argument("--length-control", action="store_true",
                   help="V5/V6: zero advantage of over-length samples")
    t.add_argument("--resume-from", type=str, default=None,
                   help="Path to a checkpoint dir to resume training from")
    t.add_argument("--sft-checkpoint", type=str, default=None,
                   help="Override config.SFT_CHECKPOINT (e.g. LoRA vs rsLoRA)")
    t.add_argument("--kl-coef", type=float, default=None,
                   help="Override config.KL_COEF (V1=0.04, V2=0.10, V3-V6=0.05)")
    t.add_argument("--grad-accum", type=int, default=None,
                   help="Override config.GRAD_ACCUM_STEPS (V1/V2=1, V3+=4)")
    t.add_argument("--temperature", type=float, default=None,
                   help="Override config.TEMPERATURE (V1/V2=0.8, V3+=0.7)")
    t.add_argument("--learning-rate", type=float, default=None,
                   help="Override config.LEARNING_RATE")
    t.add_argument("--lp-alpha", type=float, default=0.10,
                   help="V2 length penalty coefficient")
    t.add_argument("--eos-beta", type=float, default=0.05,
                   help="V2 EOS bonus coefficient")
    t.add_argument("--save-every", type=int, default=None,
                   help="Override config.SAVE_EVERY (smaller for short pilots)")
    t.set_defaults(func=cmd_train)

    e1 = sub.add_parser("eval-sm")
    e1.add_argument("--adapter", required=True)
    e1.add_argument("--num-samples", type=int, default=200)
    e1.add_argument("--out", default=None)
    e1.set_defaults(func=cmd_eval_sm)

    e2 = sub.add_parser("eval-ood")
    e2.add_argument("--adapter", required=True)
    e2.add_argument("--out", default=None)
    e2.set_defaults(func=cmd_eval_ood)

    args = p.parse_args()
    # default max_steps inside trainer when None
    if getattr(args, "max_steps", None) is None and args.cmd == "train":
        from grpo_jssp.config import NUM_TRAIN_STEPS
        args.max_steps = NUM_TRAIN_STEPS
    args.func(args)


if __name__ == "__main__":
    main()
