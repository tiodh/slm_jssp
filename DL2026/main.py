"""Entry point CLI untuk fine-tuning dan evaluasi model JSSP.

Subcommands:

    train   Fine-tune LoRA / rsLoRA pada Starjob SM.
    eval    Evaluasi adapter pada SM / FT / LA.
    infer   Inferensi satu instance benchmark (debug).

Contoh:

    python main.py train --model llama --use-rslora --max-steps 50
    python main.py eval  --model llama --adapter outputs/llama_lora_*/final_adapter --dataset all
    python main.py infer --model llama --adapter outputs/llama_lora_*/final_adapter --instance la01
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("UNSLOTH_NUM_PROC", "1")

from src.config import (
    DEFAULT_HYPERPARAMS,
    MODEL_REGISTRY,
    OUTPUTS_DIR,
)


def _add_model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Pilih model: " + ", ".join(
            f"{k} ({v['label']})" for k, v in MODEL_REGISTRY.items()
        ),
    )


def cmd_train(args: argparse.Namespace) -> int:
    import torch
    from src.data import load_sm_training_dataset
    from src.model import attach_lora, load_base_model
    from src.trainer import run_training

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model, tokenizer = load_base_model(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        dtype=dtype,
    )
    model = attach_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
    )

    train_ds = load_sm_training_dataset(tokenizer)
    print(f"[train] Dataset SM size: {len(train_ds)} contoh")

    overrides = dict(
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    out_dir = run_training(
        model,
        tokenizer,
        train_ds,
        model_key=args.model,
        use_rslora=args.use_rslora,
        max_steps=args.max_steps,
        overrides=overrides,
    )
    print(f"[train] DONE — adapter: {out_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from src.evaluator import evaluate_dataset
    from src.model import load_for_inference

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_for_inference(
        args.model,
        adapter_path=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
    )

    out_path = None
    if args.out:
        out_path = args.out
    elif args.adapter:
        from pathlib import Path
        out_path = Path(args.adapter) / f"eval_{args.dataset}.json"

    summary = evaluate_dataset(
        model,
        tokenizer,
        dataset=args.dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_sm_samples=args.n_sm_samples,
        output_json=out_path,
    )
    print("\n[eval] Summary:")
    for k, v in summary["summary"].items():
        print(f"  {k:>22}: {v}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    import json

    import torch
    from src.data import build_eval_prompt, load_eval_instances
    from src.evaluator import generate_response
    from src.jssp_checker import (
        extract_makespan,
        parse_schedule_ops_strict,
        validate_feasibility,
    )
    from src.model import load_for_inference

    instances = {i["name"]: i for i in load_eval_instances("all")}
    if args.instance not in instances:
        print(f"[infer] Instance {args.instance!r} tidak ditemukan. "
              f"Pilihan: {list(instances)[:8]} ...", file=sys.stderr)
        return 2
    inst = instances[args.instance]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_for_inference(
        args.model,
        adapter_path=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
    )

    prompt = build_eval_prompt(inst)
    response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    ops, timing_bad = parse_schedule_ops_strict(response)
    feasible, info = validate_feasibility(ops, inst["jobs"], timing_bad)
    makespan = extract_makespan(response)
    bks = inst.get("bks")
    gap = (makespan - bks) / bks if (feasible and makespan and bks) else None

    print("\n----- RESPONSE -----")
    print(response[:2000])
    print("\n----- VERDICT -----")
    print(json.dumps({
        "name": inst["name"], "bks": bks, "feasible": feasible,
        "makespan": makespan, "gap_to_bks": gap, "violations": info,
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DL2026 — JSSP fine-tuning CLI (LoRA / rsLoRA via Unsloth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    pt = sub.add_parser("train", help="Fine-tune LoRA pada Starjob SM")
    _add_model_arg(pt)
    pt.add_argument("--use-rslora", action="store_true", help="Pakai rsLoRA (α/√r)")
    pt.add_argument("--max-steps", type=int, default=-1, help="Override total step (override epochs)")
    pt.add_argument("--epochs", type=int, default=DEFAULT_HYPERPARAMS["num_train_epochs"])
    pt.add_argument("--per-device-batch-size", type=int,
                    default=DEFAULT_HYPERPARAMS["per_device_train_batch_size"])
    pt.add_argument("--grad-accum", type=int,
                    default=DEFAULT_HYPERPARAMS["gradient_accumulation_steps"])
    pt.add_argument("--learning-rate", type=float,
                    default=DEFAULT_HYPERPARAMS["learning_rate"])
    pt.add_argument("--lora-r", type=int, default=DEFAULT_HYPERPARAMS["lora_r"])
    pt.add_argument("--lora-alpha", type=int, default=DEFAULT_HYPERPARAMS["lora_alpha"])
    pt.add_argument("--lora-dropout", type=float, default=DEFAULT_HYPERPARAMS["lora_dropout"])
    pt.add_argument("--max-seq-length", type=int, default=DEFAULT_HYPERPARAMS["max_seq_length"])
    pt.add_argument("--save-steps", type=int, default=DEFAULT_HYPERPARAMS["save_steps"])
    pt.add_argument("--logging-steps", type=int, default=DEFAULT_HYPERPARAMS["logging_steps"])
    pt.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    pt.add_argument("--no-4bit", action="store_true", help="Disable bitsandbytes 4-bit")
    pt.set_defaults(func=cmd_train)

    # ---- eval ----
    pe = sub.add_parser("eval", help="Evaluasi adapter pada SM/FT/LA")
    _add_model_arg(pe)
    pe.add_argument("--adapter", required=True, help="Path ke LoRA adapter / final_adapter")
    pe.add_argument("--dataset", choices=["sm", "ft", "la", "all"], default="all")
    pe.add_argument("--n-sm-samples", type=int, default=20,
                    help="Jumlah sample untuk SM (sanity check)")
    pe.add_argument("--max-new-tokens", type=int, default=7000)
    pe.add_argument("--temperature", type=float, default=0.0)
    pe.add_argument("--max-seq-length", type=int, default=8192)
    pe.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    pe.add_argument("--out", default=None, help="Path JSON output (default: <adapter>/eval_<dataset>.json)")
    pe.set_defaults(func=cmd_eval)

    # ---- infer ----
    pi = sub.add_parser("infer", help="Inferensi satu instance benchmark")
    _add_model_arg(pi)
    pi.add_argument("--adapter", required=True)
    pi.add_argument("--instance", required=True, help="Nama instance (ft06, la01, ...)")
    pi.add_argument("--max-new-tokens", type=int, default=7000)
    pi.add_argument("--temperature", type=float, default=0.0)
    pi.add_argument("--max-seq-length", type=int, default=8192)
    pi.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    pi.set_defaults(func=cmd_infer)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
