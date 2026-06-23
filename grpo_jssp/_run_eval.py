"""Eval one adapter on OOD + SM, write to a given output prefix."""
import unsloth  # noqa: F401

import argparse
import json
from pathlib import Path

from grpo_jssp.evaluate import eval_ood, eval_starjob_sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-prefix", required=True,
                    help="output dir prefix, files: <prefix>_sm.json, <prefix>_ood.json")
    ap.add_argument("--num-sm", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] adapter = {args.adapter}", flush=True)
    print("\n=== OOD (18 instances) ===", flush=True)
    o = eval_ood(Path(args.adapter), out_path=out.with_name(out.name + "_ood.json"))
    print(json.dumps(o["summary"], indent=2), flush=True)

    print(f"\n=== StarJob SM (test split, {args.num_sm}) ===", flush=True)
    s = eval_starjob_sm(Path(args.adapter), num_samples=args.num_sm,
                        out_path=out.with_name(out.name + "_sm.json"))
    print(json.dumps(s["summary"], indent=2), flush=True)
    print("\n[eval] DONE", flush=True)


if __name__ == "__main__":
    main()
