"""One-shot baseline eval driver: load SFT checkpoint once, run OOD + SM."""
import unsloth  # noqa: F401

import json
import sys
from pathlib import Path

from grpo_jssp.config import SFT_CHECKPOINT
from grpo_jssp.evaluate import eval_ood, eval_starjob_sm

out_dir = Path("grpo_jssp/eval_results")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[baseline] checkpoint = {SFT_CHECKPOINT}", flush=True)

print("\n=== OOD (18 instances) ===", flush=True)
o = eval_ood(SFT_CHECKPOINT, out_path=out_dir / "baseline_sft_ood.json")
print(json.dumps(o["summary"], indent=2), flush=True)

print("\n=== StarJob SM (held-out test split, 200 samples) ===", flush=True)
s = eval_starjob_sm(SFT_CHECKPOINT, num_samples=200,
                    out_path=out_dir / "baseline_sft_sm.json")
print(json.dumps(s["summary"], indent=2), flush=True)

print("\n[baseline] DONE", flush=True)
