"""GRPO training: continue rsLoRA SFT adapter with constraint-based reward.

Uses trl.GRPOTrainer (0.15.x). Reward function is a closure over the chosen
mode (hybrid | uniform) and runs the repo's feasibility checker on each
generated completion.
"""
import os
# Unsloth requires being imported before transformers/trl on first use.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

from grpo_jssp.config import (
    SFT_CHECKPOINT, STARJOB_SM_PATH,
    MAX_SEQ_LENGTH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    K_SAMPLES, LEARNING_RATE, NUM_TRAIN_STEPS,
    SAVE_EVERY, LOGGING_STEPS, KL_COEF, MAX_GRAD_NORM,
    GRAD_ACCUM_STEPS, WARMUP_STEPS,
    REWARD_MODE, OUTPUT_DIR, SEED,
    GOLD_EST_SLOPE, GOLD_EST_BASE, OVERLEN_FACTOR,
)
from grpo_jssp.data_utils import load_starjob_sm
from grpo_jssp.constraint_checker import check_violations
from grpo_jssp.reward import compute_reward


def _serialize_jobs(jobs_spec):
    """jobs_spec is list[list[tuple[int,int]]]; tuples don't survive HF Dataset
    cleanly, so we serialize to JSON string and parse back inside reward_fn."""
    return json.dumps(jobs_spec)


def _deserialize_jobs(s):
    return [[tuple(op) for op in job] for job in json.loads(s)]


def make_reward_fn(mode: str):
    def reward_fn(completions, jobs_spec, bks, n_ops, **kwargs):
        rewards = []
        n_parseable = 0
        n_feasible = 0
        for comp, js_json, b, n in zip(completions, jobs_spec, bks, n_ops):
            text = comp if isinstance(comp, str) else comp[0]["content"]
            js = _deserialize_jobs(js_json)
            v = check_violations(text, js)
            bks_val = None if b in (0, None) else int(b)
            r = compute_reward(v, int(n), bks_val, mode=mode)
            rewards.append(float(r))
            if (v["ops_emitted"] + v["timing_consistency_violations"]) > 0:
                n_parseable += 1
            if v["feasible"]:
                n_feasible += 1
        # Per-batch diagnostic: r_std is the absorbing-state collapse signal;
        # parseable/feasible rates show where the reward mass is going.
        k = len(rewards)
        mean_r = sum(rewards) / k if k else 0.0
        std_r = (sum((x - mean_r) ** 2 for x in rewards) / k) ** 0.5 if k else 0.0
        print(f"[reward:{mode}] n={k} parseable={n_parseable}/{k} "
              f"feasible={n_feasible}/{k} r_mean={mean_r:.3f} r_std={std_r:.3f} "
              f"r={[round(x, 2) for x in rewards]}", flush=True)
        return rewards
    reward_fn.__name__ = f"jssp_reward_{mode}"
    return reward_fn


def build_dataset(records: list) -> Dataset:
    rows = [{
        "prompt":    r["prompt"],
        "jobs_spec": _serialize_jobs(r["jobs_spec"]),
        "bks":       r["bks"] or 0,
        "n_ops":     r["n_ops"],
    } for r in records]
    return Dataset.from_list(rows)


class LengthControlledGRPOTrainer(GRPOTrainer):
    """V5: zero the advantage of samples whose completion drifts past
    OVERLEN_FACTOR x gold_est. Over-long samples then contribute no gradient
    (neither reward nor penalty) -- this removes the length-escape collapse
    trigger without a soft-penalty reward cliff (the V2 mistake).

    `inputs` is B*G rows (each prompt repeated num_generations times), aligned
    1:1 with `advantages` and `completion_mask` on a single device.
    """

    def _prepare_inputs(self, inputs):
        out = super()._prepare_inputs(inputs)
        adv = out["advantages"]
        clen = out["completion_mask"].sum(dim=1).float()
        if len(adv) != len(inputs):       # process-slice mismatch (multi-GPU)
            return out                    # -- skip masking defensively
        n_ops = torch.tensor([float(x["n_ops"]) for x in inputs],
                             device=adv.device, dtype=torch.float)
        gold_est = GOLD_EST_SLOPE * n_ops + GOLD_EST_BASE
        over = clen > (OVERLEN_FACTOR * gold_est)
        n_over = int(over.sum().item())
        if n_over:
            out["advantages"] = torch.where(over, torch.zeros_like(adv), adv)
        self._metrics["overlen_frac"].append(over.float().mean().item())
        print(f"[lenctrl] masked {n_over}/{len(over)} over-length samples "
              f"| clen max={int(clen.max().item())}", flush=True)
        return out


def train(reward_mode: str = REWARD_MODE,
          max_records: int | None = None,
          run_name: str | None = None,
          max_steps: int = NUM_TRAIN_STEPS,
          length_control: bool = False):
    print(f"[grpo] reward_mode={reward_mode}, K={K_SAMPLES}, steps={max_steps}, "
          f"length_control={length_control}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(SFT_CHECKPOINT),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    # adapter is loaded from SFT_CHECKPOINT; switch to training mode
    model.train()

    # GRPO trains on the 98% train split; the 2% test split is held out for eval.
    records = load_starjob_sm(STARJOB_SM_PATH, limit=max_records, split="train")
    print(f"[grpo] dataset (train split, 98%): {len(records)} records")
    train_ds = build_dataset(records)

    run_name = run_name or f"grpo_{reward_mode}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # GRPO: per_device_train_batch_size must be a multiple of num_generations.
    # With K=4 and 1 prompt/step, set per_device_train_batch_size=K.
    config = GRPOConfig(
        output_dir=str(run_dir),
        run_name=run_name,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        per_device_train_batch_size=K_SAMPLES,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_generations=K_SAMPLES,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        beta=KL_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        save_steps=SAVE_EVERY,
        save_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to=["wandb"],
        seed=SEED,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        use_vllm=False,
    )

    trainer_cls = LengthControlledGRPOTrainer if length_control else GRPOTrainer
    trainer = trainer_cls(
        model=model,
        reward_funcs=[make_reward_fn(reward_mode)],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    print(f"[grpo] saved final adapter -> {final_dir}")
    return final_dir


if __name__ == "__main__":
    train()
