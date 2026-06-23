"""CAP-GRPO trainer: GRPOTrainer + optional length-control advantage masking.

Reward modes
------------
hybrid (default V4): 7-component constraint-aware reward, range [-1..7]
hybrid_v7           : hybrid + over-emit penalty, range [-1..8]
stratified          : V1 per-category weighted penalty
uniform             : {0 unparseable, 1 infeasible, 7 feasible}

Length control (--length-control)
----------------------------------
Samples whose completion length exceeds OVERLEN_FACTOR * gold_est get their
GRPO advantage zeroed — they contribute no gradient. This removes the
length-escape collapse trigger without a penalty reward cliff.
"""
from __future__ import annotations

import json
import os

import unsloth  # noqa: F401 — must be imported before trl/transformers

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from .checker import check_violations
from .config import (
    GRAD_ACCUM_STEPS, GOLD_EST_BASE, GOLD_EST_SLOPE, K_SAMPLES,
    KL_COEF, LEARNING_RATE, LOGGING_STEPS, MAX_GRAD_NORM,
    MAX_NEW_TOKENS, MAX_SEQ_LENGTH, NUM_TRAIN_STEPS,
    OUTPUTS_DIR, OVERLEN_FACTOR, SAVE_EVERY, SEED,
    TEMPERATURE, WARMUP_STEPS, DEFAULT_REWARD_MODE,
)
from .reward import compute_reward

_TOKENIZER_REF: dict = {"tok": None}


def _serialize_jobs(jobs_spec: list) -> str:
    return json.dumps(jobs_spec)


def _deserialize_jobs(s: str) -> list:
    return [[tuple(op) for op in job] for job in json.loads(s)]


def _gen_len_and_eos(text: str):
    tok = _TOKENIZER_REF.get("tok")
    if tok is None:
        return max(1, len(text) // 4), False
    ids = tok.encode(text, add_special_tokens=False)
    eos_id = tok.eos_token_id
    ended  = (eos_id is not None) and (len(ids) > 0) and (ids[-1] == eos_id)
    return len(ids), ended


def make_reward_fn(mode: str, lp_alpha: float = 0.10, eos_beta: float = 0.05):
    """Return a GRPO-compatible reward function (list of scalars)."""
    needs_len = (mode == "stratified_v2")

    def reward_fn(completions, jobs_spec, bks, n_ops, **kwargs):
        rewards, n_parseable, n_feasible = [], 0, 0
        for comp, js_json, b, n in zip(completions, jobs_spec, bks, n_ops):
            text = comp if isinstance(comp, str) else comp[0]["content"]
            js   = _deserialize_jobs(js_json)
            v    = check_violations(text, js)
            bks_val = None if b in (0, None) else int(b)
            if needs_len:
                gen_len, ended = _gen_len_and_eos(text)
                r = compute_reward(v, int(n), bks_val, mode=mode,
                                   gen_len=gen_len, ended_with_eos=ended,
                                   lp_alpha=lp_alpha, eos_beta=eos_beta)
            else:
                r = compute_reward(v, int(n), bks_val, mode=mode)
            rewards.append(float(r))
            if (v["ops_emitted"] + v["timing_consistency_violations"]) > 0:
                n_parseable += 1
            if v["feasible"]:
                n_feasible += 1
        k      = len(rewards)
        mean_r = sum(rewards) / k if k else 0.0
        std_r  = (sum((x - mean_r) ** 2 for x in rewards) / k) ** 0.5 if k else 0.0
        print(
            f"[reward:{mode}] n={k} parseable={n_parseable}/{k} "
            f"feasible={n_feasible}/{k} r_mean={mean_r:.3f} r_std={std_r:.3f} "
            f"r={[round(x, 2) for x in rewards]}",
            flush=True,
        )
        return rewards

    reward_fn.__name__ = f"cap_grpo_reward_{mode}"
    return reward_fn


class LengthControlledGRPOTrainer(GRPOTrainer):
    """Zero advantages for completions beyond OVERLEN_FACTOR * gold_est tokens.

    Removes the length-escape collapse trigger: over-long samples produce no
    gradient (neither reward nor penalty) instead of a soft penalty cliff.
    """

    def _prepare_inputs(self, inputs):
        out  = super()._prepare_inputs(inputs)
        adv  = out["advantages"]
        clen = out["completion_mask"].sum(dim=1).float()
        if len(adv) != len(inputs):
            return out
        n_ops    = torch.tensor(
            [float(x["n_ops"]) for x in inputs],
            device=adv.device, dtype=torch.float,
        )
        gold_est = GOLD_EST_SLOPE * n_ops + GOLD_EST_BASE
        over     = clen > (OVERLEN_FACTOR * gold_est)
        n_over   = int(over.sum().item())
        if n_over:
            out["advantages"] = torch.where(over, torch.zeros_like(adv), adv)
        self._metrics["overlen_frac"].append(over.float().mean().item())
        print(
            f"[lenctrl] masked {n_over}/{len(over)} over-length "
            f"| clen_max={int(clen.max().item())}",
            flush=True,
        )
        return out


def build_grpo_dataset(records: list) -> Dataset:
    rows = [{
        "prompt":    r["prompt"],
        "jobs_spec": _serialize_jobs(r["jobs_spec"]),
        "bks":       r["bks"] or 0,
        "n_ops":     r["n_ops"],
    } for r in records]
    return Dataset.from_list(rows)


def run_training(
    model,
    tokenizer,
    records: list,
    run_name: str,
    reward_mode: str = DEFAULT_REWARD_MODE,
    max_steps: int = NUM_TRAIN_STEPS,
    length_control: bool = False,
    resume_from: str | None = None,
    kl_coef: float = KL_COEF,
    grad_accum: int = GRAD_ACCUM_STEPS,
    temperature: float = TEMPERATURE,
    learning_rate: float = LEARNING_RATE,
    save_every: int = SAVE_EVERY,
    lp_alpha: float = 0.10,
    eos_beta: float = 0.05,
) -> str:
    """Run GRPO fine-tuning; return path to final saved adapter."""
    _TOKENIZER_REF["tok"] = tokenizer
    os.environ.setdefault("WANDB_MODE", "offline")

    print(
        f"[grpo] mode={reward_mode} K={K_SAMPLES} steps={max_steps} "
        f"length_control={length_control} resume={resume_from}"
    )

    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_grpo_dataset(records)

    config = GRPOConfig(
        output_dir=str(run_dir),
        run_name=run_name,
        learning_rate=learning_rate,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        per_device_train_batch_size=K_SAMPLES,
        gradient_accumulation_steps=grad_accum,
        num_generations=K_SAMPLES,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=temperature,
        beta=kl_coef,
        max_grad_norm=MAX_GRAD_NORM,
        save_steps=save_every,
        save_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to=["none"],
        seed=SEED,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        use_vllm=False,
    )

    reward_fn    = make_reward_fn(reward_mode, lp_alpha=lp_alpha, eos_beta=eos_beta)
    trainer_cls  = LengthControlledGRPOTrainer if length_control else GRPOTrainer
    trainer      = trainer_cls(
        model=model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    print(f"[grpo] saved final adapter -> {final_dir}")
    return str(final_dir)
