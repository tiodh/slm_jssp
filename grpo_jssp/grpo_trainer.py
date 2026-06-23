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


_TOKENIZER_REF = {"tok": None}  # set by train() before reward_fn runs


def _serialize_jobs(jobs_spec):
    """jobs_spec is list[list[tuple[int,int]]]; tuples don't survive HF Dataset
    cleanly, so we serialize to JSON string and parse back inside reward_fn."""
    return json.dumps(jobs_spec)


def _deserialize_jobs(s):
    return [[tuple(op) for op in job] for job in json.loads(s)]


def _gen_len_and_eos(text: str):
    """For V2: token length and EOS flag of one completion. Uses the tokenizer
    captured into _TOKENIZER_REF by train(). Falls back to a char/4 estimate if
    the tokenizer isn't set (e.g. unit tests)."""
    tok = _TOKENIZER_REF.get("tok")
    if tok is None:
        return max(1, len(text) // 4), text.rstrip().endswith("<|eot_id|>") or text.endswith(tok.eos_token if tok else "")
    ids = tok.encode(text, add_special_tokens=False)
    gen_len = len(ids)
    eos_id = tok.eos_token_id
    ended_with_eos = (eos_id is not None) and (len(ids) > 0) and (ids[-1] == eos_id)
    return gen_len, ended_with_eos


def make_reward_fn(mode: str, lp_alpha: float = 0.10, eos_beta: float = 0.05):
    needs_len = (mode == "stratified_v2")

    def reward_fn(completions, jobs_spec, bks, n_ops, **kwargs):
        rewards = []
        n_parseable = 0
        n_feasible = 0
        for comp, js_json, b, n in zip(completions, jobs_spec, bks, n_ops):
            text = comp if isinstance(comp, str) else comp[0]["content"]
            js = _deserialize_jobs(js_json)
            v = check_violations(text, js)
            bks_val = None if b in (0, None) else int(b)
            if needs_len:
                gen_len, ended_with_eos = _gen_len_and_eos(text)
                r = compute_reward(v, int(n), bks_val, mode=mode,
                                   gen_len=gen_len, ended_with_eos=ended_with_eos,
                                   lp_alpha=lp_alpha, eos_beta=eos_beta)
            else:
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
          length_control: bool = False,
          resume_from: str | None = None,
          sft_checkpoint: str | None = None,
          kl_coef: float | None = None,
          grad_accum: int | None = None,
          temperature: float | None = None,
          learning_rate: float | None = None,
          lp_alpha: float = 0.10,
          eos_beta: float = 0.05,
          save_every: int | None = None):
    sft_ckpt = sft_checkpoint or str(SFT_CHECKPOINT)
    kl = KL_COEF if kl_coef is None else kl_coef
    ga = GRAD_ACCUM_STEPS if grad_accum is None else grad_accum
    temp = TEMPERATURE if temperature is None else temperature
    lr = LEARNING_RATE if learning_rate is None else learning_rate
    save = SAVE_EVERY if save_every is None else save_every

    print(f"[grpo] reward_mode={reward_mode}, K={K_SAMPLES}, steps={max_steps}, "
          f"length_control={length_control}, resume_from={resume_from}")
    print(f"[grpo] sft_ckpt={sft_ckpt}")
    print(f"[grpo] overrides: KL={kl}, grad_accum={ga}, T={temp}, LR={lr}, "
          f"lp_alpha={lp_alpha}, eos_beta={eos_beta}, save_every={save}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_ckpt,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    # adapter is loaded from SFT_CHECKPOINT; switch to training mode
    model.train()
    _TOKENIZER_REF["tok"] = tokenizer  # for V2 reward length/EOS measurement

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
        learning_rate=lr,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        per_device_train_batch_size=K_SAMPLES,
        gradient_accumulation_steps=ga,
        num_generations=K_SAMPLES,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=temp,
        beta=kl,
        max_grad_norm=MAX_GRAD_NORM,
        save_steps=save,
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
        reward_funcs=[make_reward_fn(reward_mode, lp_alpha=lp_alpha, eos_beta=eos_beta)],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    if resume_from:
        print(f"[grpo] resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    print(f"[grpo] saved final adapter -> {final_dir}")
    return final_dir


if __name__ == "__main__":
    train()
