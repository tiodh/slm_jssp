"""Pembungkus tipis TRL SFTTrainer untuk fine-tuning JSSP."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from .config import DEFAULT_HYPERPARAMS, OUTPUTS_DIR


def make_output_dir(model_key: str, use_rslora: bool, hp: dict) -> Path:
    tag = "rslora" if use_rslora else "lora"
    name = (
        f"{model_key}_{tag}"
        f"_r{hp['lora_r']}_a{hp['lora_alpha']}"
        f"_seq{hp['max_seq_length']}"
        f"_b{hp['per_device_train_batch_size']}"
        f"_ga{hp['gradient_accumulation_steps']}"
    )
    return OUTPUTS_DIR / name


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    output_dir: Path,
    hp: dict,
    max_steps: int = -1,
):
    """Buat SFTTrainer dengan TrainingArguments dari hyperparams."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fp16 = not is_bfloat16_supported()
    bf16 = is_bfloat16_supported()

    training_args = TrainingArguments(
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        warmup_steps=hp["warmup_steps"],
        num_train_epochs=hp["num_train_epochs"] if max_steps <= 0 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=hp["learning_rate"],
        fp16=fp16,
        bf16=bf16,
        logging_steps=hp["logging_steps"],
        optim=hp["optim"],
        weight_decay=hp["weight_decay"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        seed=hp["seed"],
        output_dir=str(output_dir),
        save_steps=hp["save_steps"],
        save_total_limit=hp["save_total_limit"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=hp["max_seq_length"],
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )
    return trainer


def run_training(
    model,
    tokenizer,
    train_dataset,
    model_key: str,
    use_rslora: bool,
    max_steps: int = -1,
    overrides: dict | None = None,
):
    """Konfigurasi training argument lalu jalankan .train(); return path adapter."""
    hp = dict(DEFAULT_HYPERPARAMS)
    if overrides:
        hp.update({k: v for k, v in overrides.items() if v is not None})

    out_dir = make_output_dir(model_key, use_rslora, hp)
    trainer = build_trainer(
        model, tokenizer, train_dataset, out_dir, hp, max_steps=max_steps
    )

    print(f"[train] Output dir: {out_dir}")
    print(f"[train] Mulai training (max_steps={max_steps if max_steps > 0 else 'epoch-based'})...")
    trainer.train()

    final_dir = out_dir / "final_adapter"
    print(f"[train] Menyimpan adapter ke {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir
