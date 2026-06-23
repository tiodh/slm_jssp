"""Pembungkus loader model dan setup LoRA / rsLoRA via Unsloth."""
from __future__ import annotations

import torch
from unsloth import FastLanguageModel

from .config import LORA_TARGET_MODULES, MODEL_REGISTRY


def _ensure_downloaded(repo_id: str) -> None:
    """Download model dari HF Hub jika belum ada di cache lokal."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, "config.json")
    if isinstance(cached, str):
        return  # sudah ada di cache

    print(f"[model] Model belum di-cache, downloading {repo_id!r} ...")
    print("[model] (proses ini hanya perlu dilakukan sekali, ~4-5 GB per model)")
    snapshot_download(repo_id, ignore_patterns=["*.pt", "original/"])
    print(f"[model] Download selesai: {repo_id}")


def load_base_model(
    model_key: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load base model + tokenizer dari registry.

    Args:
        model_key: salah satu dari MODEL_REGISTRY (llama, qwen2, granite, ministral).
        max_seq_length: panjang konteks maksimum.
        load_in_4bit: aktifkan bitsandbytes 4-bit (default: True).
        dtype: torch.bfloat16 atau torch.float16.

    Returns:
        (model, tokenizer)
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"model_key {model_key!r} tidak dikenal. "
            f"Pilihan: {list(MODEL_REGISTRY.keys())}"
        )

    spec = MODEL_REGISTRY[model_key]
    _ensure_downloaded(spec["hf_name"])

    kwargs = dict(
        model_name=spec["hf_name"],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if spec.get("eager_attention"):
        kwargs["attn_implementation"] = "eager"

    print(f"[model] Loading base: {spec['label']}  ({spec['hf_name']})")
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    return model, tokenizer


def attach_lora(
    model,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_rslora: bool = False,
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 42,
):
    """Pasang LoRA / rsLoRA adapter.

    rsLoRA = LoRA dengan scaling α/√r (Kalajdzievski 2023) — biasanya stabil
    untuk r ≥ 32. Aktifkan dengan use_rslora=True.
    """
    print(
        f"[model] Attach {'rsLoRA' if use_rslora else 'LoRA'} "
        f"r={lora_r} alpha={lora_alpha} dropout={lora_dropout}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=None,
    )
    return model


def load_for_inference(
    model_key: str,
    adapter_path: str,
    max_seq_length: int = 8192,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load base + LoRA adapter untuk inferensi."""
    from peft import PeftModel

    model, tokenizer = load_base_model(model_key, max_seq_length=max_seq_length, dtype=dtype)
    print(f"[model] Loading adapter dari {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer
