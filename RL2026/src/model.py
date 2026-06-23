"""Model loading for CAP-GRPO: base model (auto-download) or local adapter path."""
from __future__ import annotations

import torch
from unsloth import FastLanguageModel

from .config import MAX_SEQ_LENGTH, MODEL_REGISTRY


def _ensure_downloaded(repo_id: str) -> None:
    """Download model from HF Hub if not already cached (idempotent)."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    cached = try_to_load_from_cache(repo_id, "config.json")
    if isinstance(cached, str):
        return
    print(f"[model] Not in cache — downloading {repo_id!r} (~4-5 GB, once only) ...")
    snapshot_download(repo_id, ignore_patterns=["*.pt", "original/"])
    print(f"[model] Download done: {repo_id}")


def load_model(
    model_key: str | None = None,
    model_path: str | None = None,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    for_training: bool = True,
):
    """Load model + tokenizer.

    Provide either model_key (registered name → auto-download base model)
    or model_path (local SFT adapter / checkpoint directory).

    Args:
        model_key  : one of MODEL_REGISTRY keys (llama/qwen2/granite/ministral)
        model_path : path to a local HF model or SFT adapter directory
        for_training: call model.train() if True, FastLanguageModel.for_inference() if False
    """
    if model_key is not None and model_path is not None:
        raise ValueError("Provide model_key or model_path, not both.")
    if model_key is None and model_path is None:
        raise ValueError("Provide model_key or model_path.")

    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"model_key {model_key!r} unknown. Choices: {list(MODEL_REGISTRY)}"
            )
        spec   = MODEL_REGISTRY[model_key]
        hf_id  = spec["hf_name"]
        eager  = spec.get("eager", False)
        label  = spec["label"]
        _ensure_downloaded(hf_id)
        name_or_path = hf_id
    else:
        eager        = False
        label        = model_path
        name_or_path = model_path

    print(f"[model] Loading: {label}  ({name_or_path})")
    kwargs = dict(
        model_name=name_or_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if eager:
        kwargs["attn_implementation"] = "eager"

    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    if for_training:
        model.train()
    else:
        FastLanguageModel.for_inference(model)

    return model, tokenizer
