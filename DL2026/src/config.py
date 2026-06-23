"""Konfigurasi terpusat: model registry, path dataset, prompt template, BKS.

Diimpor oleh modul lain di src/ dan oleh main.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

SM_TRAIN_FILE = DATA_DIR / "starjob_train_sm.jsonl"
LA_FILE = DATA_DIR / "lawrence_prompt_style.jsonl"
BENCH_DIR = DATA_DIR / "benchmarks"
JOBSHOP1 = BENCH_DIR / "jobshop1.txt"


MODEL_REGISTRY = {
    "llama": {
        "hf_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "eager_attention": False,
        "label": "LLaMA-3.1-8B-Instruct (4-bit)",
    },
    "qwen2": {
        "hf_name": "unsloth/Qwen2-7B-Instruct-bnb-4bit",
        "eager_attention": False,
        "label": "Qwen2-7B-Instruct (4-bit)",
    },
    "granite": {
        "hf_name": "unsloth/granite-3.2-8b-instruct-bnb-4bit",
        "eager_attention": True,
        "label": "Granite-3.2-8B-Instruct (4-bit)",
    },
    "ministral": {
        "hf_name": "mistralai/Ministral-8B-Instruct-2410",
        "eager_attention": True,
        "label": "Ministral-8B-Instruct-2410 (4-bit)",
    },
}


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


BEST_KNOWN_SOLUTION = {
    "ft06":  55, "ft10":  930, "ft20": 1165,
    "la01": 666, "la02":  655, "la03":  597, "la04": 590, "la05": 593,
    "la06": 926, "la07":  890, "la08":  863, "la09": 951, "la10": 958,
    "la16": 945, "la17":  784, "la18":  848, "la19": 842, "la20": 902,
}


WANTED_FT = ["ft06", "ft10", "ft20"]
WANTED_LA = [f"la{i:02d}" for i in range(1, 11)] + [f"la{i:02d}" for i in range(16, 21)]


DEFAULT_HYPERPARAMS = dict(
    max_seq_length=8192,
    lora_r=32,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    seed=42,
    save_steps=200,
    save_total_limit=5,
    logging_steps=10,
)


MAX_NEW_TOKENS_EVAL = 7000
