"""GRPO-JSSP configuration: paths, model, training, reward."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Model / SFT adapter ---
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
SFT_ADAPTER_DIR = REPO_ROOT / "output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1"
# checkpoint-9800 is the "best" by trainer_state.json (lowest eval loss), used
# in the prior rsLoRA evals reported in metrics_rslora_llama.json. checkpoint-13230
# is the final-epoch save. We use 9800 to keep continuity with prior baselines.
SFT_CHECKPOINT = SFT_ADAPTER_DIR / "checkpoint-9800"

# --- Train/test split (must match SFT training to keep test set truly held out) ---
SPLIT_SEED = 42
TEST_FRAC = 0.02

# --- Data ---
STARJOB_SM_PATH = REPO_ROOT / "data" / "starjob_train_sm.jsonl"
BENCH_DIR = REPO_ROOT / "data" / "benchmarks"
JOBSHOP1_PATH = BENCH_DIR / "jobshop1.txt"

OOD_INSTANCES = [
    "ft06", "ft10", "ft20",
    "la01", "la02", "la03", "la04", "la05",
    "la06", "la07", "la08", "la09", "la10",
    "la16", "la17", "la18", "la19", "la20",
]

BEST_KNOWN = {
    "ft06":  55, "ft10":  930, "ft20": 1165,
    "la01": 666, "la02":  655, "la03":  597, "la04": 590, "la05": 593,
    "la06": 926, "la07":  890, "la08":  863, "la09": 951, "la10": 958,
    "la16": 945, "la17":  784, "la18":  848, "la19": 842, "la20": 902,
}

# --- Generation ---
MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 4096  # JSSP 10x10 ≈ 100 ops × ~20 tok = ~2k; 4k gives headroom
TEMPERATURE = 0.7        # V3: lower stochasticity for more stable exploration
TOP_P = 0.95

# --- GRPO ---
K_SAMPLES = 4
LEARNING_RATE = 5e-6     # LoRA-friendly mid-range
NUM_TRAIN_STEPS = 2000
SAVE_EVERY = 50          # V5 relaunch: frequent checkpoints — machine hard-froze
                         # mid-V5 at step 54; 50 caps loss to <=50 steps
EVAL_EVERY = 500
LOGGING_STEPS = 5
KL_COEF = 0.05           # V3: relax from V2 (0.10) — soft anchor to SFT
MAX_GRAD_NORM = 1.0      # clip single-step gradient spikes
GRAD_ACCUM_STEPS = 4     # V4/V5: 4 prompts per weight update
WARMUP_STEPS = 20

# --- Reward (V4 Hybrid P-GRPO) ---
# R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality, range [-1.0 .. 7.0].
# Each constraint contributes equally (+1 satisfied / -(n/N_ops) violated), so
# no per-category weights are needed -- this also removes the prior OOD
# test-set leakage concern from deriving weights off a violation distribution.
REWARD_MODE = "hybrid"  # "hybrid" (V4) | "uniform" (V4-scale ablation) | "stratified" (V1)

# --- V1 stratified reward weights (frozen, do not re-derive without renaming) ---
# Original V1 design: per-category weight = count / 233, where 233 is the total
# number of violations the rsLoRA-SFT model produced on 200 SM samples (seed=42).
# Range of the resulting reward is approx [-1, 1]. Recorded here so the V1
# ablation reproduces the exact original V1 reward without re-measuring.
V1_WEIGHTS = {
    "missing_op_count":               4 / 233,   # 0.0172
    "routing_order_violations":      33 / 233,   # 0.1416
    "machine_capacity_violations":  145 / 233,   # 0.6223
    "timing_consistency_violations": 33 / 233,   # 0.1416
    "precedence_violations":         18 / 233,   # 0.0773
}

# --- V5 length control (advantage masking) ---
# A sample whose completion_length exceeds OVERLEN_FACTOR x gold_est gets its
# GRPO advantage zeroed -- it contributes no gradient (neither reward nor
# penalty). Removes the length-escape collapse trigger without a soft-penalty
# reward cliff. gold_est = GOLD_EST_SLOPE * N_ops + GOLD_EST_BASE.
GOLD_EST_SLOPE = 12.5
GOLD_EST_BASE = 50
OVERLEN_FACTOR = 2.0

# --- Output ---
OUTPUT_DIR = REPO_ROOT / "grpo_jssp" / "runs"
SEED = 42
