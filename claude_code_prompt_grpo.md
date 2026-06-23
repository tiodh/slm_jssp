# GRPO with Stratified Reward for JSSP Fine-Tuned LLaMA

## Objective

Implement a Group Relative Policy Optimization (GRPO) training pipeline that takes an existing SFT-finetuned LLaMA 3.1-8B model (trained on StarJob JSSP dataset) and further trains it using reinforcement learning with a stratified reward function. The goal is to improve feasibility rate on JSSP schedule generation.

## Background

I have a fine-tuned LLaMA 3.1-8B model (using rsLoRA, 4-bit quantization) that generates JSSP schedules in natural language format. It achieves 95.5% feasibility on in-distribution StarJob instances but drops to 56% on out-of-distribution classical benchmarks.

The model generates schedules in this format:
```
J0-M2: 0+8→8, J1-M0: 0+1→1, J2-M1: 0+4→4, J1-M2: 8+5→13, J0-M0: 8+5→13, J2-M2: 13+6→19, J0-M1: 13+5→18, J1-M1: 13+4→17, J2-M0: 13+5→18 Makespan: 19
```

Each token represents: `J{job}-M{machine}: {start}+{duration}→{end}`

## Existing Setup

- **Base model**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` fine-tuned with rsLoRA
- **SFT adapter**: saved as LoRA adapter (I will provide the path)
- **Hardware**: single NVIDIA RTX 4090 (24GB VRAM)
- **Training data**: StarJob dataset, filtered to ≤10×10 instances, Alpaca prompt format
- **Libraries**: Unsloth, Transformers, PEFT, TRL, bitsandbytes
- **Fine-tuning config**: rsLoRA r=32, α=32, 4-bit quantization, max_seq_length=8192
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## What to Implement

### 1. Constraint Checker (reward computation)

Build a constraint checker that parses a generated schedule string and returns violation counts for 5 types:

- **Missing Operation (M)**: operations that should exist but are missing from the schedule
- **Routing Order (R)**: operations of a job appear in wrong machine order (not following the input's prescribed sequence)
- **Machine Capacity (C)**: two or more operations overlap on the same machine
- **Timing Consistency (T)**: start + duration ≠ end, or an operation starts before its predecessor ends
- **Precedence (P)**: an operation of a job starts before the previous operation of the same job finishes

The checker takes:
- The generated schedule string
- The input JSSP instance specification (which defines job-machine assignments and durations)

And returns:
```python
{
    "feasible": bool,
    "missing_op_count": int,
    "routing_order_violations": int,
    "machine_capacity_violations": int,
    "timing_consistency_violations": int,
    "precedence_violations": int,
    "total_violations": int,
    "makespan": int or None  # None if unparseable
}
```

### 2. Stratified Reward Function

Implement the reward function with these pre-computed weights (derived from OOD violation analysis):

```python
# Weights derived from empirical OOD violation distribution (sum to 1.0)
WEIGHTS = {
    "missing":    0.0929,  # 288 / 3100
    "routing":    0.3190,  # 989 / 3100
    "capacity":   0.2755,  # 854 / 3100
    "timing":     0.2555,  # 792 / 3100
    "precedence": 0.0571,  # 177 / 3100
}

def compute_reward(violations: dict, n_ops: int, bks: int = None) -> float:
    """
    Infeasible: R = -(w_M * n_M/N_ops + w_R * n_R/N_ops + w_C * n_C/N_ops + w_T * n_T/N_ops + w_P * n_P/N_ops)
    Feasible:   R = BKS / Cmax  (if BKS available, else R = 1.0)
    """
```

### 3. GRPO Training Pipeline

Use TRL's `GRPOTrainer` (or implement manually if TRL version doesn't support it well). The pipeline:

1. Load the SFT model with its LoRA adapter as the base policy
2. For each training step:
   a. Sample a batch of JSSP instances from StarJob
   b. Generate K solutions per instance (e.g., K=4 or K=8, depending on VRAM)
   c. Parse each solution and run constraint checker
   d. Compute stratified reward for each solution
   e. Compute group-relative advantage (normalize within each instance's K solutions)
   f. Update policy using policy gradient
3. Save checkpoints periodically
4. Evaluate on both StarJob (in-distribution) and classical benchmarks (OOD)

### 4. Evaluation Script

After GRPO training, evaluate on:
- **In-distribution**: 200 StarJob test instances
- **Out-of-distribution**: 18 classical JSSP instances (ft06, ft10, ft20, la01-la05, la06-la10, la16-la20)

Report: feasibility rate, violation counts per type, makespan gap to BKS.

## Key Constraints

- **VRAM**: 24GB on RTX 4090. The model is 4-bit quantized (~5GB). K solutions per instance means K forward passes. Start with K=4 and see if K=8 fits.
- **Generation length**: JSSP schedules can be up to 7000 tokens for 10×10 instances. Set max_new_tokens accordingly.
- **Parsing robustness**: Generated schedules may be malformed. The parser must handle partial/broken outputs gracefully (return violations rather than crashing).
- **Temperature**: Use temperature=0.7 or 0.8 for generation diversity during GRPO. The model needs variance in outputs for group-relative comparison to work.

## Data Format

StarJob instances use Alpaca format:

```
### Instruction:
Optimize the schedule for {n} Jobs across {m} Machines to minimize the makespan...

### Input:
J0: M2:8 M0:5 M1:5
J1: M0:1 M2:5 M1:4
J2: M1:4 M2:6 M0:5

### Response:
J0-M2: 0+8→8, J1-M0: 0+1→1, ... Makespan: 19
```

The Input section defines each job's operation sequence: `J0: M2:8 M0:5 M1:5` means Job 0 must first go to Machine 2 (duration 8), then Machine 0 (duration 5), then Machine 1 (duration 5). This ordering is the required routing.

## File Structure

Please organize the code as:

```
grpo_jssp/
├── config.py           # All hyperparameters and paths
├── constraint_checker.py   # Parse schedule + check 5 constraints
├── reward.py           # Stratified reward function
├── grpo_trainer.py     # Main GRPO training loop
├── evaluate.py         # Evaluation on StarJob + OOD benchmarks
├── data_utils.py       # Dataset loading and formatting
└── run.py              # Entry point
```

## Experimental Comparison

The code should support running two reward modes for ablation:

1. **Uniform reward**: binary (1.0 if feasible, -1.0 if infeasible)  
2. **Stratified reward**: weighted violation penalty as described above

This is controlled by a config flag so I can compare both approaches.

## Additional Notes

- I will provide the path to my SFT model adapter when running the code
- I will provide the StarJob dataset path
- I will provide the OOD benchmark instances
- Use logging (wandb or tensorboard) to track: reward mean/std per epoch, feasibility rate, violation counts
- Save the model every N steps as LoRA adapter (not full model merge)
- Print progress: current step, mean reward, feasibility rate, per-violation-type counts
