"""Evaluate fine-tuned Ministral-8B on makespan prediction from generated schedules."""
import os
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
import sys
import json
import re
import random
import torch
from unsloth import FastLanguageModel

sys.stdout.reconfigure(line_buffering=True)

# Config
MODEL_DIR = "output_ministral8b_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400"
BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
DATA_FILE = "./data/starjob_train.jsonl"
NUM_SAMPLES = 50
MAX_SEQ_LENGTH = 8192
SEED = 42

random.seed(SEED)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """

def extract_makespan(schedule_text):
    times = re.findall(r'->\s*(\d+)', schedule_text)
    if not times:
        return None
    return max(int(t) for t in times)

def extract_problem_size(instruction):
    m = re.search(r'(\d+)\s*Jobs.*?(\d+)\s*Machines', instruction)
    if m:
        return f"{m.group(1)}x{m.group(2)}"
    return "unknown"

def main():
    print("Loading dataset...")
    all_data = []
    with open(DATA_FILE) as f:
        for line in f:
            all_data.append(json.loads(line))

    random.seed(SEED)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    test_size = int(len(all_data) * 0.02)
    test_indices = indices[:test_size]
    test_data = [all_data[i] for i in test_indices]

    samples = random.sample(test_data, min(NUM_SAMPLES, len(test_data)))
    print(f"Evaluating on {len(samples)} samples")

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        attn_implementation="eager",
    )
    print(f"Loading LoRA adapter from {MODEL_DIR}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, MODEL_DIR)
    FastLanguageModel.for_inference(model)

    results = []
    valid_count = 0
    exact_match = 0
    total_ae = 0
    total_ape = 0
    total_signed = 0
    pred_leq_true = 0

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        input_text = sample["input"]
        true_output = sample["output"]
        true_makespan = extract_makespan(true_output)
        size = extract_problem_size(instruction)

        if true_makespan is None:
            print(f"  [{i+1}] Skipping - can't parse true makespan")
            continue

        prompt = alpaca_prompt.format(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
            )

        n_gen_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        truncated = n_gen_tokens >= 4096
        trunc_tag = " [TRUNCATED]" if truncated else ""
        generated = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_makespan = extract_makespan(generated)

        if pred_makespan is not None:
            valid_count += 1
            ae = abs(pred_makespan - true_makespan)
            ape = ae / true_makespan * 100
            signed_err = pred_makespan - true_makespan
            total_ae += ae
            total_ape += ape
            total_signed += signed_err
            if pred_makespan == true_makespan:
                exact_match += 1
            if pred_makespan <= true_makespan:
                pred_leq_true += 1

            status = "EXACT" if pred_makespan == true_makespan else f"err={signed_err:+d} ({ape:.1f}%)"
            print(f"  [{i+1}/{len(samples)}] {size} | True={true_makespan} Pred={pred_makespan} | {status} | n_tok={n_gen_tokens}{trunc_tag}")
        else:
            print(f"  [{i+1}/{len(samples)}] {size} | True={true_makespan} | INVALID (no schedule parsed) | n_tok={n_gen_tokens}{trunc_tag}")
            print(f"    Generated tail: {generated[-200:]}")

    print("\n" + "="*60)
    print(f"MINISTRAL-8B MAKESPAN EVALUATION RESULTS")
    print(f"="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Valid predictions: {valid_count}/{len(samples)} ({valid_count/len(samples)*100:.1f}%)")
    if valid_count > 0:
        print(f"Exact match: {exact_match}/{valid_count} ({exact_match/valid_count*100:.1f}%)")
        print(f"Pred <= True (better/equal): {pred_leq_true}/{valid_count} ({pred_leq_true/valid_count*100:.1f}%)")
        print(f"MAE: {total_ae/valid_count:.1f}")
        print(f"MAPE: {total_ape/valid_count:.1f}%")
        print(f"Mean Signed Error: {total_signed/valid_count:.1f}")

if __name__ == "__main__":
    main()
