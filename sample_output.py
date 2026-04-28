"""Generate a single sample output for inspection."""
import sys
import json
import re
import random
import torch
from unsloth import FastLanguageModel

sys.stdout.reconfigure(line_buffering=True)

MODEL_DIR = "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-14400"
DATA_FILE = "./data/starjob_train.jsonl"
MAX_SEQ_LENGTH = 8192
SEED = 42

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

# Load test samples
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

# Pick samples of different sizes
sizes_wanted = ["5x5", "7x4", "10x5"]
samples = []
for td in test_data:
    m = re.search(r'(\d+)\s*Jobs.*?(\d+)\s*Machines', td["instruction"])
    if m:
        size = f"{m.group(1)}x{m.group(2)}"
        if size in sizes_wanted and size not in [s[1] for s in samples]:
            samples.append((td, size))
    if len(samples) == 3:
        break

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, MODEL_DIR)
FastLanguageModel.for_inference(model)

for sample, size in samples:
    instruction = sample["instruction"]
    input_text = sample["input"]
    true_output = sample["output"]
    true_makespan = extract_makespan(true_output)

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

    generated = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    pred_makespan = extract_makespan(generated)

    print(f"\n{'='*80}")
    print(f"PROBLEM SIZE: {size}")
    print(f"{'='*80}")
    print(f"\n--- INSTRUCTION ---")
    print(instruction)
    print(f"\n--- INPUT ---")
    print(input_text)
    print(f"\n--- TRUE OUTPUT (makespan={true_makespan}) ---")
    print(true_output)
    print(f"\n--- PREDICTED OUTPUT (makespan={pred_makespan}) ---")
    print(generated)
    print(f"\n--- COMPARISON ---")
    if pred_makespan and true_makespan:
        err = pred_makespan - true_makespan
        pct = abs(err) / true_makespan * 100
        print(f"True makespan: {true_makespan}")
        print(f"Pred makespan: {pred_makespan}")
        print(f"Error: {err:+d} ({pct:.1f}%)")
