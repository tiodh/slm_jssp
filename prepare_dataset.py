"""Download Starjob dataset from HuggingFace and filter for training."""
import json
import re
from datasets import load_dataset

def extract_size(instruction_text):
    """Extract num_jobs and num_machines from instruction field."""
    m = re.search(r'(\d+)\s*Jobs.*?(\d+)\s*Machines', instruction_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def main():
    print("Downloading dataset from HuggingFace: mideavalwisard/Starjob")
    ds = load_dataset("mideavalwisard/Starjob", data_files="starjob_130k_filled.json", split="train")
    print(f"Total samples: {len(ds)}")

    filtered = []
    for sample in ds:
        num_jobs, num_machines = extract_size(sample["instruction"])
        if num_jobs is not None and num_jobs <= 10 and num_machines <= 10:
            filtered.append({
                "instruction": sample["instruction"],
                "input": sample["input"],
                "output": sample["output"],
            })

    print(f"Filtered (jobs<=10, machines<=10): {len(filtered)}")

    out_path = "./data/starjob_train_sm.jsonl"
    with open(out_path, "w") as f:
        for item in filtered:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
