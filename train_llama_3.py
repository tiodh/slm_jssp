import os
os.environ["UNSLOTH_NUM_PROC"] = "1"
import argparse
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    parser = argparse.ArgumentParser(description="Train Llama-3.1-8B on JSSP scheduling.")

    # Model and data parameters
    parser.add_argument('--max_seq_length', type=int, default=8192)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--load_in_4bit', action='store_true', default=True)

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=str, default='none', choices=['none', 'all', 'lora_only'])

    # Additional configurations
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--use_rslora', action='store_true', default=True)
    parser.add_argument('--loftq_config', type=str, default=None)

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adamw_8bit')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_total_limit', type=int, default=50)
    parser.add_argument('--save_step', type=int, default=200)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)

    # Output directory
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    # Generate output directory name
    if args.output_dir is None:
        rslora_tag = "_rslora" if args.use_rslora else ""
        dir_out = f"output_llama8b{rslora_tag}_alpha{args.lora_alpha}_r{args.lora_r}_seq{args.max_seq_length}_b{args.per_device_train_batch_size}_ga{args.gradient_accumulation_steps}_ep{args.num_train_epochs}"
    else:
        dir_out = args.output_dir

    # Wandb disabled; tensorboard logs written inside output_dir/runs

    # Load Model and Tokenizer
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    # Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # Load and Prepare Dataset
    dataset = load_dataset('json', data_files="./data/starjob_train_sm.jsonl", split="train")
    split_dataset = dataset.train_test_split(test_size=0.02, seed=args.seed)
    train_dataset = split_dataset['train'].map(formatting_prompts_func, batched=True)
    eval_dataset = split_dataset['test'].map(formatting_prompts_func, batched=True)

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Initialize the Trainer (no custom eval callback — avoids segfault with Unsloth + use_cache)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=1,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_step,
            eval_strategy="steps",
            eval_steps=args.save_step,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        ),
    )

    # GPU Memory Info
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Start Training (auto-resume from latest checkpoint if any)
    latest_ckpt = None
    if os.path.isdir(dir_out):
        ckpts = [d for d in os.listdir(dir_out) if d.startswith("checkpoint-")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("-")[1]))
            latest_ckpt = os.path.join(dir_out, ckpts[-1])
            print(f"Resuming from {latest_ckpt}")
    trainer_stats = trainer.train(resume_from_checkpoint=latest_ckpt) if latest_ckpt else trainer.train()

if __name__ == "__main__":
    main()
