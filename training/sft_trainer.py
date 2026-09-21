"""
Supervised Fine-Tuning (SFT) Trainer script for Qwen model using TRL and PEFT LoRA.
Designed to run on Cloud GPU, with CPU dry-run safety flags for local verification.
"""

import argparse
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="StartupTN Proposal Evaluation LLM SFT Trainer")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="Base model path")
    parser.add_argument("--dataset_path", type=str, default="data/processed/startup_proposals_synthetic.jsonl",
                        help="Processed JSONL dataset path or glob pattern (e.g. data/processed/*.jsonl to merge all files)")
    parser.add_argument("--output_dir", type=str, default="models/qwen_startup_lora", help="Output directory for LoRA adapter")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Perform setup check without launching full training")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("StartupTN LLM Supervised Fine-Tuning Pipeline (LoRA)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target execution device: {device}")
    
    if device == "cpu" and not args.dry_run:
        print("[WARNING] Running full LLM fine-tuning on CPU is very slow. Consider running with --dry-run locally or on a Cloud GPU.")

    # 1. Load Dataset (supports glob patterns to merge multiple JSONL files)
    import glob as _glob
    matched_files = _glob.glob(args.dataset_path)
    if not matched_files:
        raise FileNotFoundError(f"No dataset files matched pattern '{args.dataset_path}'.")

    matched_files = sorted(matched_files)
    print(f"Loading dataset from {len(matched_files)} file(s):")
    for f in matched_files:
        print(f"  - {f}")
    raw_dataset = load_dataset("json", data_files=matched_files, split="train")
    print(f"Loaded {len(raw_dataset)} dataset records total.")

    # 2. Load Tokenizer & Model
    print(f"Loading tokenizer & model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    # 3. Configure LoRA PEFT
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    if args.dry_run:
        print("\n[DRY RUN SUCCESSFUL] LoRA configuration, tokenizer, dataset, and model initialization verified cleanly!")
        return

    # 4. Training Arguments & SFTTrainer Setup
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        fp16=(device == "cuda"),
        bf16=False,
        use_cpu=(device == "cpu"),
        dataset_text_field="messages",
        max_length=1024
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=raw_dataset,
        processing_class=tokenizer
    )

    print("\nStarting SFT Training...")
    trainer.train()
    
    # Save fine-tuned adapter
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA Adapter saved successfully to: {args.output_dir}")

if __name__ == "__main__":
    main()
