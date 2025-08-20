#!/usr/bin/env python3
"""
Fine-tune CodeS-1B for Italian Intent Classification
Fair comparison: train CodeS-1B on the same data as DistilBERT
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_example(text: str, intent: str) -> str:
    """Create a training example in CodeS format for Italian intent classification"""
    return f"""### Task: Classify this Italian customer service message into an intent category
### Message: {text}
### Intent: {intent}"""

def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess examples for training with proper label masking"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for text, intent in zip(examples["text"], examples["intent"]):
        # Create the input prompt (what model sees during inference)
        input_prompt = f"""### Task: Classify this Italian customer service message into an intent category
### Message: {text}
### Intent: """
        
        # Create the target (what we want model to generate)
        target = f"{intent}"
        
        # Tokenize input and target separately
        input_tokens = tokenizer(input_prompt, add_special_tokens=False)["input_ids"]
        target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
        
        # Combine for full sequence
        full_input_ids = input_tokens + target_tokens
        
        # Create labels: -100 for input portion, actual tokens for target portion
        labels = [-100] * len(input_tokens) + target_tokens
        
        # Truncate if too long
        if len(full_input_ids) > max_length:
            full_input_ids = full_input_ids[:max_length]
            labels = labels[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(full_input_ids)
        
        input_ids_list.append(full_input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in input_ids_list)
    max_len = min(max_len, max_length)
    
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    
    for input_ids, labels, attention_mask in zip(input_ids_list, labels_list, attention_mask_list):
        # Pad to max_len
        pad_length = max_len - len(input_ids)
        
        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_length)
        padded_labels.append(labels + [-100] * pad_length)  # Pad labels with -100
        padded_attention_mask.append(attention_mask + [0] * pad_length)
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_mask
    }

def load_data(train_path: str, val_path: str):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {train_path}")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    
    logger.info(f"Loading validation data from {val_path}")
    val_dataset = load_dataset("json", data_files=val_path, split="train")
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Show intent distribution
    intents = train_dataset['intent']
    intent_counts = {}
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    logger.info("Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        logger.info(f"  {intent}: {count}")
    
    return train_dataset, val_dataset

def setup_model_and_tokenizer(model_name: str, lora_r: int = 16):
    """Setup CodeS-1B model and tokenizer with LoRA"""
    logger.info(f"Loading CodeS-1B model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    logger.info(f"Applying LoRA with r={lora_r}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"]  # GPT BigCode architecture modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeS-1B for Italian Intent Classification")
    parser.add_argument("--model_name", type=str, default="seeklhy/codes-1b", 
                       help="CodeS-1B model to fine-tune")
    parser.add_argument("--train_path", type=str, default="data/train_intent_classification.jsonl",
                       help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/val_intent_classification.jsonl",
                       help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="outputs/codes-1b-intent-classifier",
                       help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    args = parser.parse_args()
    
    logger.info("Starting CodeS-1B Italian Intent Classification fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Load data
    train_dataset, val_dataset = load_data(args.train_path, args.val_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_r)
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments optimized for CodeS-1B and small dataset
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=20,  # Small for tiny dataset
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps", 
        save_steps=40,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=not args.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Get unique intents for inference template
    original_train = load_dataset("json", data_files=args.train_path, split="train")
    unique_intents = sorted(set(original_train['intent']))
    
    # Save inference template
    inference_template = f"""### CodeS-1B Italian Intent Classification Usage:

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")
base_model = AutoModelForCausalLM.from_pretrained("seeklhy/codes-1b", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "{args.output_dir}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Classify intent
def classify_intent(text):
    prompt = '''### Task: Classify this Italian customer service message into an intent category
### Message: {{text}}
### Intent: '''.format(text=text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract only generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean result
    intent = result.strip().split('\\n')[0].strip()
    if '###' in intent:
        intent = intent.split('###')[0].strip()
    
    return intent

# Available intents:
{', '.join(unique_intents)}

# Example usage:
# intent = classify_intent("Non riesco ad accedere al mio account")
# print(intent)  # Should output: account_access
"""
    
    with open(f"{args.output_dir}/inference_template.py", "w") as f:
        f.write(inference_template)
    
    # Save training configuration
    config = {
        "model_name": args.model_name,
        "training_args": training_args.to_dict(),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "unique_intents": unique_intents,
        "num_intents": len(unique_intents),
        "training_time_seconds": training_time,
        "final_train_loss": train_result.training_loss,
        "lora_r": args.lora_r,
        "timestamp": datetime.now().isoformat(),
        "comparison_note": "CodeS-1B vs Gemma3-270M vs DistilBERT for Italian intent classification"
    }
    
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("CodeS-1B Italian Intent Classification fine-tuning completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")

if __name__ == "__main__":
    main()