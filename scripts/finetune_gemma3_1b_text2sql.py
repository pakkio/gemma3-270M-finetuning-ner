#!/usr/bin/env python3
"""
Fine-tune Gemma3-1B for Italian Text-to-SQL task with 4-bit quantization
Comparison test against Gemma3-270M and Codes-1B
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

def create_training_example(question: str, sql: str) -> str:
    """Create training example for Italian text-to-SQL"""
    return f"""### Domanda
{question}

### Query SQL
{sql}"""

def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess examples for text-to-SQL training"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for question, sql in zip(examples["question"], examples["sql"]):
        # Create training text
        training_text = create_training_example(question, sql)
        
        # Find where SQL starts for label masking
        prompt_part = f"""### Domanda
{question}

### Query SQL
"""
        
        # Tokenize
        encoding = tokenizer(
            training_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Create labels - mask everything except SQL part
        prompt_encoding = tokenizer(prompt_part, add_special_tokens=True)
        prompt_length = len(prompt_encoding["input_ids"]) - 1  # -1 for the last token
        
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        
        # Pad to same length
        if len(labels) < len(input_ids):
            labels.extend(input_ids[len(labels):])
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
            
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }

def setup_model_and_tokenizer(model_name: str, lora_r: int = 16, lora_alpha: int = 32):
    """Setup Gemma3-1B model and tokenizer with LoRA and 4-bit quantization"""
    logger.info(f"Loading Gemma3-1B model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model {model_name} with 4-bit quantization for 4GB VRAM compatibility...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        device_map="auto",
        attn_implementation="eager"  # Recommended for Gemma3 training stability
    )
    
    logger.info(f"Applying LoRA with r={lora_r}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_text2sql_model(
    model_name: str = "google/gemma-3-1b",
    train_path: str = "data/train_text2sql.jsonl",
    val_path: str = "data/val_text2sql.jsonl",
    output_dir: str = "outputs/gemma3-1b-text2sql",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    """Train Gemma3-1B for Italian text-to-SQL"""
    
    start_time = time.time()
    logger.info("🚀 Starting Gemma3-1B Text-to-SQL Training")
    logger.info(f"📁 Training data: {train_path}")
    logger.info(f"📁 Validation data: {val_path}")
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"📈 Learning rate: {learning_rate}")
    logger.info(f"🔧 LoRA r={lora_r}, alpha={lora_alpha}")
    
    # Load and preprocess data
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
    
    val_data = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))
    
    logger.info(f"📊 Training examples: {len(train_data)}")
    logger.info(f"📊 Validation examples: {len(val_data)}")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(
        model_name, lora_r=lora_r, lora_alpha=lora_alpha
    )
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Preprocess
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("🎯 Starting training...")
    training_start = time.time()
    trainer.train()
    training_time = time.time() - training_start
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    total_time = time.time() - start_time
    
    # Save metrics
    metrics = {
        "model_name": model_name,
        "training_time_seconds": training_time,
        "total_time_seconds": total_time,
        "training_examples": len(train_data),
        "validation_examples": len(val_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "quantization": "4-bit",
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "unknown"),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Training completed!")
    logger.info(f"⏱️  Training time: {training_time:.1f} seconds")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"💾 Model saved to: {output_dir}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma3-1B for Text-to-SQL')
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b",
                       help="Model name")
    parser.add_argument("--train_path", type=str, default="data/train_text2sql.jsonl",
                       help="Training data path")
    parser.add_argument("--val_path", type=str, default="data/val_text2sql.jsonl",
                       help="Validation data path")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-1b-text2sql",
                       help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    metrics = train_text2sql_model(
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    print(f"\n🎯 Final Metrics:")
    print(f"Training time: {metrics['training_time_seconds']:.1f}s")
    print(f"Final eval loss: {metrics.get('final_eval_loss', 'N/A')}")

if __name__ == "__main__":
    main()