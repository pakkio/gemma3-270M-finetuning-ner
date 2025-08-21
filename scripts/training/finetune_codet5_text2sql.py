#!/usr/bin/env python3
"""
Fine-tune CodeT5 for Italian Text-to-SQL task
Baseline comparison model for Gemma3 evaluation
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
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_prompt(question: str) -> str:
    """Create a prompt for the Text-to-SQL task"""
    return f"Converti la seguente domanda italiana in una query SQL: {question}"

def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess examples for training"""
    # Create input prompts
    inputs = [create_prompt(q) for q in examples["question"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    # Tokenize targets (SQL queries)
    labels = tokenizer(
        text_target=examples["sql"],
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_data(train_path: str, val_path: str):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {train_path}")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    
    logger.info(f"Loading validation data from {val_path}")
    val_dataset = load_dataset("json", data_files=val_path, split="train")
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def setup_model_and_tokenizer(model_name: str, use_lora: bool = True, lora_r: int = 16):
    """Setup model and tokenizer with optional LoRA"""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if use_lora:
        logger.info(f"Applying LoRA with r={lora_r}")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.1,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for Italian Text-to-SQL")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5p-220m", 
                       help="CodeT5 model to fine-tune")
    parser.add_argument("--train_path", type=str, default="data/train_text2sql.jsonl",
                       help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/val_text2sql.jsonl",
                       help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="outputs/codet5-text2sql",
                       help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    args = parser.parse_args()
    
    logger.info("Starting CodeT5 Text-to-SQL fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Load data
    train_dataset, val_dataset = load_data(args.train_path, args.val_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, 
        use_lora=not args.no_lora,
        lora_r=args.lora_r
    )
    
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
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=not args.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"  # Disable wandb
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
    
    # Save training configuration
    config = {
        "model_name": args.model_name,
        "training_args": training_args.to_dict(),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "training_time_seconds": training_time,
        "final_train_loss": train_result.training_loss,
        "use_lora": not args.no_lora,
        "lora_r": args.lora_r if not args.no_lora else None,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save inference template
    template = f"""# CodeT5 Text-to-SQL Inference Template

## Model Information
- Base Model: {args.model_name}
- Fine-tuned for: Italian Text-to-SQL
- Training Time: {training_time:.2f} seconds
- Training Examples: {len(train_dataset)}
- Validation Examples: {len(val_dataset)}

## Usage Example
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")
model = T5ForConditionalGeneration.from_pretrained("{args.output_dir}")

question = "Mostra tutti i clienti di Milano"
prompt = "Converti la seguente domanda italiana in una query SQL: " + question
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(sql)
```

## Prompt Format
Input: "Converti la seguente domanda italiana in una query SQL: {{question}}"
Output: "{{sql_query}}"
"""
    
    with open(f"{args.output_dir}/inference_template.txt", "w") as f:
        f.write(template)
    
    logger.info("CodeT5 fine-tuning completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")

if __name__ == "__main__":
    main()