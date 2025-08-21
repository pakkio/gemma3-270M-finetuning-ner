#!/usr/bin/env python3
"""
Training migliorato per Codes-1B Intent Classification
Affronta i problemi identificati: dataset piccolo, parametri sub-ottimali
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_example(text: str, intent: str) -> str:
    """Create training example for intent classification"""
    return f"""### Testo
{text}

### Intent
{intent}"""

def preprocess_function(examples, tokenizer, max_length=256):
    """Preprocess examples for intent classification"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for text, intent in zip(examples["text"], examples["intent"]):
        # Create training text
        training_text = create_training_example(text, intent)
        
        # Find where intent starts for label masking
        prompt_part = f"""### Testo
{text}

### Intent
"""
        
        # Tokenize with consistent padding
        encoding = tokenizer(
            training_text,
            truncation=True,
            max_length=max_length,
            padding='max_length',  # Sempre pad a max_length
            add_special_tokens=True,
            return_tensors=None  # Keep as lists for now
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Create labels - mask everything except intent part
        prompt_encoding = tokenizer(prompt_part, add_special_tokens=True, padding=False)
        prompt_length = len(prompt_encoding["input_ids"]) - 1  # -1 for the last token
        
        # Initialize labels with -100 (ignored in loss)
        labels = [-100] * max_length
        
        # Copy actual tokens for the intent part, respect padding
        for i in range(prompt_length, len(input_ids)):
            if input_ids[i] != tokenizer.pad_token_id:  # Don't learn from padding
                labels[i] = input_ids[i]
            
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }

def setup_model_and_tokenizer(model_name: str, lora_r: int = 16, lora_alpha: int = 32):
    """Setup Codes-1B model with optimal parameters for intent classification"""
    logger.info(f"Loading Codes-1B model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model {model_name} with 4-bit quantization optimized for intent classification...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    logger.info(f"Applying LoRA with optimized parameters: r={lora_r}, alpha={lora_alpha}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,  # Ridotto per intent classification
        bias="none",
        target_modules=["c_attn", "c_proj", "c_fc"]  # Architettura Codes-1B
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_intent_model(
    model_name: str = "seeklhy/codes-1b",
    train_path: str = "data/intent_classification/expanded/train_expanded.jsonl",
    val_path: str = "data/intent_classification/expanded/val_expanded.jsonl",
    output_dir: str = "models/production/codes1b-intent-improved",
    epochs: int = 8,  # Aumentato
    batch_size: int = 8,  # Aumentato
    learning_rate: float = 5e-4,  # Aumentato
    max_length: int = 256,
    lora_r: int = 32,  # Aumentato
    lora_alpha: int = 64  # Aumentato
):
    """Train Codes-1B for Italian intent classification with improved parameters"""
    
    start_time = time.time()
    logger.info("🚀 Starting IMPROVED Codes-1B Intent Classification Training")
    logger.info(f"📁 Training data: {train_path}")
    logger.info(f"📁 Validation data: {val_path}")
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"📈 Learning rate: {learning_rate} (INCREASED)")
    logger.info(f"🔧 LoRA r={lora_r}, alpha={lora_alpha} (OPTIMIZED)")
    
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
    
    logger.info(f"📊 Training examples: {len(train_data)} (EXPANDED DATASET)")
    logger.info(f"📊 Validation examples: {len(val_data)}")
    
    # Analyze intent distribution
    intent_counts = {}
    for item in train_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    logger.info("📈 Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        logger.info(f"  {intent}: {count} examples")
    
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
    
    # Data collator con padding specifico
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Training arguments - OTTIMIZZATI
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,  # Ridotto con batch size maggiore
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=50,  # Aumentato
        logging_dir=f"{output_dir}/logs",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",  # Migliore per intent classification
        adam_beta2=0.95,  # Ottimizzato per testi corti
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Aumentato
    )
    
    # Train
    logger.info("🎯 Starting IMPROVED training...")
    training_start = time.time()
    trainer.train()
    training_time = time.time() - training_start
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    total_time = time.time() - start_time
    
    # Save metrics
    final_eval_loss = trainer.state.log_history[-1].get("eval_loss", "unknown")
    
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
        "final_eval_loss": final_eval_loss,
        "quantization": "4-bit",
        "improvements": [
            "8x larger dataset (960 vs 120 examples)",
            "Increased learning rate (5e-4 vs 2e-4)",
            "Larger LoRA parameters (r=32 vs 16)",
            "More epochs (8 vs 5)",
            "Larger batch size (8 vs 4)",
            "Cosine learning rate scheduler",
            "Extended early stopping patience"
        ],
        "intent_distribution": intent_counts,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ IMPROVED Training completed!")
    logger.info(f"⏱️  Training time: {training_time:.1f} seconds")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"📉 Final eval loss: {final_eval_loss}")
    logger.info(f"💾 Model saved to: {output_dir}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Improved Codes-1B Intent Classification Training')
    parser.add_argument("--model_name", type=str, default="seeklhy/codes-1b")
    parser.add_argument("--train_path", type=str, 
                       default="data/intent_classification/expanded/train_expanded.jsonl")
    parser.add_argument("--val_path", type=str, 
                       default="data/intent_classification/expanded/val_expanded.jsonl")
    parser.add_argument("--output_dir", type=str, 
                       default="models/production/codes1b-intent-improved")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    
    args = parser.parse_args()
    
    metrics = train_intent_model(
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
    
    print(f"\n🎯 IMPROVED Training Results:")
    print(f"Training examples: {metrics['training_examples']}")
    print(f"Training time: {metrics['training_time_seconds']:.1f}s")
    print(f"Final eval loss: {metrics.get('final_eval_loss', 'N/A')}")
    print(f"\n🔥 Key Improvements Applied:")
    for improvement in metrics['improvements']:
        print(f"  • {improvement}")

if __name__ == "__main__":
    main()