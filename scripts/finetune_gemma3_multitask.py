#!/usr/bin/env python3
"""
Multi-task fine-tuning of Gemma3 270M for Italian NER + Document Categorization + Hashtag Generation
Extends the original NER model with hierarchical categorization capabilities.
"""
import json
import time
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import random

def load_multitask_data(file_path):
    """Load multi-task training data with NER + categories + hashtags."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append({
                    'document': data['document'],
                    'output': data['output']
                })
    return examples

def create_enhanced_prompt_template():
    """Enhanced prompt template for multi-task learning."""
    return """Analizza il seguente testo italiano ed estrai:
1. Entità nominate (persone, date, luoghi)
2. Categorie tematiche gerarchiche
3. Hashtags rilevanti

Testo: {document}

Output JSON: {output}"""

def prepare_multitask_dataset(examples, tokenizer, max_length=1024):
    """Prepare dataset for multi-task training."""
    template = create_enhanced_prompt_template()
    
    formatted_examples = []
    for example in examples:
        prompt = template.format(
            document=example['document'],
            output=example['output']
        )
        formatted_examples.append(prompt)
    
    # Tokenize
    encodings = tokenizer(
        formatted_examples,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create labels (shift input_ids by 1 for causal LM)
    labels = encodings['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset

def setup_multitask_model(model_id, lora_config):
    """Setup Gemma3 model for multi-task learning."""
    print(f"🔄 Loading model: {model_id}")
    
    # Load model and tokenizer with 8-bit quantization
    print(f"Loading model {model_id} with 8-bit quantization for 4GB VRAM compatibility...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Recommended for Gemma3 training stability
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    print(f"📊 Trainable parameters: {model.print_trainable_parameters()}")
    
    return model, tokenizer

def evaluate_multitask_output(output_text):
    """Evaluate multi-task output quality."""
    try:
        # Extract JSON from output
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return None, "No JSON found"
        
        json_str = output_text[start_idx:end_idx]
        result = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['people', 'dates', 'places', 'categories', 'hashtags']
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            return result, f"Missing fields: {missing_fields}"
        
        # Validate data types
        for field in required_fields:
            if not isinstance(result[field], list):
                return result, f"Field {field} should be a list"
        
        # Validate hashtags format
        hashtags = result.get('hashtags', [])
        invalid_hashtags = [h for h in hashtags if not h.startswith('#')]
        if invalid_hashtags:
            return result, f"Invalid hashtags (missing #): {invalid_hashtags}"
        
        return result, "Valid"
        
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, f"Error: {e}"

def train_multitask_model(
    model_id="google/gemma-3-270m",
    train_path="data/train_multitask.jsonl",
    val_path="data/val_multitask.jsonl", 
    output_dir="outputs/gemma3-multitask",
    epochs=8,
    batch_size=4,
    learning_rate=2e-4,
    lora_r=32,
    lora_alpha=64,
    max_seq_len=1024
):
    """Train multi-task Gemma3 model."""
    
    print("🚀 GEMMA3 270M MULTI-TASK TRAINING")
    print("=" * 60)
    print(f"📋 Tasks: NER + Categorization + Hashtag Generation")
    print(f"🤖 Model: {model_id}")
    print(f"📁 Training data: {train_path}")
    print(f"📁 Validation data: {val_path}")
    print(f"💾 Output: {output_dir}")
    print(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"🔧 LoRA r={lora_r}, alpha={lora_alpha}")
    print()
    
    # Setup paths
    train_path = Path(train_path)
    val_path = Path(val_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LoRA configuration for multi-task learning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_multitask_model(model_id, lora_config)
    
    # Load and prepare datasets
    print("📊 Loading training data...")
    train_examples = load_multitask_data(train_path)
    print(f"✅ Loaded {len(train_examples)} training examples")
    
    val_examples = []
    if val_path.exists():
        val_examples = load_multitask_data(val_path)
        print(f"✅ Loaded {len(val_examples)} validation examples")
    
    # Prepare datasets
    train_dataset = prepare_multitask_dataset(train_examples, tokenizer, max_seq_len)
    val_dataset = prepare_multitask_dataset(val_examples, tokenizer, max_seq_len) if val_examples else None
    
    # Training arguments optimized for multi-task learning
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    print("🔥 Starting multi-task training...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Save final model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "model_id": model_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "max_seq_len": max_seq_len,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples) if val_examples else 0,
            "tasks": ["ner", "categorization", "hashtag_generation"]
        }
        
        with open(output_dir / "training_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save prompt template
        template = create_enhanced_prompt_template()
        with open(output_dir / "multitask_template.txt", 'w', encoding='utf-8') as f:
            f.write(template)
        
        print(f"🎉 Multi-task model saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma3 270M for multi-task Italian NER + Categorization + Hashtags')
    parser.add_argument('--model_id', default='google/gemma-3-270m', help='Model ID')
    parser.add_argument('--train_path', default='data/train_multitask.jsonl', help='Training data path')
    parser.add_argument('--val_path', default='data/val_multitask.jsonl', help='Validation data path')
    parser.add_argument('--output_dir', default='outputs/gemma3-multitask', help='Output directory')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha parameter')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    success = train_multitask_model(
        model_id=args.model_id,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len
    )
    
    if success:
        print("\n🏆 Multi-task training completed successfully!")
        print("Ready for NER + Categorization + Hashtag Generation!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)