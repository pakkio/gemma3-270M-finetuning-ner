#!/usr/bin/env python3
"""
Specialized Gemma3 270M fine-tuning for Italian Hashtag Generation
Focused training for automatic hashtag extraction from Italian documents.
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

def load_hashtagger_data(file_path):
    """Load hashtag training data."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append({
                    'document': data['document'],
                    'hashtags': data['hashtags']
                })
    return examples

def create_hashtag_prompt_template():
    """Prompt template optimized for hashtag generation."""
    return """Genera hashtags rilevanti per il seguente testo italiano. Gli hashtags devono essere pertinenti, specifici e utili per la categorizzazione sui social media.

Testo: {document}

Hashtags: {hashtags}"""

def prepare_hashtagger_dataset(examples, tokenizer, max_length=512):
    """Prepare dataset optimized for hashtag generation."""
    template = create_hashtag_prompt_template()
    
    formatted_examples = []
    for example in examples:
        prompt = template.format(
            document=example['document'],
            hashtags=example['hashtags']
        )
        formatted_examples.append(prompt)
    
    # Tokenize with shorter sequence length for hashtag task
    encodings = tokenizer(
        formatted_examples,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create labels for language modeling
    labels = encodings['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Convert to lists for Dataset compatibility
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist(),
        'labels': labels.tolist()
    })
    
    return dataset

def validate_hashtags(hashtag_string):
    """Validate generated hashtags."""
    if not hashtag_string.strip():
        return False, "Empty hashtags"
    
    hashtags = hashtag_string.strip().split()
    
    # Check if all start with #
    invalid_tags = [tag for tag in hashtags if not tag.startswith('#')]
    if invalid_tags:
        return False, f"Invalid hashtags (missing #): {invalid_tags}"
    
    # Check reasonable length
    if len(hashtags) < 3:
        return False, "Too few hashtags (minimum 3 expected)"
    
    if len(hashtags) > 15:
        return False, "Too many hashtags (maximum 15 expected)"
    
    return True, f"Valid: {len(hashtags)} hashtags"

def train_hashtagger_model(
    model_id="google/gemma-3-270m",
    train_path="data/train_hashtagger.jsonl",
    val_path="data/val_hashtagger.jsonl",
    output_dir="outputs/gemma3-hashtagger",
    epochs=6,
    batch_size=8,
    learning_rate=3e-4,
    lora_r=16,
    lora_alpha=32,
    max_seq_len=512
):
    """Train specialized hashtagger model."""
    
    print("🏷️  GEMMA3 270M HASHTAGGER TRAINING")
    print("=" * 50)
    print(f"🎯 Task: Automatic Hashtag Generation")
    print(f"🤖 Model: {model_id}")
    print(f"📁 Training data: {train_path}")
    print(f"💾 Output: {output_dir}")
    print(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"🔧 LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"📏 Max sequence length: {max_seq_len} (optimized for hashtags)")
    print()
    
    # Setup paths
    train_path = Path(train_path)
    val_path = Path(val_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LoRA configuration optimized for hashtag generation
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],  # Include all linear layers
        bias="none",
        use_rslora=False,  # Disable RSLoRA for compatibility
        init_lora_weights=True
    )
    
    # Load model and tokenizer with 8-bit quantization
    print(f"🔄 Loading model: {model_id} with 8-bit quantization for 4GB VRAM compatibility")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Recommended for Gemma3 training stability
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Setup padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print(f"📊 Trainable parameters: {model.print_trainable_parameters()}")
    
    # Ensure gradients are enabled for trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            param.retain_grad()
    
    # Load and prepare datasets
    print("📊 Loading hashtag training data...")
    train_examples = load_hashtagger_data(train_path)
    print(f"✅ Loaded {len(train_examples)} training examples")
    
    val_examples = []
    if val_path.exists():
        val_examples = load_hashtagger_data(val_path)
        print(f"✅ Loaded {len(val_examples)} validation examples")
    
    # Prepare datasets
    train_dataset = prepare_hashtagger_dataset(train_examples, tokenizer, max_seq_len)
    val_dataset = prepare_hashtagger_dataset(val_examples, tokenizer, max_seq_len) if val_examples else None
    
    # Training arguments optimized for hashtag generation
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=5,
        save_steps=50,
        eval_steps=50 if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None
    )
    
    # Data collator - ensure proper tensor handling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8  # Better memory alignment
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
    print("🔥 Starting hashtagger training...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✅ Hashtagger training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Save final model
        print("💾 Saving hashtagger model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "model_id": model_id,
            "task": "hashtag_generation",
            "language": "italian",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "max_seq_len": max_seq_len,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples) if val_examples else 0
        }
        
        with open(output_dir / "hashtagger_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save prompt template
        template = create_hashtag_prompt_template()
        with open(output_dir / "hashtag_template.txt", 'w', encoding='utf-8') as f:
            f.write(template)
        
        print(f"🎉 Hashtagger model saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Hashtagger training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma3 270M for Italian Hashtag Generation')
    parser.add_argument('--model_id', default='google/gemma-3-270m', help='Model ID')
    parser.add_argument('--train_path', default='data/train_hashtagger.jsonl', help='Training data path')
    parser.add_argument('--val_path', default='data/val_hashtagger.jsonl', help='Validation data path')
    parser.add_argument('--output_dir', default='outputs/gemma3-hashtagger', help='Output directory')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    success = train_hashtagger_model(
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
        print("\n🏷️  Hashtagger training completed successfully!")
        print("Ready for automatic Italian hashtag generation!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)