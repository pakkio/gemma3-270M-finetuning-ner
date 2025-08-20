#!/usr/bin/env python3
"""
Simplified Gemma3 hashtagger training script
Addressing gradient computation issues with a minimal approach.
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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

def create_simple_prompt(document, hashtags=None):
    """Simple prompt for hashtag generation."""
    if hashtags:
        return f"Document: {document}\nHashtags: {hashtags}"
    else:
        return f"Document: {document}\nHashtags:"

def prepare_simple_dataset(examples, tokenizer, max_length=256):
    """Prepare dataset with simpler approach."""
    
    # Create training texts
    texts = []
    for example in examples:
        text = create_simple_prompt(example['document'], example['hashtags'])
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
    })
    
    # Add labels (same as input_ids for language modeling)
    def add_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples
    
    dataset = dataset.map(add_labels, batched=True)
    
    return dataset

def simple_gemma_training(
    model_id="google/gemma-3-270m",
    train_path="data/train_hashtagger.jsonl",
    output_dir="outputs/simple-gemma3-hashtagger",
    epochs=2,
    batch_size=2,
    learning_rate=1e-4
):
    """Simplified Gemma3 hashtagger training."""
    
    print("🔧 SIMPLIFIED GEMMA3 HASHTAGGER TRAINING")
    print("=" * 50)
    print(f"🤖 Model: {model_id}")
    print(f"📁 Training data: {train_path}")
    print(f"💾 Output: {output_dir}")
    print(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"📈 Learning rate: {learning_rate}")
    print()
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with eager attention
    print(f"🔄 Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False  # Disable cache for training
    )
    
    # Prepare model for k-bit training (may help with gradients)
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Simple LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Smaller r
        lora_alpha=16,  # Smaller alpha
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Only attention projections
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("📊 Loading training data...")
    train_examples = load_hashtagger_data(train_path)
    print(f"✅ Loaded {len(train_examples)} training examples")
    
    # Prepare dataset
    train_dataset = prepare_simple_dataset(train_examples, tokenizer, max_length=256)
    
    # Simple training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        warmup_steps=10,
        bf16=False,  # Use fp16 or fp32
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Disable for simplicity
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None
    )
    
    # Simple data collator
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
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("🔥 Starting simplified training...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"🎉 Model saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple Gemma3 Hashtagger Training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', default='outputs/simple-gemma3-hashtagger', help='Output directory')
    
    args = parser.parse_args()
    
    success = simple_gemma_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)