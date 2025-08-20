#!/usr/bin/env python3
"""
BERT fine-tuning for Italian hashtag generation.
Fair comparison baseline against Gemma3 hashtagger.
"""
import json
import time
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import argparse
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def load_hashtag_data(file_path):
    """Load hashtag training data for BERT."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append({
                    'input_text': data['document'],
                    'target_text': data['hashtags']
                })
    return examples

def prepare_bert_dataset(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Prepare dataset for BERT seq2seq training."""
    
    input_texts = [f"Genera hashtags per: {example['input_text']}" for example in examples]
    target_texts = [example['target_text'] for example in examples]
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        max_length=max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    targets = tokenizer(
        target_texts,
        max_length=max_target_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    })
    
    return dataset

def evaluate_hashtag_quality(predicted_hashtags, ground_truth_hashtags):
    """Evaluate hashtag generation quality."""
    
    # Normalize hashtags
    def normalize_hashtags(hashtags_string):
        hashtags = hashtags_string.split()
        return set(tag.lower().replace('#', '').strip() for tag in hashtags if tag.startswith('#'))
    
    pred_set = normalize_hashtags(predicted_hashtags)
    true_set = normalize_hashtags(ground_truth_hashtags)
    
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    elif not true_set:
        return 0.0, 0.0, 0.0
    elif not pred_set:
        return 0.0, 0.0, 0.0
    
    # Calculate metrics
    intersection = pred_set.intersection(true_set)
    
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

class HashtagEvaluationCallback(TrainerCallback):
    """Custom callback for hashtag evaluation during training."""
    
    def __init__(self, eval_dataset, tokenizer, eval_examples):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Evaluate hashtag generation quality."""
        model.eval()
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        valid_predictions = 0
        
        with torch.no_grad():
            for i, example in enumerate(self.eval_examples[:20]):  # Sample for speed
                input_text = f"Genera hashtags per: {example['input_text']}"
                
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove input from prediction
                predicted_text = predicted_text.replace(input_text, "").strip()
                
                # Evaluate
                precision, recall, f1 = evaluate_hashtag_quality(
                    predicted_text, example['target_text']
                )
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_predictions += 1
        
        if valid_predictions > 0:
            avg_precision = total_precision / valid_predictions
            avg_recall = total_recall / valid_predictions
            avg_f1 = total_f1 / valid_predictions
            
            print(f"\n📊 Hashtag Eval - P: {avg_precision:.3f}, R: {avg_recall:.3f}, F1: {avg_f1:.3f}")

def train_bert_hashtagger(
    model_id="google/flan-t5-base",  # Good multilingual seq2seq model
    train_path="data/train_hashtagger.jsonl",
    val_path="data/val_hashtagger.jsonl",
    output_dir="outputs/bert-hashtagger",
    epochs=5,
    batch_size=8,
    learning_rate=5e-5,
    max_input_length=512,
    max_target_length=128
):
    """Train BERT-based hashtagger for comparison."""
    
    print("🤖 BERT HASHTAGGER FINE-TUNING")
    print("=" * 50)
    print(f"🎯 Task: Hashtag Generation (Seq2Seq)")
    print(f"🤖 Model: {model_id}")
    print(f"📁 Training data: {train_path}")
    print(f"💾 Output: {output_dir}")
    print(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"📈 Learning rate: {learning_rate}")
    print()
    
    # Setup paths
    train_path = Path(train_path)
    val_path = Path(val_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"🔄 Loading BERT model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Setup special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Model parameters: {model.num_parameters():,}")
    
    # Load and prepare datasets
    print("📊 Loading hashtag training data...")
    train_examples = load_hashtag_data(train_path)
    print(f"✅ Loaded {len(train_examples)} training examples")
    
    val_examples = []
    if val_path.exists():
        val_examples = load_hashtag_data(val_path)
        print(f"✅ Loaded {len(val_examples)} validation examples")
    
    # Prepare datasets
    train_dataset = prepare_bert_dataset(
        train_examples, tokenizer, max_input_length, max_target_length
    )
    val_dataset = prepare_bert_dataset(
        val_examples, tokenizer, max_input_length, max_target_length
    ) if val_examples else None
    
    # Training arguments
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
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        bf16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
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
    print("🔥 Starting BERT hashtagger training...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✅ BERT training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Save final model
        print("💾 Saving BERT hashtagger model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "model_id": model_id,
            "architecture": "seq2seq",
            "task": "hashtag_generation",
            "language": "italian",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_input_length": max_input_length,
            "max_target_length": max_target_length,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples) if val_examples else 0,
            "model_parameters": model.num_parameters()
        }
        
        with open(output_dir / "bert_hashtagger_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"🎉 BERT hashtagger saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ BERT training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for Italian Hashtag Generation')
    parser.add_argument('--model_id', default='google/flan-t5-base', help='Base model ID')
    parser.add_argument('--train_path', default='data/train_hashtagger.jsonl', help='Training data')
    parser.add_argument('--val_path', default='data/val_hashtagger.jsonl', help='Validation data')
    parser.add_argument('--output_dir', default='outputs/bert-hashtagger', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_target_len', type=int, default=128, help='Max target length')
    
    args = parser.parse_args()
    
    success = train_bert_hashtagger(
        model_id=args.model_id,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_input_length=args.max_input_len,
        max_target_length=args.max_target_len
    )
    
    if success:
        print("\n🤖 BERT hashtagger training completed!")
        print("Ready for three-way comparison: Gemma3 vs BERT vs KeyBERT")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)