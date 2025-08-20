#!/usr/bin/env python3
"""
Fine-tune DistilBERT for Italian Intent Classification
Proper classification approach - much better than generative models for this task
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(train_path: str, val_path: str):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {train_path}")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    
    logger.info(f"Loading validation data from {val_path}")
    val_dataset = load_dataset("json", data_files=val_path, split="train")
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def create_label_mappings(train_dataset):
    """Create label to ID mappings"""
    unique_intents = sorted(set(train_dataset['intent']))
    label2id = {label: i for i, label in enumerate(unique_intents)}
    id2label = {i: label for label, i in label2id.items()}
    
    logger.info(f"Found {len(unique_intents)} unique intents:")
    for i, intent in enumerate(unique_intents):
        logger.info(f"  {i}: {intent}")
    
    return label2id, id2label, unique_intents

def preprocess_function(examples, tokenizer, label2id, max_length=128):
    """Preprocess examples for classification"""
    # Tokenize texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # Will pad in data collator
        max_length=max_length,
        return_tensors=None
    )
    
    # Convert intent labels to IDs
    tokenized["labels"] = [label2id[intent] for intent in examples["intent"]]
    
    return tokenized

def compute_metrics(eval_pred):
    """Compute accuracy and F1 scores"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Intent Classification")
    parser.add_argument("--model_name", type=str, default="distilbert-base-multilingual-cased", 
                       help="DistilBERT model to fine-tune")
    parser.add_argument("--train_path", type=str, default="data/train_intent_classification.jsonl",
                       help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/val_intent_classification.jsonl",
                       help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="outputs/distilbert-intent-classifier",
                       help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    
    args = parser.parse_args()
    
    logger.info("Starting DistilBERT Intent Classification fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Load data
    train_dataset, val_dataset = load_data(args.train_path, args.val_path)
    
    # Create label mappings
    label2id, id2label, unique_intents = create_label_mappings(train_dataset)
    
    # Setup tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(unique_intents),
        label2id=label2id,
        id2label=id2label
    )
    
    logger.info(f"Model loaded with {len(unique_intents)} output classes")
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=True,
        report_to="none",
        save_total_limit=2,
        weight_decay=0.01,
        lr_scheduler_type="linear"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Test on validation set for detailed results
    logger.info("Generating detailed evaluation...")
    predictions = trainer.predict(val_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Create classification report
    intent_names = [id2label[i] for i in range(len(unique_intents))]
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=intent_names,
        output_dict=True
    )
    
    # Save inference template
    inference_template = f"""### DistilBERT Intent Classification Usage:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model
model_path = "{args.output_dir}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Classify intent
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
    
    intent = model.config.id2label[predicted_class_id]
    return intent, confidence

# Get all probabilities
def get_intent_probabilities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    results = {{}}
    for i, prob in enumerate(probabilities[0]):
        intent = model.config.id2label[i]
        results[intent] = prob.item()
    
    return results

# Available intents: {', '.join(unique_intents)}

# Example usage:
# intent, confidence = classify_intent("Non riesco ad accedere al mio account")
# print(f"Intent: {{intent}} (Confidence: {{confidence:.3f}})")

# Get all probabilities:
# probs = get_intent_probabilities("Voglio cancellare il mio ordine")
# for intent, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
#     print(f"{{intent}}: {{prob:.3f}}")
"""
    
    with open(f"{args.output_dir}/inference_template.py", "w") as f:
        f.write(inference_template)
    
    # Save training configuration and results
    config = {
        "model_name": args.model_name,
        "training_args": training_args.to_dict(),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "unique_intents": unique_intents,
        "num_intents": len(unique_intents),
        "label2id": label2id,
        "id2label": id2label,
        "training_time_seconds": training_time,
        "final_train_loss": train_result.training_loss,
        "eval_results": eval_result,
        "classification_report": report,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{args.output_dir}/training_results.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Print results
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    logger.info(f"Validation accuracy: {eval_result['eval_accuracy']:.3f}")
    logger.info(f"Validation F1: {eval_result['eval_f1']:.3f}")
    logger.info(f"Number of intents: {len(unique_intents)}")
    
    logger.info("\n=== PER-INTENT PERFORMANCE ===")
    for intent in unique_intents:
        if intent in report:
            logger.info(f"{intent}: F1={report[intent]['f1-score']:.3f}, Precision={report[intent]['precision']:.3f}, Recall={report[intent]['recall']:.3f}")

if __name__ == "__main__":
    main()