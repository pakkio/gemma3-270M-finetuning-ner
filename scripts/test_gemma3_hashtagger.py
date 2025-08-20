#!/usr/bin/env python3
"""
Test Gemma3 hashtagger directly
"""
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_test_data(file_path):
    """Load test data."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append({
                    'document': data['document'],
                    'ground_truth_hashtags': data['hashtags'].split()
                })
    return examples

def create_simple_prompt(document):
    """Create prompt for hashtag generation."""
    return f"Document: {document}\nHashtags:"

def generate_hashtags(model, tokenizer, text):
    """Generate hashtags using the model."""
    prompt = create_simple_prompt(text)
    
    start_time = time.time()
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()
    
    generation_time = time.time() - start_time
    
    # Extract hashtags
    hashtag_line = generated_text.split('\n')[0].strip()
    hashtags = [tag.strip() for tag in hashtag_line.split() if tag.startswith('#')]
    
    return hashtags, generation_time

def evaluate_hashtags(predicted, ground_truth):
    """Evaluate predicted hashtags against ground truth."""
    # Normalize hashtags (remove #, lowercase)
    pred_normalized = set(tag.lower().replace('#', '') for tag in predicted)
    true_normalized = set(tag.lower().replace('#', '') for tag in ground_truth)
    
    if not true_normalized and not pred_normalized:
        return 1.0, 1.0, 1.0
    elif not true_normalized:
        return 0.0, 0.0, 0.0
    elif not pred_normalized:
        return 0.0, 0.0, 0.0
    
    # Calculate metrics
    intersection = pred_normalized.intersection(true_normalized)
    
    precision = len(intersection) / len(pred_normalized) if pred_normalized else 0.0
    recall = len(intersection) / len(true_normalized) if true_normalized else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def main():
    print("🔧 GEMMA3 HASHTAGGER TEST")
    print("=" * 50)
    
    model_path = "outputs/simple-gemma3-hashtagger"
    
    # Load model
    print("🔄 Loading Gemma3 hashtagger...")
    try:
        base_model_path = "google/gemma-3-270m"
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Gemma3 hashtagger loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load Gemma3 hashtagger: {e}")
        return
    
    # Load test data
    test_data_path = "data/val_hashtagger.jsonl"
    examples = load_test_data(test_data_path)
    print(f"📊 Loaded {len(examples)} test examples")
    
    # Test a few examples
    print("\n🔄 Testing examples...")
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_time = 0.0
    valid_examples = 0
    
    for i, example in enumerate(examples[:10], 1):  # Test first 10 examples
        print(f"\rExample {i}/10", end="", flush=True)
        
        text = example['document']
        ground_truth = example['ground_truth_hashtags']
        
        try:
            hashtags, gen_time = generate_hashtags(model, tokenizer, text)
            precision, recall, f1 = evaluate_hashtags(hashtags, ground_truth)
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_time += gen_time
            valid_examples += 1
            
        except Exception as e:
            print(f"\n❌ Error processing example {i}: {e}")
    
    if valid_examples > 0:
        # Calculate averages
        avg_precision = total_precision / valid_examples
        avg_recall = total_recall / valid_examples
        avg_f1 = total_f1 / valid_examples
        avg_time = total_time / valid_examples
        examples_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
        
        print(f"\n\n🏆 GEMMA3 RESULTS (on {valid_examples} examples)")
        print("=" * 50)
        print(f"Precision: {avg_precision:.3f}")
        print(f"Recall: {avg_recall:.3f}")
        print(f"F1 Score: {avg_f1:.3f}")
        print(f"Avg time per example: {avg_time:.3f}s")
        print(f"Examples per second: {examples_per_sec:.1f}")
    else:
        print("\n❌ No valid examples processed")

if __name__ == "__main__":
    main()