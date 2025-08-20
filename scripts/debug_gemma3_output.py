#!/usr/bin/env python3
"""
Debug Gemma3 hashtag generation to understand low F1 score
"""
import json
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
                    'ground_truth_hashtags': data['hashtags']
                })
    return examples

def create_simple_prompt(document):
    """Create prompt for hashtag generation."""
    return f"Document: {document}\nHashtags:"

def debug_generation():
    print("🔍 DEBUG: Gemma3 Hashtag Generation")
    print("=" * 50)
    
    model_path = "outputs/simple-gemma3-hashtagger"
    
    # Load model
    print("🔄 Loading Gemma3 model...")
    try:
        base_model_path = "google/gemma-3-270m"
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map=None,
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
        
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test examples
    examples = load_test_data("data/val_hashtagger.jsonl")
    print(f"📊 Loaded {len(examples)} examples")
    
    print("\n🔍 Analyzing first 5 examples:")
    print("=" * 80)
    
    for i, example in enumerate(examples[:5]):
        print(f"\n📄 EXAMPLE {i+1}:")
        print(f"Document: {example['document'][:100]}...")
        print(f"Ground Truth: {example['ground_truth_hashtags']}")
        
        # Generate with current approach
        prompt = create_simple_prompt(example['document'])
        print(f"\nPrompt used: {prompt[:150]}...")
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            # Try different generation strategies
            
            # Strategy 1: Current approach
            outputs1 = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                top_p=0.9,
                top_k=50
            )
            
            response1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
            generated1 = response1[len(prompt):].strip()
            
            # Strategy 2: Greedy decoding
            outputs2 = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
            generated2 = response2[len(prompt):].strip()
            
            # Strategy 3: Lower temperature
            outputs3 = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.8
            )
            
            response3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)
            generated3 = response3[len(prompt):].strip()
        
        print(f"\n🤖 Generated (Current): {generated1}")
        print(f"🤖 Generated (Greedy):  {generated2}")
        print(f"🤖 Generated (Low T):   {generated3}")
        
        # Extract hashtags from each
        hashtags1 = [tag.strip() for tag in generated1.split() if tag.startswith('#')]
        hashtags2 = [tag.strip() for tag in generated2.split() if tag.startswith('#')]
        hashtags3 = [tag.strip() for tag in generated3.split() if tag.startswith('#')]
        
        print(f"📌 Hashtags (Current): {hashtags1}")
        print(f"📌 Hashtags (Greedy):  {hashtags2}")  
        print(f"📌 Hashtags (Low T):   {hashtags3}")
        
        print("-" * 80)

def analyze_training_format():
    """Check how training data was formatted."""
    print("\n🔍 TRAINING DATA FORMAT ANALYSIS:")
    print("=" * 50)
    
    examples = load_test_data("data/train_hashtagger.jsonl")
    
    for i, example in enumerate(examples[:3]):
        prompt = create_simple_prompt(example['document'])
        expected = example['ground_truth_hashtags']
        
        print(f"\nTraining Example {i+1}:")
        print(f"Input: {prompt}")
        print(f"Expected Output: {expected}")
        print(f"Full Training Text: {prompt} {expected}")

if __name__ == "__main__":
    analyze_training_format()
    debug_generation()