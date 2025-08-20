#!/usr/bin/env python3
"""
Test Gemma3 performance with CUDA acceleration
Compare CPU vs GPU inference speed
"""
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_gemma3_speed():
    """Test Gemma3 inference speed with CUDA vs CPU"""
    
    print("🚀 GEMMA3 CUDA vs CPU SPEED TEST")
    print("=" * 50)
    
    # Load test data
    test_file = "data/val_hashtagger.jsonl"
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip() and i < 10:  # Test with 10 examples
                data = json.loads(line)
                examples.append(data)
    
    print(f"Testing with {len(examples)} examples")
    
    # Test CUDA configuration
    print("\n🔥 Testing CUDA (GPU) Configuration")
    cuda_time = test_cuda_inference(examples)
    
    # Test CPU configuration  
    print("\n🐌 Testing CPU Configuration")
    cpu_time = test_cpu_inference(examples)
    
    # Results
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"GPU Time: {cuda_time:.2f} seconds ({cuda_time/len(examples):.3f}s per example)")
    print(f"CPU Time: {cpu_time:.2f} seconds ({cpu_time/len(examples):.3f}s per example)")
    print(f"Speedup: {cpu_time/cuda_time:.1f}x faster with GPU")
    
    return cuda_time, cpu_time

def test_cuda_inference(examples):
    """Test inference with CUDA acceleration"""
    
    # Load model with CUDA config
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False
    )
    
    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, "outputs/gemma3-6epoch-hashtagger")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    start_time = time.time()
    
    for example in examples:
        text = example['document']
        prompt = f"Document: {text}\\nHashtags:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        
        # Inputs automatically go to GPU with device_map="auto"
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Just decode, don't process (for speed test)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    total_time = time.time() - start_time
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return total_time

def test_cpu_inference(examples):
    """Test inference with CPU"""
    
    # Load model with CPU config
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m",
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False
    )
    
    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, "outputs/gemma3-6epoch-hashtagger")
    model.eval()
    model = model.cpu()
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    start_time = time.time()
    
    for example in examples:
        text = example['document']
        prompt = f"Document: {text}\\nHashtags:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        
        # Force CPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Just decode, don't process (for speed test)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    total_time = time.time() - start_time
    
    # Cleanup
    del model, tokenizer
    
    return total_time

if __name__ == "__main__":
    test_gemma3_speed()