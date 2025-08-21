#!/usr/bin/env python3
"""Quick test of Gemma3 model basic functionality"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_base_model():
    """Test if base model works"""
    print("Testing base Gemma3 model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer("Test: ", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Base model works: {result}")
        return True
    except Exception as e:
        print(f"Base model failed: {e}")
        return False

def test_finetuned_model():
    """Test if fine-tuned model works"""
    print("Testing fine-tuned Gemma3 model...")
    try:
        from peft import PeftModel
        
        tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-text2sql")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/gemma3-text2sql")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer("Test: ", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Fine-tuned model works: {result}")
        return True
    except Exception as e:
        print(f"Fine-tuned model failed: {e}")
        return False

if __name__ == "__main__":
    base_works = test_base_model()
    ft_works = test_finetuned_model()
    
    print(f"\nResults:")
    print(f"Base model: {'✅' if base_works else '❌'}")
    print(f"Fine-tuned model: {'✅' if ft_works else '❌'}")
    
    if base_works and not ft_works:
        print("🔧 Recommendation: Retrain - fine-tuning corrupted the model")
    elif not base_works:
        print("🚨 Recommendation: Hardware/driver issue - base model broken") 
    else:
        print("🤔 Both work - issue might be in evaluation script")