#!/usr/bin/env python3
"""Test simple SQL generation with minimal parameters"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_minimal():
    """Test with absolutely minimal parameters"""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-text2sql")
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, "outputs/gemma3-text2sql")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Simple question
    prompt = "SQL query for Milano clients:"
    
    print(f"Input: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Most basic generation possible
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    
    print(f"Generated tokens: {generated_tokens.tolist()}")
    print(f"Full result: '{result}'")
    print(f"Generated part: '{tokenizer.decode(generated_tokens, skip_special_tokens=True)}'")

if __name__ == "__main__":
    test_minimal()