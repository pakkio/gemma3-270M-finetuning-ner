#!/usr/bin/env python3
"""Test different inference formats to match training"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_formats():
    """Test different prompt formats"""
    print("Loading Text-to-SQL Gemma3 model...")
    
    tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-text2sql")
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, "outputs/gemma3-text2sql")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    question = "Mostra tutti i clienti di Milano"
    
    # Format 1: Current evaluation format (truncated)
    prompt1 = f"""### Istruzione:
Converti la seguente domanda italiana in una query SQL.

### Domanda:
{question}

### SQL:
"""
    
    # Format 2: Add partial SQL to see if model continues
    prompt2 = f"""### Istruzione:
Converti la seguente domanda italiana in una query SQL.

### Domanda:
{question}

### SQL:
SELECT"""
    
    print("=== Format 1: Standard truncated prompt ===")
    print(f"Prompt: '{prompt1}'")
    inputs1 = tokenizer(prompt1, return_tensors="pt")
    outputs1 = model.generate(
        **inputs1,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated1 = outputs1[0][inputs1['input_ids'].shape[1]:]
    result1 = tokenizer.decode(generated1, skip_special_tokens=True)
    print(f"Generated: '{result1}'")
    print(f"Token IDs: {generated1.tolist()[:10]}...")  # Show first 10
    
    print("\n=== Format 2: With partial SQL start ===")
    print(f"Prompt: '{prompt2}'")
    inputs2 = tokenizer(prompt2, return_tensors="pt")
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated2 = outputs2[0][inputs2['input_ids'].shape[1]:]
    result2 = tokenizer.decode(generated2, skip_special_tokens=True)
    print(f"Generated: '{result2}'")
    print(f"Token IDs: {generated2.tolist()[:10]}...")  # Show first 10
    
    # Check what tokens correspond to SQL keywords
    print("\n=== Token Analysis ===")
    sql_tokens = tokenizer("SELECT * FROM clienti WHERE", return_tensors="pt")
    print(f"SQL tokens: {sql_tokens['input_ids'][0].tolist()}")
    print(f"SQL decoded: '{tokenizer.decode(sql_tokens['input_ids'][0])}'")

if __name__ == "__main__":
    test_formats()