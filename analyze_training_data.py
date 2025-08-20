#!/usr/bin/env python3
"""Analyze what the training data looks like after tokenization"""

import torch
from transformers import AutoTokenizer

def analyze_training_format():
    """Analyze the training data format"""
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    question = "Mostra tutti i clienti di Milano"
    sql = "SELECT * FROM clienti WHERE citta = 'Milano';"
    
    # What we use for training - the full example
    input_prompt = f"""### Istruzione:
Converti la seguente domanda italiana in una query SQL.

### Domanda:
{question}

### SQL:
"""
    
    target = f"{sql}\n\n### Fine"
    
    print("=== Training Data Analysis ===")
    print(f"Input prompt:\n'{input_prompt}'")
    print(f"Target:\n'{target}'")
    print("-" * 50)
    
    # Tokenize parts
    input_tokens = tokenizer(input_prompt, add_special_tokens=False)["input_ids"]
    target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
    
    print(f"Input tokens ({len(input_tokens)}): {input_tokens}")
    print(f"Target tokens ({len(target_tokens)}): {target_tokens}")
    
    # Show what each target token decodes to
    print("\nTarget token breakdown:")
    for i, token_id in enumerate(target_tokens):
        token_text = tokenizer.decode([token_id])
        print(f"  {i}: {token_id} -> '{token_text}'")
    
    # Full sequence
    full_input_ids = [tokenizer.bos_token_id] + input_tokens + target_tokens
    labels = [-100] * (len(input_tokens) + 1) + target_tokens
    
    print(f"\nFull sequence length: {len(full_input_ids)}")
    print(f"Labels (showing trainable part): {[x for x in labels if x != -100]}")
    
    # Show what the model should learn to predict
    print(f"\nModel should learn to predict:")
    trainable_tokens = [x for x in labels if x != -100]
    for i, token_id in enumerate(trainable_tokens[:10]):  # First 10
        token_text = tokenizer.decode([token_id])
        print(f"  {i}: {token_id} -> '{token_text}'")

if __name__ == "__main__":
    analyze_training_format()