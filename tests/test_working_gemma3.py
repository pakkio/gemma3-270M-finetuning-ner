#!/usr/bin/env python3
"""Test if existing working Gemma3 models still function"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_hashtag_gemma3():
    """Test the working hashtag Gemma3 model"""
    print("Testing hashtag Gemma3 model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-comprehensive")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/gemma3-comprehensive")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        prompt = "Generate hashtags for: La tecnologia italiana cresce rapidamente"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Hashtag model works: {result}")
        return True
    except Exception as e:
        print(f"❌ Hashtag model failed: {e}")
        return False

def test_text2sql_gemma3_simple():
    """Test Text-to-SQL model with the same pattern as hashtag model"""
    print("Testing Text-to-SQL Gemma3 with simple parameters...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-text2sql")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/gemma3-text2sql")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        prompt = """### Istruzione:
Converti la seguente domanda italiana in una query SQL.

### Domanda:
Mostra tutti i clienti di Milano

### SQL:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt")
        # Use EXACT same parameters as working hashtag model
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Text-to-SQL model works: {result}")
        return True
    except Exception as e:
        print(f"❌ Text-to-SQL model failed: {e}")
        return False

if __name__ == "__main__":
    hashtag_works = test_hashtag_gemma3()
    text2sql_works = test_text2sql_gemma3_simple()
    
    print(f"\nResults:")
    print(f"Hashtag Gemma3: {'✅' if hashtag_works else '❌'}")
    print(f"Text-to-SQL Gemma3: {'✅' if text2sql_works else '❌'}")
    
    if hashtag_works and not text2sql_works:
        print("🔧 Issue is specific to Text-to-SQL model/training")
    elif hashtag_works and text2sql_works:
        print("🤔 Both work - issue is in evaluation script parameters")
    else:
        print("🚨 General hardware issue")