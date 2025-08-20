#!/usr/bin/env python3
"""Debug why Gemma3 Text-to-SQL model generates empty predictions"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_text2sql_prediction():
    """Test Text-to-SQL model prediction in detail"""
    print("Loading Text-to-SQL Gemma3 model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-text2sql")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/gemma3-text2sql")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Test with the exact same prompt format as training
        prompt = """### Istruzione:
Converti la seguente domanda italiana in una query SQL.

### Domanda:
Mostra tutti i clienti di Milano

### SQL:
"""
        
        print(f"Input prompt:\n{prompt}")
        print("-" * 50)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"Input token IDs shape: {inputs['input_ids'].shape}")
        print(f"Input tokens: {inputs['input_ids'][0].tolist()}")
        
        # Test different generation parameters
        print("\n=== Test 1: Working parameters (temperature=0.1, do_sample=False) ===")
        outputs1 = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Show full output
        full_result1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
        print(f"Full output: '{full_result1}'")
        
        # Show only generated part
        generated_tokens1 = outputs1[0][inputs['input_ids'].shape[1]:]
        generated_result1 = tokenizer.decode(generated_tokens1, skip_special_tokens=True)
        print(f"Generated only: '{generated_result1}'")
        print(f"Generated token count: {len(generated_tokens1)}")
        print(f"Generated token IDs: {generated_tokens1.tolist()}")
        
        # Show raw decode without skip_special_tokens
        raw_result1 = tokenizer.decode(generated_tokens1, skip_special_tokens=False)
        print(f"Raw generated (with special tokens): '{raw_result1}'")
        
        print("\n=== Test 2: Greedy search (num_beams=1) ===")
        try:
            outputs2 = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_tokens2 = outputs2[0][inputs['input_ids'].shape[1]:]
            generated_result2 = tokenizer.decode(generated_tokens2, skip_special_tokens=True)
            print(f"Greedy result: '{generated_result2}'")
        except Exception as e:
            print(f"Greedy search failed: {e}")
        
        print("\n=== Test 3: Simple generate (no special params) ===")
        try:
            outputs3 = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_tokens3 = outputs3[0][inputs['input_ids'].shape[1]:]
            generated_result3 = tokenizer.decode(generated_tokens3, skip_special_tokens=True)
            print(f"Simple result: '{generated_result3}'")
        except Exception as e:
            print(f"Simple generate failed: {e}")
            
        # Check if model learned the end token
        print(f"\n=== Token Analysis ===")
        print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        
        # Check what the model would predict for the training end marker
        end_prompt = prompt + "SELECT * FROM clienti WHERE citta = 'Milano';\n\n### Fine"
        end_inputs = tokenizer(end_prompt, return_tensors="pt")
        print(f"Training end prompt length: {end_inputs['input_ids'].shape[1]} tokens")
        
    except Exception as e:
        print(f"Failed to test model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text2sql_prediction()