#!/usr/bin/env python3
"""Test the Gemma3 Intent Classifier"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_intent_classifier():
    """Test the trained intent classifier"""
    print("Loading Gemma3 Intent Classifier...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-intent-classifier")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/gemma3-intent-classifier")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test cases with expected intents
        test_cases = [
            ("Non riesco ad accedere al mio account", "account_access"),
            ("Voglio cancellare il mio ordine", "order_cancellation"),
            ("Dove è il mio pacco?", "order_tracking"),
            ("Voglio restituire questo prodotto", "return_refund"),
            ("La carta non funziona", "payment_issues"),
            ("Quanto costa la spedizione?", "shipping_info"),
            ("È disponibile in magazzino?", "product_availability"),
            ("Caratteristiche del prodotto", "product_info"),
            ("Ci sono sconti?", "promotions_discounts"),
            ("Il sito non funziona", "technical_support"),
            ("Voglio cambiare la mia email", "account_management"),
            ("Buongiorno, ho una domanda", "general_inquiry")
        ]
        
        print("\n=== Testing Intent Classification ===")
        for text, expected in test_cases:
            prompt = f"""### Testo:
{text}

### Intent:
"""
            
            inputs = tokenizer(prompt, return_tensors="pt")
            print(f"\nInput: '{text}'")
            print(f"Expected: {expected}")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Extract generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean result
            intent = result.strip().split('\n')[0].strip()
            if '###' in intent:
                intent = intent.split('###')[0].strip()
            
            print(f"Generated: '{result.strip()}'")
            print(f"Cleaned Intent: '{intent}'")
            print(f"Match: {'✓' if intent == expected else '✗'}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intent_classifier()