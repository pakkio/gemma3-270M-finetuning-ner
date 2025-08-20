#!/usr/bin/env python3
"""
Test CodeS-1B vs Gemma3-270M on hashtag extraction
Testing if CodeS-1B can match Gemma3-270M's known success case
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

def test_codes_1b_hashtags():
    """Test CodeS-1B (TRAINED) on hashtag extraction"""
    print("🚀 Testing TRAINED CodeS-1B for hashtag extraction...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("outputs/codes-1b-hashtag-generator")
        base_model = AutoModelForCausalLM.from_pretrained("seeklhy/codes-1b", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/codes-1b-hashtag-generator")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def extract_hashtags(text):
            prompt = f"""### Task: Generate relevant hashtags for this Italian text
### Text: {text}
### Hashtags: """
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            start_time = time.time()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            inference_time = time.time() - start_time
            
            # Extract generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean result
            hashtags = result.strip().split('\n')[0].split('###')[0].strip()
            
            return hashtags, inference_time
        
        return extract_hashtags, "CodeS-1B (TRAINED)"
        
    except Exception as e:
        print(f"Error loading CodeS-1B: {e}")
        return None, "CodeS-1B (Failed to Load)"

def test_gemma3_270m_hashtags():
    """Test trained Gemma3-270M on hashtag extraction (known success)"""
    print("✨ Testing Gemma3-270M for hashtag extraction...")
    
    try:
        # Load the successful Gemma3 hashtag model
        tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3-comprehensive")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, "outputs/gemma3-comprehensive")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def extract_hashtags(text):
            prompt = f"""Extract hashtags for: {text}
Hashtags: """
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            start_time = time.time()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            inference_time = time.time() - start_time
            
            # Extract generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean result
            hashtags = result.strip().split('\n')[0].strip()
            
            return hashtags, inference_time
        
        return extract_hashtags, "Gemma3-270M (Trained)"
        
    except Exception as e:
        print(f"Error loading Gemma3-270M: {e}")
        return None, "Gemma3-270M (Failed to Load)"

def run_hashtag_comparison():
    """Compare CodeS-1B vs Gemma3-270M on hashtag extraction"""
    print("=" * 80)
    print("🆚 HASHTAG EXTRACTION COMPARISON")
    print("CodeS-1B (TRAINED) vs Gemma3-270M (TRAINED)")
    print("=" * 80)
    
    # Test cases for hashtag extraction (mix of Italian and English)
    test_cases = [
        "Bologna, 15 maggio 2025 — L'Università di Bologna inaugura il nuovo centro di ricerca sulla vulcanologia",
        "Roma, 10 marzo 2025 — Il festival del cinema italiano presenta 50 film di registi emergenti",
        "Milano, 5 aprile 2025 — La settimana della moda presenta le nuove collezioni primavera-estate",
        "Firenze, 20 gennaio 2025 — Il museo degli Uffizi espone una nuova collezione di arte rinascimentale",
        "Just finished my morning workout! Feeling energized and ready for the day 💪",
        "Beautiful sunset at the beach tonight 🌅 Perfect end to a perfect vacation",
        "Weekend hiking trip in the mountains. Nature therapy is the best! 🏔️",
        "Coffee shop vibes on this rainy Monday morning ☕️ Perfect for productivity",
    ]
    
    # Load models
    models = []
    
    codes_1b_func, codes_1b_name = test_codes_1b_hashtags()
    if codes_1b_func:
        models.append((codes_1b_func, codes_1b_name))
    
    gemma3_func, gemma3_name = test_gemma3_270m_hashtags()
    if gemma3_func:
        models.append((gemma3_func, gemma3_name))
    
    if not models:
        print("❌ No models loaded successfully!")
        return
    
    print(f"\n🧪 Testing {len(test_cases)} hashtag extraction cases...")
    
    # Results tracking
    results = {model_name: {"predictions": [], "avg_time": 0} for _, model_name in models}
    
    # Run tests
    for i, text in enumerate(test_cases):
        print(f"\n--- Test {i+1}/{len(test_cases)} ---")
        print(f"Text: {text[:60]}...")
        
        for model_func, model_name in models:
            try:
                predicted_hashtags, inference_time = model_func(text)
                
                results[model_name]["predictions"].append({
                    "text": text,
                    "predicted": predicted_hashtags,
                    "time": inference_time
                })
                
                results[model_name]["avg_time"] += inference_time
                
                print(f"{model_name}: {predicted_hashtags} ({inference_time:.3f}s)")
                
            except Exception as e:
                print(f"{model_name}: ERROR - {e}")
                results[model_name]["predictions"].append({
                    "text": text,
                    "predicted": f"ERROR: {e}",
                    "time": 0
                })
    
    # Calculate averages
    print("\n" + "=" * 80)
    print("📊 HASHTAG EXTRACTION RESULTS")
    print("=" * 80)
    
    for model_name in results:
        total_tests = len(results[model_name]["predictions"])
        avg_time = results[model_name]["avg_time"] / total_tests if total_tests > 0 else 0
        
        print(f"\n{model_name}:")
        print(f"  Tests completed: {total_tests}")
        print(f"  Avg Time: {avg_time:.3f} seconds")
        
        # Show sample predictions
        print(f"  Sample predictions:")
        for pred in results[model_name]["predictions"][:3]:
            if not pred["predicted"].startswith("ERROR"):
                print(f"    '{pred['text'][:40]}...' → {pred['predicted']}")
    
    # Analysis
    print(f"\n📝 ANALYSIS:")
    print(f"🎯 Gemma3-270M: Trained specifically for hashtag extraction")
    print(f"🚀 CodeS-1B: TRAINED on the same hashtag dataset")
    print(f"💡 This shows if CodeS-1B can compete with/beat Gemma3 when both are trained")
    
    return results

if __name__ == "__main__":
    run_hashtag_comparison()