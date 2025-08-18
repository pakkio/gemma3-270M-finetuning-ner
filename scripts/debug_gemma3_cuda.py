#!/usr/bin/env python3
"""
Debug Gemma3 CUDA compatibility issues
Test different configurations to identify the root cause
"""
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def test_cuda_configuration():
    """Test various CUDA configurations with Gemma3"""
    
    print("🔧 GEMMA3 CUDA COMPATIBILITY DIAGNOSTIC")
    print("=" * 60)
    
    # Basic CUDA info
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    model_id = "google/gemma-3-270m"
    
    # Test configurations
    configurations = [
        {
            "name": "Float32 + CPU",
            "torch_dtype": torch.float32,
            "device_map": None,
            "attn_implementation": "eager"
        },
        {
            "name": "Float16 + Auto Device",
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "attn_implementation": "eager"
        },
        {
            "name": "BFloat16 + Auto Device", 
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "eager"
        },
        {
            "name": "Float16 + CUDA:0",
            "torch_dtype": torch.float16,
            "device_map": {"": 0},
            "attn_implementation": "eager"
        },
        {
            "name": "Float16 + Flash Attention",
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2"
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"🧪 Testing: {config['name']}")
        
        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=config['torch_dtype'],
                device_map=config['device_map'],
                trust_remote_code=True,
                attn_implementation=config['attn_implementation'],
                use_cache=False
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test basic inference
            test_text = "Document: Genova, oggi il porto registra traffico elevato.\nHashtags:"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Move to same device as model if needed
            if config['device_map'] and config['device_map'] != None:
                if isinstance(config['device_map'], dict):
                    device = list(config['device_map'].values())[0]
                    inputs = {k: v.cuda(device) for k, v in inputs.items()}
                elif config['device_map'] == "auto":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up
            del model, tokenizer, inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            results.append((config['name'], "✅ SUCCESS", response[len(test_text):].strip()))
            print(f"   ✅ SUCCESS")
            
        except Exception as e:
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            results.append((config['name'], f"❌ FAILED: {error_msg}", None))
            print(f"   ❌ FAILED: {error_msg}")
            
            # Clean up on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print()
    
    # Results summary
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    for name, status, output in results:
        print(f"{name:25} {status}")
        if output:
            print(f"{'':25} Generated: {output}")
    
    return results

def test_peft_model_cuda():
    """Test PEFT model with CUDA"""
    
    print("\n🔧 TESTING PEFT MODEL CUDA COMPATIBILITY")
    print("=" * 60)
    
    try:
        # Load base model with safest config that worked
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-270m",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            use_cache=False
        )
        
        # Try loading PEFT adapter
        peft_path = "outputs/gemma3-6epoch-hashtagger"
        if os.path.exists(peft_path):
            model = PeftModel.from_pretrained(model, peft_path)
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test inference
            test_text = "Document: Genova, oggi il porto registra traffico elevato.\nHashtags:"
            inputs = tokenizer(test_text, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ PEFT Model CUDA test successful!")
            print(f"Generated: {response[len(test_text):].strip()}")
            
            return True
            
        else:
            print(f"❌ PEFT model not found at {peft_path}")
            return False
            
    except Exception as e:
        print(f"❌ PEFT Model CUDA test failed: {e}")
        traceback.print_exc()
        return False
    
    finally:
        torch.cuda.empty_cache()

def main():
    """Run all CUDA diagnostic tests"""
    
    # Test base model configurations
    base_results = test_cuda_configuration()
    
    # Test PEFT model if base model works
    successful_configs = [r for r in base_results if "SUCCESS" in r[1]]
    if successful_configs:
        print(f"\n✅ Found {len(successful_configs)} working configurations")
        peft_success = test_peft_model_cuda()
        
        if peft_success:
            print("\n🎉 SOLUTION FOUND: PEFT model works with CUDA!")
            print("💡 Recommended configuration:")
            print("   - torch_dtype: torch.float16")
            print("   - device_map: 'auto'") 
            print("   - attn_implementation: 'eager'")
            print("   - use_cache: False")
        else:
            print("\n⚠️  Base model works but PEFT has issues")
    else:
        print("\n❌ No working CUDA configurations found")
        print("💡 Recommendation: Use CPU inference for stability")

if __name__ == "__main__":
    main()