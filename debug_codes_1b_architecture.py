#!/usr/bin/env python3
"""Debug CodeS-1B model architecture to find correct LoRA target modules"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_model_architecture():
    """Analyze CodeS-1B architecture"""
    print("Loading CodeS-1B model to analyze architecture...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained("seeklhy/codes-1b")
        tokenizer = AutoTokenizer.from_pretrained("seeklhy/codes-1b")
        
        print(f"Model class: {model.__class__.__name__}")
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"Model config: {model.config}")
        print("\n=== Model Architecture ===")
        
        # Print all named modules to find correct targets
        for name, module in model.named_modules():
            if 'linear' in name.lower() or 'proj' in name.lower() or 'attention' in name.lower():
                print(f"{name}: {module.__class__.__name__}")
        
        print("\n=== All Module Names ===")
        for name, _ in model.named_modules():
            if len(name) > 0:  # Skip empty names
                print(name)
                
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_model_architecture()