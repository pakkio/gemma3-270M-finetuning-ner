#!/usr/bin/env python3
"""
Quick test to verify NER models are still functioning
"""
import json
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def test_spacy_ner():
    """Test spaCy NER model"""
    print("🔍 Testing spaCy NER Model")
    
    try:
        nlp = spacy.load("outputs/spacy-ner-italian/model-best")
        
        text = "Mario Rossi è nato a Roma il 15 gennaio 1980 e lavora per Google."
        doc = nlp(text)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✅ spaCy NER working: {entities}")
        return True
        
    except Exception as e:
        print(f"❌ spaCy NER failed: {e}")
        return False

def test_gemma3_ner():
    """Test Gemma3 NER model"""
    print("🔍 Testing Gemma3 NER Model")
    
    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-270m",
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, "outputs/gemma3-comprehensive")
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test inference
        document = "Mario Rossi è nato a Roma il 15 gennaio 1980 e lavora per Google."
        
        template = f"""Sei una funzione di estrazione. Non usi strumenti esterni.
Regole output:
- Rispondi SOLO con un oggetto JSON valido.
- Chiavi: "people","dates","places". Valori: array di stringhe.
- Se assente: [].

### Documento
{document}

### Risposta JSON
"""
        
        inputs = tokenizer(template, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(template):].strip()
        
        print(f"✅ Gemma3 NER working: {generated_text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemma3 NER failed: {e}")
        return False

def main():
    print("🧪 NER MODELS FUNCTIONALITY TEST")
    print("=" * 40)
    
    spacy_ok = test_spacy_ner()
    gemma3_ok = test_gemma3_ner()
    
    print(f"\n📊 RESULTS:")
    print(f"spaCy NER:  {'✅ Working' if spacy_ok else '❌ Broken'}")
    print(f"Gemma3 NER: {'✅ Working' if gemma3_ok else '❌ Broken'}")
    
    if spacy_ok and gemma3_ok:
        print("\n🎉 Both NER models are functioning correctly!")
    elif spacy_ok or gemma3_ok:
        print("\n⚠️  One NER model is working, one has issues")
    else:
        print("\n❌ Both NER models have issues")

if __name__ == "__main__":
    main()