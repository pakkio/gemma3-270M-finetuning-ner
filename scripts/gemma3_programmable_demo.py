#!/usr/bin/env python3
"""
Gemma3 Programmable Logic Demonstration
Show flexible, adaptive behavior through different prompts
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def load_gemma3_model():
    """Load Gemma3 model for flexible prompting"""
    
    # Load base model  
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m",
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False
    )
    
    # Load NER adapter (we'll use this for entity extraction)
    model = PeftModel.from_pretrained(model, "outputs/gemma3-comprehensive")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def test_programmable_prompts():
    """Test different programmable prompts on the same text"""
    
    print("🧠 GEMMA3 PROGRAMMABLE LOGIC DEMONSTRATION")
    print("=" * 60)
    
    # Load model
    print("Loading Gemma3 model...")
    model, tokenizer = load_gemma3_model()
    
    # Test document
    document = """
    Milano, 15 marzo 2024 - Mario Draghi, ex presidente della Banca Centrale Europea, 
    ha incontrato oggi Giorgia Meloni a Palazzo Chigi per discutere delle riforme 
    economiche. La riunione, durata tre ore, ha affrontato temi cruciali come 
    l'inflazione, il PNRR e le relazioni con l'Unione Europea. Draghi ha sottolineato 
    l'importanza della stabilità monetaria per la crescita italiana.
    """
    
    # Different programming "modes" through prompts
    prompts = [
        {
            "name": "Standard JSON Mode",
            "prompt": f"""Sei una funzione di estrazione. Non usi strumenti esterni.
Regole output:
- Rispondi SOLO con un oggetto JSON valido.
- Chiavi: "people","dates","places". Valori: array di stringhe.
- Se assente: [].

### Documento
{document}

### Risposta JSON
"""
        },
        
        {
            "name": "Political Analysis Mode", 
            "prompt": f"""Sei un analista politico esperto. Analizza questo documento e estrai:

DOCUMENTO: {document}

ANALISI RICHIESTA:
- Persone: Nome, ruolo politico, importanza (1-10)
- Eventi: Natura, significato politico, impatto potenziale
- Luoghi: Tipo (istituzionale/simbolico), rilevanza
- Temi: Economici, politici, europei

Rispondi in JSON con analisi dettagliata:
"""
        },
        
        {
            "name": "News Hashtag Generator Mode",
            "prompt": f"""Sei un social media manager esperto. Leggi questa notizia:

{document}

Genera hashtags per diversi pubblici:

HASHTAGS FORMALI (stampa economica):
HASHTAGS TRENDING (social media):
HASHTAGS POLITICI (twitter politico):
HASHTAGS LOCALI (media milanesi):

Spiega brevemente la strategia per ogni categoria.
"""
        },
        
        {
            "name": "Risk Assessment Mode",
            "prompt": f"""Sei un analista di rischio. Valuta questo documento:

{document}

RISK ANALYSIS:
- Market Impact: [Alto/Medio/Basso] + spiegazione
- Political Risk: [Alto/Medio/Basso] + fattori
- Key Entities: Nome, ruolo, rischio associato
- Timeline: Eventi temporali critici
- Sentiment: Positivo/Neutro/Negativo per mercati

JSON output con assessment completo:
"""
        },
        
        {
            "name": "Multi-Language Summary Mode",
            "prompt": f"""You are a multilingual analyst. Process this Italian document:

{document}

OUTPUT REQUIRED:
- English summary (2 lines)
- Key entities with English translations
- Hashtags in both Italian and English
- Strategic importance for EU politics

Respond in structured JSON format:
"""
        }
    ]
    
    results = []
    
    for i, prompt_config in enumerate(prompts, 1):
        print(f"\n📋 Testing Mode {i}: {prompt_config['name']}")
        print("-" * 50)
        
        try:
            # Generate response
            inputs = tokenizer(
                prompt_config['prompt'], 
                return_tensors="pt", 
                max_length=800,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt_config['prompt']):].strip()
            
            # Clean up response (remove repetitions, truncate)
            lines = generated_text.split('\n')
            clean_lines = []
            for line in lines:
                if line.strip() and len(clean_lines) < 20:  # Limit output
                    clean_lines.append(line.strip())
            
            clean_response = '\n'.join(clean_lines)
            
            results.append({
                'mode': prompt_config['name'],
                'response': clean_response[:500] + "..." if len(clean_response) > 500 else clean_response
            })
            
            print(f"✅ Generated: {clean_response[:200]}...")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'mode': prompt_config['name'], 
                'response': f"Error: {e}"
            })
    
    # Summary of programmable flexibility
    print(f"\n🎯 PROGRAMMABLE FLEXIBILITY DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"✅ Tested {len(prompts)} different 'programming modes'")
    print("✅ Same model, same document, completely different behaviors")
    print("✅ Each prompt 'programs' the model for specific use cases")
    
    print(f"\n📊 RESULTS SUMMARY:")
    for result in results:
        status = "✅" if not result['response'].startswith("Error") else "❌"
        print(f"{status} {result['mode']}: {'Success' if not result['response'].startswith('Error') else 'Failed'}")
    
    return results

def compare_with_spacy():
    """Show what spaCy would extract (fixed behavior)"""
    
    print(f"\n🔧 COMPARISON: spaCy Fixed Behavior")
    print("-" * 40)
    
    try:
        import spacy
        nlp = spacy.load("outputs/spacy-ner-italian/model-best")
        
        document = """Mario Draghi, ex presidente della Banca Centrale Europea, 
        ha incontrato oggi Giorgia Meloni a Palazzo Chigi per discutere delle riforme 
        economiche."""
        
        doc = nlp(document)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        print("spaCy Output (Always same format):")
        for ent, label in entities:
            print(f"  {ent} -> {label}")
        
        print(f"\n💡 KEY DIFFERENCE:")
        print("🧠 Gemma3: Infinitely programmable through prompts")
        print("🔧 spaCy: Fixed output format, no adaptability")
        
    except Exception as e:
        print(f"spaCy test failed: {e}")

def main():
    """Run the programmable logic demonstration"""
    
    # Test programmable prompts
    results = test_programmable_prompts()
    
    # Compare with fixed behavior
    compare_with_spacy()
    
    print(f"\n🎉 CONCLUSION:")
    print("Gemma3 = Programmable Logic Chip (1980s analogy)")
    print("spaCy = Fixed-Function Chip (1980s analogy)")
    print("Both have their place: Flexibility vs Speed!")

if __name__ == "__main__":
    main()