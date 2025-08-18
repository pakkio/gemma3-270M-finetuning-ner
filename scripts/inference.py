#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_first_json(text):
    """Estrae il primo JSON bilanciato dal testo"""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def repair_json(raw_text):
    """Ripara JSON con chiavi duplicate e problemi di formato"""
    # Prima prova a estrarre il primo JSON completo
    json_text = extract_first_json(raw_text)
    if json_text:
        try:
            return json.loads(json_text)
        except:
            pass
    
    # Fallback: estrai tutte le entità con regex e merge
    result = {"people": [], "dates": [], "places": []}
    
    for key in ["people", "dates", "places"]:
        # Trova tutte le occorrenze di ogni chiave
        pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
        matches = re.findall(pattern, raw_text, re.DOTALL)
        
        for match in matches:
            try:
                # Prova a parsare l'array
                array_text = f"[{match}]"
                items = json.loads(array_text)
                if isinstance(items, list):
                    result[key].extend(items)
            except:
                # Fallback: split manuale
                items = [item.strip().strip('"') for item in match.split(',') if item.strip()]
                result[key].extend(items)
    
    # Rimuovi duplicati preservando l'ordine
    for key in result:
        seen = set()
        unique_items = []
        for item in result[key]:
            if item and item not in seen:
                seen.add(item)
                unique_items.append(item)
        result[key] = unique_items
    
    return result

def load_model_and_tokenizer(model_path, base_model_path=None):
    from peft import PeftModel, PeftConfig
    # Load tokenizer from base model if provided, else from model_path
    tokenizer_path = base_model_path if base_model_path is not None else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model_path is not None:
        # Load base model, then apply adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
    else:
        # Try to load as a full model (for backward compatibility)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    return model, tokenizer

def extract_entities(model, tokenizer, document, template_path=None):
    if template_path and os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    else:
        template = """Sei una funzione di estrazione. Non usi strumenti esterni.

REGOLE OUTPUT:
- Rispondi con UN SOLO oggetto JSON valido.
- Chiavi esattamente: "people","dates","places". 
- Ogni valore è un array di stringhe. Se vuoto: [].
- Non ripetere le chiavi. Non aggiungere altro testo.

### Documento
{document}

### Risposta JSON
"""
    
    prompt = template.format(document=document)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Estrai solo la parte dopo "### Risposta JSON"
    if "### Risposta JSON" in generated_text:
        prediction = generated_text.split("### Risposta JSON")[-1].strip()
    else:
        prediction = generated_text[len(prompt):].strip()
    
    # Usa la funzione di repair per gestire JSON malformati
    try:
        parsed_json = repair_json(prediction)
        return parsed_json, prediction
    except Exception as e:
        print(f"Warning: Failed to parse entities: {e}")
        return {"people": [], "dates": [], "places": []}, prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model_path", type=str, help="Path to the base model for tokenizer (optional)")
    parser.add_argument("--document", type=str, help="Document text to extract from")
    parser.add_argument("--file", type=str, help="File containing document text")
    parser.add_argument("--template", type=str, help="Path to inference template file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, base_model_path=args.base_model_path)
    print("Model loaded successfully!")

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        while True:
            document = input("\nEnter document text: ")
            if document.lower() == 'quit':
                break
            
            result, raw_output = extract_entities(model, tokenizer, document, args.template)
            print(f"\nRaw output: {raw_output}")
            if result:
                print(f"Parsed result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                print("Failed to parse JSON")
    
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            document = f.read()
        
        result, raw_output = extract_entities(model, tokenizer, document, args.template)
        print(f"Document: {document[:100]}...")
        print(f"Raw output: {raw_output}")
        if result:
            print(f"Extracted entities: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print("Failed to parse JSON")
    
    elif args.document:
        result, raw_output = extract_entities(model, tokenizer, args.document, args.template)
        print(f"Raw output: {raw_output}")
        if result:
            print(f"Extracted entities: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print("Failed to parse JSON")
    
    else:
        # Test con esempio predefinito
        test_document = """Roma, 15 agosto 2025 — L'assessora alla cultura Maria De Santis ha presentato il nuovo programma museale insieme al direttore Luca Bianchi. L'evento si terrà al Museo di Palazzo Barberini il 15 settembre 2025."""
        
        print(f"Testing with example document: {test_document}")
        result, raw_output = extract_entities(model, tokenizer, test_document, args.template)
        print(f"Raw output: {raw_output}")
        if result:
            print(f"Extracted entities: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print("Failed to parse JSON")

if __name__ == "__main__":
    import os
    main()