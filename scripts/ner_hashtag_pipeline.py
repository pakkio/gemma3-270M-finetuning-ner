#!/usr/bin/env python3
"""
Combined NER + Hashtag Pipeline
Uses the proven NER model and specialized hashtagger in sequence.
"""
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

class NERHashtagPipeline:
    """Pipeline combining NER extraction with hashtag generation."""
    
    def __init__(self, ner_model_path, hashtagger_model_path):
        self.ner_model_path = Path(ner_model_path)
        self.hashtagger_model_path = Path(hashtagger_model_path)
        
        # Load models
        print("🔄 Loading NER model...")
        self.ner_model, self.ner_tokenizer = self._load_model(ner_model_path, "ner")
        
        print("🔄 Loading Hashtagger model...")
        self.hashtagger_model, self.hashtagger_tokenizer = self._load_model(hashtagger_model_path, "hashtagger")
        
        # Load templates
        self.ner_template = self._load_template(ner_model_path, "inference_template.txt")
        self.hashtag_template = self._load_template(hashtagger_model_path, "hashtag_template.txt")
        
        print("✅ Pipeline ready!")
    
    def _load_model(self, model_path, task_name):
        """Load PEFT model and tokenizer."""
        try:
            base_model_path = "google/gemma-3-270m"
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load PEFT adapter
            model = PeftModel.from_pretrained(model, model_path)
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ {task_name.upper()} model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load {task_name} model: {e}")
            raise
    
    def _load_template(self, model_path, template_file):
        """Load prompt template."""
        template_path = Path(model_path) / template_file
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        
        # Fallback templates
        if "hashtag" in template_file:
            return """Genera hashtags rilevanti per il seguente testo italiano. Gli hashtags devono essere pertinenti, specifici e utili per la categorizzazione sui social media.

Testo: {document}

Hashtags: """
        else:
            return """Estrai le entità nominate dal seguente testo italiano e restituisci il risultato in formato JSON con le chiavi: people, dates, places.

Testo: {document}

JSON: """
    
    def extract_entities(self, text):
        """Extract named entities using the NER model."""
        prompt = self.ner_template.format(document=text)
        
        inputs = self.ner_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.ner_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.ner_tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.ner_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        # Extract JSON
        try:
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = generated_text[start_idx:end_idx]
                entities = json.loads(json_str)
                return entities, None
            else:
                return None, "No JSON found in NER output"
        
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"
    
    def generate_hashtags(self, text):
        """Generate hashtags using the specialized hashtagger."""
        prompt = self.hashtag_template.format(document=text)
        
        inputs = self.hashtagger_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.hashtagger_model.generate(
                **inputs,
                max_new_tokens=50,  # Shorter for hashtags
                temperature=0.3,    # Bit more creative for hashtags
                do_sample=True,
                pad_token_id=self.hashtagger_tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode response
        response = self.hashtagger_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        # Extract hashtags
        hashtag_line = generated_text.split('\n')[0].strip()
        hashtags = [tag.strip() for tag in hashtag_line.split() if tag.startswith('#')]
        
        return hashtags
    
    def process_document(self, text):
        """Process document through both NER and hashtag pipeline."""
        print(f"📄 Processing document...")
        start_time = time.time()
        
        # Step 1: Extract entities
        entities, ner_error = self.extract_entities(text)
        ner_time = time.time() - start_time
        
        if ner_error:
            print(f"⚠️  NER error: {ner_error}")
            entities = {"people": [], "dates": [], "places": []}
        
        # Step 2: Generate hashtags
        hashtag_start = time.time()
        hashtags = self.generate_hashtags(text)
        hashtag_time = time.time() - hashtag_start
        
        total_time = time.time() - start_time
        
        # Combine results
        result = {
            "text": text,
            "entities": entities,
            "hashtags": hashtags,
            "timing": {
                "ner_time_seconds": ner_time,
                "hashtag_time_seconds": hashtag_time,
                "total_time_seconds": total_time
            }
        }
        
        print(f"✅ Processing complete in {total_time:.2f}s (NER: {ner_time:.2f}s, Hashtags: {hashtag_time:.2f}s)")
        return result

def main():
    parser = argparse.ArgumentParser(description='NER + Hashtag Pipeline')
    parser.add_argument('--ner-model', default='outputs/gemma3-comprehensive',
                       help='Path to NER model')
    parser.add_argument('--hashtagger-model', default='outputs/gemma3-hashtagger', 
                       help='Path to hashtagger model')
    parser.add_argument('--text', help='Text to process')
    parser.add_argument('--file', help='File containing text to process')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not Path(args.ner_model).exists():
        print(f"❌ NER model not found: {args.ner_model}")
        return False
    
    if not Path(args.hashtagger_model).exists():
        print(f"❌ Hashtagger model not found: {args.hashtagger_model}")
        return False
    
    # Initialize pipeline
    try:
        pipeline = NERHashtagPipeline(args.ner_model, args.hashtagger_model)
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return False
    
    results = []
    
    if args.interactive:
        print("\n🎯 NER + HASHTAG PIPELINE - Interactive Mode")
        print("Enter Italian text (empty line to quit):")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n📝 Text: ").strip()
                if not text:
                    break
                
                result = pipeline.process_document(text)
                
                print(f"\n👥 Entities:")
                print(f"   People: {result['entities'].get('people', [])}")
                print(f"   Dates: {result['entities'].get('dates', [])}")
                print(f"   Places: {result['entities'].get('places', [])}")
                print(f"\n🏷️  Hashtags: {' '.join(result['hashtags'])}")
                
                results.append(result)
                
            except KeyboardInterrupt:
                break
    
    elif args.text:
        result = pipeline.process_document(args.text)
        results.append(result)
        
        print("\n📊 RESULTS")
        print("=" * 50)
        print(f"👥 Entities:")
        print(f"   People: {result['entities'].get('people', [])}")
        print(f"   Dates: {result['entities'].get('dates', [])}")
        print(f"   Places: {result['entities'].get('places', [])}")
        print(f"\n🏷️  Hashtags: {' '.join(result['hashtags'])}")
    
    elif args.file:
        text = Path(args.file).read_text(encoding='utf-8')
        result = pipeline.process_document(text)
        results.append(result)
        
        print(f"\n📊 RESULTS for {args.file}")
        print("=" * 50)
        print(f"👥 Entities:")
        print(f"   People: {result['entities'].get('people', [])}")
        print(f"   Dates: {result['entities'].get('dates', [])}")
        print(f"   Places: {result['entities'].get('places', [])}")
        print(f"\n🏷️  Hashtags: {' '.join(result['hashtags'])}")
    
    # Save results if requested
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)