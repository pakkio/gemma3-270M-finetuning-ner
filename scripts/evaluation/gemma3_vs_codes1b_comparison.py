#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import time
import psutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re

class ModelComparator:
    def __init__(self):
        self.results = {}
    
    def load_model(self, model_path, base_model_id=None):
        """Load a fine-tuned model"""
        print(f"Loading model from {model_path}...")
        
        if base_model_id:
            # Load PEFT model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, 
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            # Load regular model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def generate_response(self, model, tokenizer, prompt, max_length=512):
        """Generate model response"""
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = inputs.to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        
        return response.strip(), generation_time
    
    def extract_entities_from_json(self, response):
        """Extract entities from JSON response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                entities = []
                for key in ["people", "dates", "places"]:
                    if key in data and isinstance(data[key], list):
                        entities.extend([(ent, key) for ent in data[key]])
                return entities
            return []
        except:
            return []
    
    def extract_hashtags_from_response(self, response):
        """Extract hashtags from response"""
        # Find hashtags in response
        hashtags = re.findall(r'#\w+', response)
        # Also try to find comma-separated tags
        if not hashtags:
            # Look for pattern after "Hashtag:" or similar
            lines = response.split('\n')
            for line in lines:
                if 'hashtag' in line.lower() or '#' in line:
                    tags = re.findall(r'#?\w+', line)
                    hashtags.extend([f"#{tag}" if not tag.startswith('#') else tag for tag in tags if len(tag) > 1])
                    break
        return hashtags[:5]  # Limit to 5 hashtags
    
    def evaluate_ner(self, model, tokenizer, test_data):
        """Evaluate NER performance"""
        results = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "total_time": 0,
            "avg_time_per_sample": 0,
            "predictions": [],
            "ground_truth": []
        }
        
        total_time = 0
        all_predictions = []
        all_ground_truth = []
        
        for i, example in enumerate(test_data[:100]):  # Test on first 100 samples
            document = example["document"]
            ground_truth = example.get("output", "{}")
            
            # Create prompt for NER
            prompt = f"""Sei una funzione di estrazione. Non usi strumenti esterni.
Regole output:
- Rispondi SOLO con un oggetto JSON valido.
- Chiavi: "people","dates","places". Valori: array di stringhe.
- Se assente: [].

### Documento
{document}

### Risposta JSON
"""
            
            response, gen_time = self.generate_response(model, tokenizer, prompt)
            total_time += gen_time
            
            # Extract entities
            predicted_entities = self.extract_entities_from_json(response)
            
            try:
                gt_data = json.loads(ground_truth)
                gt_entities = []
                for key in ["people", "dates", "places"]:
                    if key in gt_data and isinstance(gt_data[key], list):
                        gt_entities.extend([(ent, key) for ent in gt_data[key]])
            except:
                gt_entities = []
            
            all_predictions.append(predicted_entities)
            all_ground_truth.append(gt_entities)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1} NER samples...")
        
        # Calculate metrics
        if all_predictions and all_ground_truth:
            # Flatten for sklearn metrics
            pred_flat = [entity for entities in all_predictions for entity in entities]
            gt_flat = [entity for entities in all_ground_truth for entity in entities]
            
            # Calculate precision, recall, F1
            if pred_flat or gt_flat:
                pred_set = set(pred_flat)
                gt_set = set(gt_flat)
                
                if len(pred_set) > 0:
                    precision = len(pred_set.intersection(gt_set)) / len(pred_set)
                else:
                    precision = 0
                
                if len(gt_set) > 0:
                    recall = len(pred_set.intersection(gt_set)) / len(gt_set)
                else:
                    recall = 0
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results.update({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "total_time": total_time,
                    "avg_time_per_sample": total_time / len(all_predictions),
                    "predictions": all_predictions[:10],  # Sample predictions
                    "ground_truth": all_ground_truth[:10]  # Sample ground truth
                })
        
        return results
    
    def evaluate_hashtag(self, model, tokenizer, test_data):
        """Evaluate hashtag generation performance"""
        results = {
            "bleu_score": 0,
            "exact_match": 0,
            "partial_match": 0,
            "total_time": 0,
            "avg_time_per_sample": 0,
            "predictions": [],
            "ground_truth": []
        }
        
        total_time = 0
        exact_matches = 0
        partial_matches = 0
        all_predictions = []
        all_ground_truth = []
        
        for i, example in enumerate(test_data[:100]):
            text = example["text"]
            gt_hashtags = example.get("hashtags", [])
            
            prompt = f"""Data la seguente frase, estrai gli hashtag più appropriati.

Frase: {text}
Hashtag: """
            
            response, gen_time = self.generate_response(model, tokenizer, prompt)
            total_time += gen_time
            
            predicted_hashtags = self.extract_hashtags_from_response(response)
            
            all_predictions.append(predicted_hashtags)
            all_ground_truth.append(gt_hashtags)
            
            # Calculate exact match
            if set(predicted_hashtags) == set(gt_hashtags):
                exact_matches += 1
            
            # Calculate partial match (at least one hashtag in common)
            if set(predicted_hashtags).intersection(set(gt_hashtags)):
                partial_matches += 1
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1} hashtag samples...")
        
        results.update({
            "exact_match": exact_matches / len(all_predictions) if all_predictions else 0,
            "partial_match": partial_matches / len(all_predictions) if all_predictions else 0,
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(all_predictions) if all_predictions else 0,
            "predictions": all_predictions[:10],
            "ground_truth": all_ground_truth[:10]
        })
        
        return results
    
    def get_model_info(self, model):
        """Get model information"""
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
        
        return {
            "trainable_parameters": num_params,
            "model_size_mb": model_size,
            "memory_usage_mb": torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        }
    
    def compare_models(self, gemma3_path, codes1b_path, test_data_ner, test_data_hashtag):
        """Compare Gemma3 1B vs CodeS-1B models"""
        comparison_results = {
            "gemma3_1b": {},
            "codes1b": {},
            "comparison_summary": {}
        }
        
        print("=" * 60)
        print("GEMMA3 1B vs CODES-1B COMPARISON")
        print("=" * 60)
        
        # Test Gemma3 1B
        print("\n🔵 Testing Gemma3 1B...")
        gemma3_model, gemma3_tokenizer = self.load_model(gemma3_path, "google/gemma-3-1b")
        
        print("Evaluating NER...")
        gemma3_ner = self.evaluate_ner(gemma3_model, gemma3_tokenizer, test_data_ner)
        
        print("Evaluating hashtag generation...")
        gemma3_hashtag = self.evaluate_hashtag(gemma3_model, gemma3_tokenizer, test_data_hashtag)
        
        gemma3_info = self.get_model_info(gemma3_model)
        
        comparison_results["gemma3_1b"] = {
            "ner_performance": gemma3_ner,
            "hashtag_performance": gemma3_hashtag,
            "model_info": gemma3_info
        }
        
        # Clean up GPU memory
        del gemma3_model, gemma3_tokenizer
        torch.cuda.empty_cache()
        
        # Test CodeS-1B
        print("\n🟠 Testing CodeS-1B...")
        codes1b_model, codes1b_tokenizer = self.load_model(codes1b_path, "microsoft/CodeS-1b-tokenizer")
        
        print("Evaluating NER...")
        codes1b_ner = self.evaluate_ner(codes1b_model, codes1b_tokenizer, test_data_ner)
        
        print("Evaluating hashtag generation...")
        codes1b_hashtag = self.evaluate_hashtag(codes1b_model, codes1b_tokenizer, test_data_hashtag)
        
        codes1b_info = self.get_model_info(codes1b_model)
        
        comparison_results["codes1b"] = {
            "ner_performance": codes1b_ner,
            "hashtag_performance": codes1b_hashtag, 
            "model_info": codes1b_info
        }
        
        # Generate comparison summary
        comparison_results["comparison_summary"] = {
            "ner_winner": "gemma3_1b" if gemma3_ner["f1"] > codes1b_ner["f1"] else "codes1b",
            "hashtag_winner": "gemma3_1b" if gemma3_hashtag["exact_match"] > codes1b_hashtag["exact_match"] else "codes1b",
            "speed_winner": "gemma3_1b" if gemma3_ner["avg_time_per_sample"] < codes1b_ner["avg_time_per_sample"] else "codes1b",
            "ner_f1_diff": abs(gemma3_ner["f1"] - codes1b_ner["f1"]),
            "hashtag_accuracy_diff": abs(gemma3_hashtag["exact_match"] - codes1b_hashtag["exact_match"]),
            "speed_diff_percent": abs((gemma3_ner["avg_time_per_sample"] - codes1b_ner["avg_time_per_sample"]) / codes1b_ner["avg_time_per_sample"] * 100) if codes1b_ner["avg_time_per_sample"] > 0 else 0
        }
        
        return comparison_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma3_model_path", type=str, default="outputs/gemma3-1b-ner")
    parser.add_argument("--codes1b_model_path", type=str, default="models/production/codes1b-hashtag-best")
    parser.add_argument("--ner_test_data", type=str, default="data/ner/val.jsonl")
    parser.add_argument("--hashtag_test_data", type=str, default="data/hashtag_generation/val.jsonl")
    parser.add_argument("--output_file", type=str, default="results/gemma3_vs_codes1b_comparison.json")
    args = parser.parse_args()
    
    # Load test data
    print("Loading test datasets...")
    ner_data = load_dataset("json", data_files=args.ner_test_data)["train"]
    hashtag_data = load_dataset("json", data_files=args.hashtag_test_data)["train"]
    
    # Run comparison
    comparator = ModelComparator()
    results = comparator.compare_models(
        args.gemma3_model_path,
        args.codes1b_model_path,
        ner_data,
        hashtag_data
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    summary = results["comparison_summary"]
    gemma3_results = results["gemma3_1b"]
    codes1b_results = results["codes1b"]
    
    print(f"\n🎯 NER Task:")
    print(f"  Gemma3 1B F1: {gemma3_results['ner_performance']['f1']:.3f}")
    print(f"  CodeS-1B F1:  {codes1b_results['ner_performance']['f1']:.3f}")
    print(f"  Winner: {summary['ner_winner'].upper()}")
    
    print(f"\n🏷️  Hashtag Task:")
    print(f"  Gemma3 1B Accuracy: {gemma3_results['hashtag_performance']['exact_match']:.3f}")
    print(f"  CodeS-1B Accuracy:  {codes1b_results['hashtag_performance']['exact_match']:.3f}")
    print(f"  Winner: {summary['hashtag_winner'].upper()}")
    
    print(f"\n⚡ Speed:")
    print(f"  Gemma3 1B: {gemma3_results['ner_performance']['avg_time_per_sample']:.3f}s/sample")
    print(f"  CodeS-1B:  {codes1b_results['ner_performance']['avg_time_per_sample']:.3f}s/sample")
    print(f"  Winner: {summary['speed_winner'].upper()}")
    
    print(f"\n📊 Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()