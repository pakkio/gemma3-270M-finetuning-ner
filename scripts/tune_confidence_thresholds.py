#!/usr/bin/env python3
"""
Tune model confidence thresholds to improve recall while maintaining reasonable precision.
This script helps address the ultra-conservative model behavior causing 1.0 precision with terrible recall.
"""
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import re
from collections import defaultdict
import numpy as np

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tokenizer

def load_test_data(file_path):
    """Load test data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_entities_with_scores(model_output, confidence_threshold=0.5):
    """
    Extract entities from model output with confidence filtering.
    This is a placeholder - you'll need to adapt based on your model's actual output format.
    """
    try:
        # Try to parse JSON output
        if '{' in model_output and '}' in model_output:
            json_match = re.search(r'\{.*\}', model_output, re.DOTALL)
            if json_match:
                entities_dict = json.loads(json_match.group())
                return entities_dict
        
        # Fallback to empty entities if parsing fails
        return {"people": [], "places": [], "dates": []}
    except:
        return {"people": [], "places": [], "dates": []}

def generate_prediction(model, tokenizer, document, max_length=512):
    """Generate prediction for a document."""
    prompt = f"""Extract entities from this Italian text and return JSON with people, places, and dates.

Text: {document}

JSON:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.7,  # Increase temperature to encourage more predictions
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

def calculate_metrics(predictions, ground_truth):
    """Calculate precision, recall, and F1 for each entity type."""
    metrics = {}
    
    for entity_type in ['people', 'places', 'dates']:
        tp = 0  # True positives
        fp = 0  # False positives  
        fn = 0  # False negatives
        
        for pred, gt in zip(predictions, ground_truth):
            pred_entities = set(pred.get(entity_type, []))
            gt_entities = set(gt.get(entity_type, []))
            
            tp += len(pred_entities & gt_entities)
            fp += len(pred_entities - gt_entities)
            fn += len(gt_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return metrics

def tune_generation_parameters(model, tokenizer, test_data, output_file):
    """
    Tune generation parameters to improve recall.
    Test different temperature, top_p, and prompt strategies.
    """
    results = []
    
    # Test different parameter combinations
    test_configs = [
        {"temperature": 0.1, "top_p": 0.9, "do_sample": False, "name": "conservative"},
        {"temperature": 0.3, "top_p": 0.9, "do_sample": True, "name": "low_temp"},
        {"temperature": 0.7, "top_p": 0.9, "do_sample": True, "name": "medium_temp"},
        {"temperature": 1.0, "top_p": 0.9, "do_sample": True, "name": "high_temp"},
        {"temperature": 0.7, "top_p": 0.95, "do_sample": True, "name": "high_top_p"},
        {"temperature": 0.7, "top_p": 0.8, "do_sample": True, "name": "low_top_p"},
    ]
    
    # Different prompt strategies
    prompt_strategies = [
        {
            "name": "standard",
            "template": "Extract entities from this Italian text and return JSON with people, places, and dates.\n\nText: {document}\n\nJSON:"
        },
        {
            "name": "encouraging",
            "template": "Carefully extract ALL people, places, and dates from this Italian text. Be thorough and include even ambiguous cases.\n\nText: {document}\n\nEntities (JSON format):"
        },
        {
            "name": "detailed",
            "template": "Extract entities from Italian text:\n- People: full names, titles (Prof., Dott., etc.)\n- Places: cities, buildings, institutions\n- Dates: any temporal references\n\nText: {document}\n\nJSON:"
        }
    ]
    
    for config in test_configs:
        for prompt_strategy in prompt_strategies:
            print(f"Testing {config['name']} with {prompt_strategy['name']} prompt...")
            
            predictions = []
            ground_truth = []
            
            for i, example in enumerate(test_data[:50]):  # Test on first 50 examples
                document = example['document']
                gt = json.loads(example['output'])
                ground_truth.append(gt)
                
                # Generate prompt
                prompt = prompt_strategy['template'].format(document=document)
                
                # Generate prediction
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=256,
                        temperature=config.get('temperature', 0.7),
                        top_p=config.get('top_p', 0.9),
                        do_sample=config.get('do_sample', True),
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Extract entities
                entities = extract_entities_with_scores(response)
                predictions.append(entities)
                
                if i % 10 == 0:
                    print(f"  Processed {i+1}/50 examples")
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, ground_truth)
            
            result = {
                "config": config,
                "prompt_strategy": prompt_strategy['name'],
                "metrics": metrics,
                "macro_f1": np.mean([m['f1'] for m in metrics.values()])
            }
            results.append(result)
            
            print(f"  Macro F1: {result['macro_f1']:.4f}")
            for entity_type, m in metrics.items():
                print(f"    {entity_type}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['macro_f1'])
    
    print(f"\n=== BEST CONFIGURATION ===")
    print(f"Config: {best_result['config']['name']}")
    print(f"Prompt: {best_result['prompt_strategy']}")
    print(f"Macro F1: {best_result['macro_f1']:.4f}")
    print(f"Parameters: {best_result['config']}")
    
    return best_result

def main():
    parser = argparse.ArgumentParser(description='Tune confidence thresholds')
    parser.add_argument('--model-path', required=True, help='Path to fine-tuned model')
    parser.add_argument('--test-data', default='data/val_comprehensive.jsonl', help='Test data file')
    parser.add_argument('--output', default='outputs/threshold_tuning_results.json', help='Output results file')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    test_data = load_test_data(args.test_data)
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Tune parameters
    best_config = tune_generation_parameters(model, tokenizer, test_data, args.output)
    
    # Save best configuration
    best_config_file = Path(args.output).parent / 'best_generation_config.json'
    with open(best_config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Best configuration saved to: {best_config_file}")

if __name__ == '__main__':
    main()