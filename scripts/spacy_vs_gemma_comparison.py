#!/usr/bin/env python3
"""
Head-to-head comparison: Gemma3 vs spaCy fine-tuned vs spaCy generic
Fair fight on the same dataset with detailed metrics.
"""
import json
import time
import spacy
from pathlib import Path
import argparse
from scripts.inference import load_model_and_tokenizer, extract_entities
from sklearn.metrics import precision_recall_fscore_support
import subprocess
import sys

def load_test_data(data_path):
    """Load test data from JSONL."""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['document']
            entities = json.loads(data['output'])
            examples.append((text, entities))
    return examples

def evaluate_spacy_model(model_path, test_data):
    """Evaluate spaCy model on test data."""
    print(f"🔄 Loading spaCy model from {model_path}")
    
    try:
        nlp = spacy.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load spaCy model: {e}")
        return None
    
    results = {"people": [], "dates": [], "places": []}
    predictions = {"people": [], "dates": [], "places": []}
    
    # spaCy label mapping
    label_map = {
        'PERSON': 'people',
        'GPE': 'places', 
        'DATE': 'dates'
    }
    
    start_time = time.time()
    
    for text, ground_truth in test_data:
        doc = nlp(text)
        
        pred = {"people": [], "dates": [], "places": []}
        
        for ent in doc.ents:
            if ent.label_ in label_map:
                category = label_map[ent.label_]
                pred[category].append(ent.text)
        
        # Store results
        for entity_type in ['people', 'dates', 'places']:
            predictions[entity_type].append(pred[entity_type])
            results[entity_type].append(ground_truth[entity_type])
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, results)
    metrics['inference_time'] = inference_time
    metrics['avg_time_per_doc'] = inference_time / len(test_data)
    
    return metrics

def evaluate_gemma_model(model_path, test_data):
    """Evaluate Gemma model on test data."""
    print(f"🔄 Loading Gemma model from {model_path}")
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
    except Exception as e:
        print(f"❌ Failed to load Gemma model: {e}")
        return None
    
    results = {"people": [], "dates": [], "places": []}
    predictions = {"people": [], "dates": [], "places": []}
    valid_outputs = 0
    
    start_time = time.time()
    
    for text, ground_truth in test_data:
        pred, raw_output = extract_entities(model, tokenizer, text)
        
        if pred is not None:
            valid_outputs += 1
            
        # Handle None predictions
        if pred is None:
            pred = {"people": [], "dates": [], "places": []}
        
        # Store results
        for entity_type in ['people', 'dates', 'places']:
            predictions[entity_type].append(pred.get(entity_type, []))
            results[entity_type].append(ground_truth[entity_type])
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, results)
    metrics['inference_time'] = inference_time
    metrics['avg_time_per_doc'] = inference_time / len(test_data)
    metrics['valid_json_rate'] = valid_outputs / len(test_data)
    
    return metrics

def calculate_metrics(predictions, ground_truth):
    """Calculate F1, precision, recall for each entity type."""
    metrics = {}
    
    for entity_type in ['people', 'dates', 'places']:
        pred_entities = predictions[entity_type]
        true_entities = ground_truth[entity_type]
        
        # Convert to sets for comparison
        precisions, recalls, f1s = [], [], []
        
        for pred, true in zip(pred_entities, true_entities):
            pred_set = set([e.strip().lower() for e in pred])
            true_set = set([e.strip().lower() for e in true])
            
            if not true_set and not pred_set:
                precisions.append(1.0)
                recalls.append(1.0)
                f1s.append(1.0)
            elif not true_set:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
            elif not pred_set:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
            else:
                tp = len(pred_set.intersection(true_set))
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
        
        metrics[entity_type] = {
            'precision': sum(precisions) / len(precisions),
            'recall': sum(recalls) / len(recalls),
            'f1': sum(f1s) / len(f1s)
        }
    
    # Calculate macro averages
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
    
    metrics['macro'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }
    
    return metrics

def run_comparison(gemma_path, spacy_path, test_data_path, output_path):
    """Run comprehensive comparison."""
    
    print("🥊 GEMMA3 vs SPACY: HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    
    # Load test data
    test_data = load_test_data(test_data_path)
    print(f"📊 Test examples: {len(test_data)}")
    print()
    
    results = {}
    
    # 1. Evaluate Gemma3 fine-tuned
    print("🤖 EVALUATING GEMMA3 FINE-TUNED")
    print("-" * 40)
    gemma_metrics = evaluate_gemma_model(gemma_path, test_data)
    if gemma_metrics:
        results['gemma3_finetuned'] = gemma_metrics
        print("✅ Gemma3 evaluation complete")
    
    # 2. Evaluate spaCy fine-tuned  
    print("\n🔧 EVALUATING SPACY FINE-TUNED")
    print("-" * 40)
    spacy_metrics = evaluate_spacy_model(spacy_path, test_data)
    if spacy_metrics:
        results['spacy_finetuned'] = spacy_metrics
        print("✅ spaCy fine-tuned evaluation complete")
    
    # 3. Evaluate spaCy generic (from baseline)
    print("\n📦 EVALUATING SPACY GENERIC")
    print("-" * 40)
    try:
        generic_spacy = spacy.load("it_core_news_sm")
        spacy_generic_metrics = evaluate_spacy_generic(generic_spacy, test_data)
        results['spacy_generic'] = spacy_generic_metrics
        print("✅ spaCy generic evaluation complete")
    except Exception as e:
        print(f"❌ spaCy generic evaluation failed: {e}")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display comparison
    display_comparison(results)
    
    return results

def evaluate_spacy_generic(nlp, test_data):
    """Evaluate generic spaCy model."""
    results = {"people": [], "dates": [], "places": []}
    predictions = {"people": [], "dates": [], "places": []}
    
    start_time = time.time()
    
    for text, ground_truth in test_data:
        doc = nlp(text)
        
        pred = {"people": [], "dates": [], "places": []}
        
        for ent in doc.ents:
            if ent.label_ == "PER":
                pred["people"].append(ent.text)
            elif ent.label_ in ["LOC", "ORG", "GPE"]:
                pred["places"].append(ent.text)
            # Generic spaCy has poor Italian date support
        
        for entity_type in ['people', 'dates', 'places']:
            predictions[entity_type].append(pred[entity_type])
            results[entity_type].append(ground_truth[entity_type])
    
    inference_time = time.time() - start_time
    
    metrics = calculate_metrics(predictions, results)
    metrics['inference_time'] = inference_time
    metrics['avg_time_per_doc'] = inference_time / len(test_data)
    
    return metrics

def display_comparison(results):
    """Display comparison table."""
    print("\n🏆 FINAL COMPARISON RESULTS")
    print("=" * 70)
    
    models = ['gemma3_finetuned', 'spacy_finetuned', 'spacy_generic']
    model_names = ['Gemma3 Fine-tuned', 'spaCy Fine-tuned', 'spaCy Generic']
    
    print(f"{'Model':<20} {'People F1':<10} {'Dates F1':<10} {'Places F1':<10} {'Macro F1':<10} {'Time(s)':<8}")
    print("-" * 70)
    
    for model, name in zip(models, model_names):
        if model in results:
            r = results[model]
            people_f1 = r['people']['f1']
            dates_f1 = r['dates']['f1'] 
            places_f1 = r['places']['f1']
            macro_f1 = r['macro']['f1']
            time_s = r['inference_time']
            
            print(f"{name:<20} {people_f1:<10.3f} {dates_f1:<10.3f} {places_f1:<10.3f} {macro_f1:<10.3f} {time_s:<8.1f}")

def main():
    parser = argparse.ArgumentParser(description='Compare Gemma3 vs spaCy models')
    parser.add_argument('--gemma-path', default='outputs/gemma3-comprehensive',
                       help='Path to Gemma3 model')
    parser.add_argument('--spacy-path', default='outputs/spacy-ner-italian/model-best',
                       help='Path to spaCy model')
    parser.add_argument('--test-data', default='data/val_comprehensive.jsonl',
                       help='Test data path')
    parser.add_argument('--output', default='outputs/model_comparison.json',
                       help='Output results file')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not Path(args.gemma_path).exists():
        print(f"❌ Gemma model not found: {args.gemma_path}")
        return False
        
    if not Path(args.spacy_path).exists():
        print(f"❌ spaCy model not found: {args.spacy_path}")
        print("⏳ spaCy model may still be training...")
        return False
    
    results = run_comparison(args.gemma_path, args.spacy_path, args.test_data, args.output)
    print(f"\n💾 Results saved to: {args.output}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)