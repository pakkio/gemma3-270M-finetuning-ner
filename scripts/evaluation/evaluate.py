#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset
from scripts.inference import load_model_and_tokenizer, extract_entities

def normalize_entity(entity):
    """Normalizza un'entità per il confronto"""
    return entity.strip().lower()

def compute_metrics(predictions, ground_truth, entity_type):
    """Calcola precision, recall, F1 per un tipo di entità"""
    pred_entities = set(normalize_entity(e) for e in predictions.get(entity_type, []))
    true_entities = set(normalize_entity(e) for e in ground_truth.get(entity_type, []))
    
    if not true_entities and not pred_entities:
        return 1.0, 1.0, 1.0  # Perfect match quando entrambi sono vuoti
    elif not true_entities:
        return 0.0, 0.0, 0.0  # Nessuna entità vera, ma predette alcune
    elif not pred_entities:
        return 0.0, 0.0, 0.0  # Entità vere presenti, ma nessuna predetta
    
    tp = len(pred_entities.intersection(true_entities))
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_model(model, tokenizer, dataset, template_path=None):
    """Valuta il modello su un dataset"""
    results = {
        'people': {'predictions': [], 'ground_truth': []},
        'dates': {'predictions': [], 'ground_truth': []},
        'places': {'predictions': [], 'ground_truth': []}
    }
    
    valid_predictions = 0
    total_predictions = 0
    
    for example in dataset:
        document = example['document']
        ground_truth = json.loads(example['output'])
        
        prediction, raw_output = extract_entities(model, tokenizer, document, template_path)
        total_predictions += 1
        
        if prediction is None:
            # Se non riusciamo a parsare il JSON, trattiamo come predizione vuota
            prediction = {'people': [], 'dates': [], 'places': []}
        else:
            valid_predictions += 1
        
        for entity_type in ['people', 'dates', 'places']:
            results[entity_type]['predictions'].append(prediction.get(entity_type, []))
            results[entity_type]['ground_truth'].append(ground_truth.get(entity_type, []))
    
    # Calcola metriche per ogni tipo di entità
    metrics = {}
    for entity_type in ['people', 'dates', 'places']:
        precisions, recalls, f1s = [], [], []
        
        for pred, true in zip(results[entity_type]['predictions'], results[entity_type]['ground_truth']):
            p, r, f = compute_metrics({entity_type: pred}, {entity_type: true}, entity_type)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        
        metrics[entity_type] = {
            'precision': sum(precisions) / len(precisions),
            'recall': sum(recalls) / len(recalls),
            'f1': sum(f1s) / len(f1s)
        }
    
    # Calcola metriche macro
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
    
    metrics['macro'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }
    
    metrics['valid_json_rate'] = valid_predictions / total_predictions
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model_path", type=str, help="Path to the base model for tokenizer (optional)")
    parser.add_argument("--data_path", type=str, default="data/val.jsonl", help="Path to validation data")
    parser.add_argument("--template", type=str, help="Path to inference template file")
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, base_model_path=args.base_model_path)
    print("Model loaded successfully!")

    print(f"Loading validation data from {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    print(f"Loaded {len(dataset)} examples")

    print("Evaluating model...")
    metrics = evaluate_model(model, tokenizer, dataset, args.template)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Valid JSON Rate: {metrics['valid_json_rate']:.3f}")
    print()
    
    for entity_type in ['people', 'dates', 'places']:
        m = metrics[entity_type]
        print(f"{entity_type.upper()}:")
        print(f"  Precision: {m['precision']:.3f}")
        print(f"  Recall:    {m['recall']:.3f}")
        print(f"  F1:        {m['f1']:.3f}")
        print()
    
    print("MACRO AVERAGES:")
    m = metrics['macro']
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall:    {m['recall']:.3f}")
    print(f"  F1:        {m['f1']:.3f}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()