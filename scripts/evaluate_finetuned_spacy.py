#!/usr/bin/env python3
"""
Evaluate a fine-tuned spaCy NER model.
"""

import json
import spacy
from pathlib import Path
import argparse
from sklearn.metrics import precision_recall_fscore_support

def evaluate_spacy_model(model_path, data_path):
    """Evaluates a fine-tuned spaCy model."""
    try:
        nlp = spacy.load(model_path)
        print(f"✅ Loaded fine-tuned spaCy model from {model_path}")
    except OSError:
        print(f"❌ Could not load model from {model_path}")
        return None

    with open(data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    predictions = []
    ground_truth = []

    for item in test_data:
        doc = nlp(item['document'])
        
        # Extract predicted entities
        pred_entities = {'people': [], 'places': [], 'dates': []}
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                pred_entities['people'].append(ent.text)
            elif ent.label_ == 'LOC':
                pred_entities['places'].append(ent.text)
            elif ent.label_ == 'DATE':
                pred_entities['dates'].append(ent.text)
        predictions.append(pred_entities)

        # Extract ground truth entities
        gt_entities = json.loads(item['output'])
        ground_truth.append(gt_entities)

    # Calculate metrics
    metrics = {}
    all_true = []
    all_pred = []

    for entity_type in ['people', 'places', 'dates']:
        true_entities = [item.get(entity_type, []) for item in ground_truth]
        pred_entities = [item.get(entity_type, []) for item in predictions]

        # Flatten lists of lists and create binary labels for all unique entities
        flat_true = []
        flat_pred = []
        all_entities = set()
        for p_list, t_list in zip(pred_entities, true_entities):
            all_entities.update(p_list)
            all_entities.update(t_list)

        for entity in all_entities:
            for p_list, t_list in zip(pred_entities, true_entities):
                flat_pred.append(1 if entity in p_list else 0)
                flat_true.append(1 if entity in t_list else 0)

        if len(flat_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                flat_true, flat_pred, average='binary', zero_division=0
            )
            metrics[entity_type] = {'precision': precision, 'recall': recall, 'f1': f1}
        else:
            metrics[entity_type] = {'precision': 0, 'recall': 0, 'f1': 0}

    # Macro averages
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    metrics['macro'] = {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1}

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/spacy-ner-italian/model-best", help="Path to the fine-tuned spaCy model")
    parser.add_argument("--data_path", type=str, default="data/val.jsonl", help="Path to validation data")
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    args = parser.parse_args()

    metrics = evaluate_spacy_model(args.model_path, args.data_path)

    if metrics:
        print("\n" + "="*50)
        print("spaCy Fine-tuned Model Evaluation Results")
        print("="*50)
        for entity_type in ['people', 'places', 'dates']:
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
