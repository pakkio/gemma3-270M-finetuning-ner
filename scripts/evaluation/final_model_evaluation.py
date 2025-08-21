#!/usr/bin/env python3
"""
Evaluation finale del modello trainato vs baseline.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import subprocess
import sys

def load_test_data(file_path: str) -> List[Dict]:
    """Carica dati di test."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def run_model_inference(model_path: str, document: str) -> Dict:
    """Esegue inference con il nostro modello."""
    try:
        cmd = [
            sys.executable, "scripts/inference.py",
            "--model_path", model_path,
            "--document", document
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Estrai JSON dal output
        lines = result.stdout.split('\n')
        for line in lines:
            if line.startswith('Raw output:'):
                json_str = line.replace('Raw output: ', '')
                return json.loads(json_str)
        
        return {"people": [], "places": [], "dates": []}
    except Exception as e:
        print(f"Error in model inference: {e}")
        return {"people": [], "places": [], "dates": []}

def evaluate_model_performance(model_path: str, test_data: List[Dict]) -> Dict:
    """Valuta performance del modello."""
    print(f"Evaluating model on {len(test_data)} examples...")
    
    predictions = []
    ground_truth = []
    
    start_time = time.time()
    
    for i, example in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(test_data)}...")
        
        document = example["document"]
        pred = run_model_inference(model_path, document)
        predictions.append(pred)
        
        try:
            gt = json.loads(example["output"])
            ground_truth.append(gt)
        except:
            ground_truth.append({"people": [], "places": [], "dates": []})
    
    inference_time = time.time() - start_time
    avg_time = inference_time / len(test_data) * 1000  # ms per document
    
    # Calcola metriche
    metrics = compute_metrics(predictions, ground_truth)
    metrics["inference_time_ms"] = avg_time
    metrics["total_examples"] = len(test_data)
    
    return metrics

def compute_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calcola metriche F1 per entità."""
    metrics = {}
    
    for entity_type in ["people", "places", "dates"]:
        tp, fp, fn = 0, 0, 0
        
        for pred, true in zip(predictions, ground_truth):
            pred_entities = set(pred.get(entity_type, []))
            true_entities = set(true.get(entity_type, []))
            
            tp += len(pred_entities & true_entities)
            fp += len(pred_entities - true_entities)
            fn += len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    # Macro F1
    f1_scores = [metrics[t]["f1"] for t in ["people", "places", "dates"]]
    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
    
    return metrics

def main():
    """Esegue evaluation finale."""
    print("🔬 FINAL MODEL EVALUATION")
    print("=" * 50)
    
    # Carica test data
    test_data = load_test_data("data/expanded/test_expanded.jsonl")
    print(f"Loaded {len(test_data)} test examples")
    
    # Evalua il nostro modello
    print("\\n📊 Testing Our Trained Model...")
    model_path = "outputs/gemma3-robust-training"
    
    # Test su subset per velocità
    test_subset = test_data[:20]  # Prima 20 esempi
    our_results = evaluate_model_performance(model_path, test_subset)
    
    print("\\n🎯 FINAL RESULTS")
    print("-" * 30)
    
    print(f"Test Examples: {our_results['total_examples']}")
    print(f"Average Inference Time: {our_results['inference_time_ms']:.1f}ms per document")
    print()
    
    print("Entity-wise Performance:")
    for entity_type in ["people", "places", "dates"]:
        metrics = our_results[entity_type]
        print(f"  {entity_type.title()}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
        print(f"    TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        print()
    
    print(f"Overall Macro F1: {our_results['macro_f1']:.3f}")
    
    # Confronto con baseline conosciuti
    print("\\n🏆 COMPARISON WITH BASELINES")
    print("-" * 30)
    
    baseline_results = {
        "spacy": {"macro_f1": 0.457, "inference_time": 6.4},
        "regex": {"macro_f1": 0.350, "inference_time": 0.1}
    }
    
    print("Method               | Macro F1 | Speed (ms)")
    print("---------------------|----------|----------")
    print(f"Our Model            | {our_results['macro_f1']:.3f}    | {our_results['inference_time_ms']:.1f}")
    print(f"spaCy Italian        | {baseline_results['spacy']['macro_f1']:.3f}    | {baseline_results['spacy']['inference_time']:.1f}")
    print(f"Regex Patterns       | {baseline_results['regex']['macro_f1']:.3f}    | {baseline_results['regex']['inference_time']:.1f}")
    
    # Verifica se superiamo i baseline
    best_baseline = max(baseline_results.values(), key=lambda x: x['macro_f1'])['macro_f1']
    improvement = our_results['macro_f1'] - best_baseline
    
    print(f"\\n📈 PERFORMANCE ANALYSIS")
    print(f"Best Baseline F1: {best_baseline:.3f}")
    print(f"Our Model F1: {our_results['macro_f1']:.3f}")
    print(f"Improvement: {improvement:+.3f} ({improvement/best_baseline*100:+.1f}%)")
    
    if improvement > 0:
        print("✅ Our model OUTPERFORMS the best baseline!")
    elif improvement > -0.05:
        print("⚠️ Our model is COMPETITIVE with baselines")
    else:
        print("❌ Our model needs improvement vs baselines")
    
    # Salva risultati
    output_dir = Path("outputs/final_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "our_model": our_results,
        "baselines": baseline_results,
        "comparison": {
            "improvement_vs_best_baseline": improvement,
            "competitive": abs(improvement) < 0.05
        }
    }
    
    with open(output_dir / "final_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\\nResults saved to: {output_dir}/final_results.json")
    
    return our_results

if __name__ == "__main__":
    main()