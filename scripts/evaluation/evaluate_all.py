#!/usr/bin/env python3
"""
Script principale di valutazione per tutti i modelli
Utilizza la nuova struttura organizzata del progetto
"""

import json
import sys
from pathlib import Path

# Aggiungi la root del progetto al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def evaluate_all_models():
    """Esegue la valutazione completa di tutti i modelli"""
    
    print("🚀 Avvio valutazione completa di tutti i modelli...")
    print("=" * 60)
    
    results = {
        "text2sql": {},
        "ner": {},
        "intent_classification": {},
        "hashtag_generation": {}
    }
    
    # Carica i risultati esistenti
    results_dir = project_root / "results"
    
    # Text2SQL results
    text2sql_file = results_dir / "text2sql_final_comparison.json"
    if text2sql_file.exists():
        with open(text2sql_file) as f:
            text2sql_data = json.load(f)
            results["text2sql"] = text2sql_data
    
    # Model metrics
    metrics_file = results_dir / "detailed_model_metrics.json"  
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            results.update(metrics_data)
    
    print("\n📊 RISULTATI CONSOLIDATI")
    print("=" * 60)
    
    # Text2SQL Summary
    if "metrics_comparison" in results["text2sql"]:
        print("\n🔍 TEXT-TO-SQL RESULTS:")
        metrics = results["text2sql"]["metrics_comparison"]
        for metric, values in metrics.items():
            print(f"  {metric}:")
            for model, value in values.items():
                print(f"    {model}: {value}")
    
    # Intent Classification
    if "intent_classification" in results:
        print("\n🎯 INTENT CLASSIFICATION RESULTS:")
        for model, data in results["intent_classification"].items():
            if isinstance(data, dict) and "accuracy" in data:
                print(f"  {model}: {data['accuracy']:.1%} accuracy, {data['f1']:.3f} F1")
    
    # Text2SQL from detailed metrics
    if "text_to_sql" in results:
        print("\n📈 TEXT-TO-SQL DETAILED:")
        for model, data in results["text_to_sql"].items():
            if isinstance(data, dict) and "accuracy" in data:
                print(f"  {model}: {data['accuracy']:.1%} accuracy, {data['f1']:.3f} F1")
    
    print("\n✅ Valutazione completata!")
    print(f"📁 Risultati salvati in: {results_dir}")
    
    return results

if __name__ == "__main__":
    evaluate_all_models()