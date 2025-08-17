#!/usr/bin/env python3
"""
Sistema di valutazione robusto per rispondere alle critiche sui dataset piccoli.
Implementa cross-validation, confidence intervals, e error analysis dettagliata.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import scipy.stats as stats
from collections import defaultdict, Counter

class RobustEvaluator:
    """Valutazione robusta con confidence intervals e error analysis."""
    
    def __init__(self, output_dir: str = "outputs/robust_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_analysis = defaultdict(list)
    
    def bootstrap_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
        """Calcola confidence interval con bootstrap."""
        n_bootstrap = 1000
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_scores.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        mean_score = np.mean(scores)
        
        return mean_score, lower, upper
    
    def cross_validation_evaluation(self, dataset: List[Dict], model_inference_fn, k: int = 5) -> Dict:
        """Valutazione cross-validation con confidence intervals."""
        if len(dataset) < k:
            print(f"Warning: Dataset too small for {k}-fold CV. Using {len(dataset)}-fold instead.")
            k = len(dataset)
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"Processing fold {fold_idx + 1}/{k}...")
            
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            # Assumiamo che model_inference_fn restituisca predizioni per val_data
            if model_inference_fn:
                predictions = model_inference_fn(val_data)
            else:
                # Placeholder per demo
                predictions = [{"people": [], "places": [], "dates": []} for _ in val_data]
            
            # Calcola metriche per questo fold
            fold_metrics = self.compute_detailed_metrics(predictions, val_data)
            fold_metrics["fold"] = fold_idx
            fold_results.append(fold_metrics)
        
        # Aggrega risultati cross-validation
        cv_summary = self.aggregate_cv_results(fold_results)
        return cv_summary
    
    def compute_detailed_metrics(self, predictions: List[Dict], ground_truth_data: List[Dict]) -> Dict:
        """Calcola metriche dettagliate con error analysis."""
        ground_truth = []
        for item in ground_truth_data:
            try:
                gt = json.loads(item["output"])
                ground_truth.append(gt)
            except:
                ground_truth.append({"people": [], "places": [], "dates": []})
        
        metrics = {}
        error_details = defaultdict(list)
        
        for entity_type in ["people", "places", "dates"]:
            tp, fp, fn, tn = 0, 0, 0, 0
            type_errors = []
            
            for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                pred_entities = set(pred.get(entity_type, []))
                true_entities = set(true.get(entity_type, []))
                
                # True Positives: entità correttamente predette
                tp_entities = pred_entities & true_entities
                tp += len(tp_entities)
                
                # False Positives: entità predette ma non vere
                fp_entities = pred_entities - true_entities
                fp += len(fp_entities)
                
                # False Negatives: entità vere ma non predette
                fn_entities = true_entities - pred_entities
                fn += len(fn_entities)
                
                # Salva errori per analisi
                if fp_entities or fn_entities:
                    document = ground_truth_data[i]["document"]
                    error_details[f"{entity_type}_errors"].append({
                        "document": document,
                        "false_positives": list(fp_entities),
                        "false_negatives": list(fn_entities),
                        "true_positives": list(tp_entities)
                    })
            
            # Calcola metriche
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn
            }
        
        # Macro F1
        f1_scores = [metrics[t]["f1"] for t in ["people", "places", "dates"]]
        metrics["macro_f1"] = np.mean(f1_scores)
        
        # Salva error details
        metrics["error_analysis"] = dict(error_details)
        
        return metrics
    
    def aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggrega risultati cross-validation con confidence intervals."""
        summary = {"cv_folds": len(fold_results), "detailed_results": {}}
        
        # Per ogni metrica, calcola media e confidence interval
        for entity_type in ["people", "places", "dates", "macro_f1"]:
            if entity_type == "macro_f1":
                scores = [fold[entity_type] for fold in fold_results]
                mean, lower, upper = self.bootstrap_confidence_interval(scores)
                summary["detailed_results"][entity_type] = {
                    "mean_f1": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "std": np.std(scores),
                    "all_scores": scores
                }
            else:
                for metric in ["precision", "recall", "f1"]:
                    scores = [fold[entity_type][metric] for fold in fold_results]
                    mean, lower, upper = self.bootstrap_confidence_interval(scores)
                    
                    key = f"{entity_type}_{metric}"
                    summary["detailed_results"][key] = {
                        "mean": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "std": np.std(scores),
                        "all_scores": scores
                    }
        
        return summary
    
    def generate_evaluation_report(self, cv_results: Dict, baseline_comparison: Dict = None) -> str:
        """Genera report completo di valutazione."""
        report = []
        report.append("# 📊 Robust Evaluation Report")
        report.append("")
        report.append(f"**Cross-Validation:** {cv_results['cv_folds']}-fold")
        report.append("")
        
        # Risultati principali con confidence intervals
        report.append("## 🎯 Main Results (with 95% Confidence Intervals)")
        report.append("")
        report.append("| Entity Type | F1 Score | 95% CI | Std Dev |")
        report.append("|-------------|----------|--------|---------|")
        
        for entity_type in ["people", "places", "dates"]:
            key = f"{entity_type}_f1"
            if key in cv_results["detailed_results"]:
                result = cv_results["detailed_results"][key]
                report.append(f"| {entity_type.title()} | {result['mean']:.3f} | [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] | {result['std']:.3f} |")
        
        # Macro F1
        if "macro_f1" in cv_results["detailed_results"]:
            macro = cv_results["detailed_results"]["macro_f1"]
            report.append(f"| **Overall (Macro)** | **{macro['mean_f1']:.3f}** | **[{macro['ci_lower']:.3f}, {macro['ci_upper']:.3f}]** | **{macro['std']:.3f}** |")
        
        report.append("")
        
        # Interpretazione dei risultati
        report.append("## 🔍 Statistical Interpretation")
        report.append("")
        
        if "macro_f1" in cv_results["detailed_results"]:
            macro = cv_results["detailed_results"]["macro_f1"]
            ci_width = macro['ci_upper'] - macro['ci_lower']
            
            if ci_width < 0.1:
                stability = "**Very stable**"
            elif ci_width < 0.2:
                stability = "**Moderately stable**"
            else:
                stability = "**Highly variable**"
            
            report.append(f"- **Model Stability:** {stability} (CI width: {ci_width:.3f})")
            
            if macro['mean_f1'] > 0.7:
                performance = "**Good performance**"
            elif macro['mean_f1'] > 0.5:
                performance = "**Moderate performance**"
            else:
                performance = "**Needs improvement**"
            
            report.append(f"- **Performance Level:** {performance} (F1: {macro['mean_f1']:.3f})")
        
        report.append("")
        
        # Raccomandazioni basate sui risultati
        report.append("## 💡 Evidence-Based Recommendations")
        report.append("")
        
        if "macro_f1" in cv_results["detailed_results"]:
            macro = cv_results["detailed_results"]["macro_f1"]
            
            if macro['std'] > 0.15:
                report.append("1. **High variance detected** → Need larger, more diverse dataset")
            
            if macro['ci_lower'] < 0.4:
                report.append("2. **Lower confidence bound concerning** → Model may fail in some scenarios")
            
            if cv_results['cv_folds'] < 10:
                report.append("3. **Small sample warning** → Results may not generalize, need more data")
        
        return "\n".join(report)
    
    def save_results(self, cv_results: Dict, report: str, additional_data: Dict = None):
        """Salva tutti i risultati della valutazione."""
        # Salva risultati JSON
        with open(self.output_dir / "cv_results.json", 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, ensure_ascii=False, indent=2)
        
        # Salva report
        with open(self.output_dir / "evaluation_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Salva dati aggiuntivi se forniti
        if additional_data:
            with open(self.output_dir / "additional_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(additional_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {self.output_dir}")

def main():
    """Esegue valutazione robusta completa."""
    # Carica dataset espanso
    dataset = []
    for data_file in ["data/expanded/train_expanded.jsonl", "data/expanded/val_expanded.jsonl"]:
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                dataset.extend([json.loads(line) for line in f])
        except FileNotFoundError:
            print(f"File {data_file} not found")
    
    if len(dataset) < 10:
        print("Warning: Dataset very small, results will be unreliable")
    
    print(f"Loaded {len(dataset)} examples for cross-validation")
    
    evaluator = RobustEvaluator()
    
    # Placeholder per model inference function
    def dummy_inference(data):
        # Simula performance realistica per demo
        predictions = []
        for item in data:
            # Parse ground truth per avere baseline ragionevole
            try:
                gt = json.loads(item["output"])
                # Simula predizioni con 70% accuratezza
                pred = {"people": [], "places": [], "dates": []}
                for key in gt:
                    # 70% delle entità vengono predette correttamente
                    correct_entities = gt[key][:int(len(gt[key]) * 0.7)]
                    pred[key] = correct_entities
                predictions.append(pred)
            except:
                predictions.append({"people": [], "places": [], "dates": []})
        return predictions
    
    # Esegui cross-validation
    print("Running cross-validation evaluation...")
    cv_results = evaluator.cross_validation_evaluation(dataset, dummy_inference, k=5)
    
    # Genera report
    report = evaluator.generate_evaluation_report(cv_results)
    
    # Salva risultati
    evaluator.save_results(cv_results, report)
    
    print("\n" + report)

if __name__ == "__main__":
    main()