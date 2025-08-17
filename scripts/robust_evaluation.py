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
import matplotlib.pyplot as plt
import seaborn as sns

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
            
            for i, (pred, true) in enumerate(zip(predictions, ground_truth)):\n                pred_entities = set(pred.get(entity_type, []))\n                true_entities = set(true.get(entity_type, []))\n                \n                # True Positives: entità correttamente predette\n                tp_entities = pred_entities & true_entities\n                tp += len(tp_entities)\n                \n                # False Positives: entità predette ma non vere\n                fp_entities = pred_entities - true_entities\n                fp += len(fp_entities)\n                \n                # False Negatives: entità vere ma non predette\n                fn_entities = true_entities - pred_entities\n                fn += len(fn_entities)\n                \n                # Salva errori per analisi\n                if fp_entities or fn_entities:\n                    document = ground_truth_data[i][\"document\"]\n                    error_details[f\"{entity_type}_errors\"].append({\n                        \"document\": document,\n                        \"false_positives\": list(fp_entities),\n                        \"false_negatives\": list(fn_entities),\n                        \"true_positives\": list(tp_entities)\n                    })\n            \n            # Calcola metriche\n            precision = tp / (tp + fp) if (tp + fp) > 0 else 0\n            recall = tp / (tp + fn) if (tp + fn) > 0 else 0\n            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0\n            \n            metrics[entity_type] = {\n                \"precision\": precision,\n                \"recall\": recall,\n                \"f1\": f1,\n                \"tp\": tp,\n                \"fp\": fp,\n                \"fn\": fn,\n                \"support\": tp + fn\n            }\n        \n        # Macro F1\n        f1_scores = [metrics[t][\"f1\"] for t in [\"people\", \"places\", \"dates\"]]\n        metrics[\"macro_f1\"] = np.mean(f1_scores)\n        \n        # Salva error details\n        metrics[\"error_analysis\"] = dict(error_details)\n        \n        return metrics\n    \n    def aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:\n        \"\"\"Aggrega risultati cross-validation con confidence intervals.\"\"\"\n        summary = {\"cv_folds\": len(fold_results), \"detailed_results\": {}}\n        \n        # Per ogni metrica, calcola media e confidence interval\n        for entity_type in [\"people\", \"places\", \"dates\", \"macro_f1\"]:\n            if entity_type == \"macro_f1\":\n                scores = [fold[entity_type] for fold in fold_results]\n                mean, lower, upper = self.bootstrap_confidence_interval(scores)\n                summary[\"detailed_results\"][entity_type] = {\n                    \"mean_f1\": mean,\n                    \"ci_lower\": lower,\n                    \"ci_upper\": upper,\n                    \"std\": np.std(scores),\n                    \"all_scores\": scores\n                }\n            else:\n                for metric in [\"precision\", \"recall\", \"f1\"]:\n                    scores = [fold[entity_type][metric] for fold in fold_results]\n                    mean, lower, upper = self.bootstrap_confidence_interval(scores)\n                    \n                    key = f\"{entity_type}_{metric}\"\n                    summary[\"detailed_results\"][key] = {\n                        \"mean\": mean,\n                        \"ci_lower\": lower,\n                        \"ci_upper\": upper,\n                        \"std\": np.std(scores),\n                        \"all_scores\": scores\n                    }\n        \n        return summary\n    \n    def statistical_significance_test(self, scores_a: List[float], scores_b: List[float]) -> Dict:\n        \"\"\"Test di significatività statistica tra due set di scores.\"\"\"\n        # Paired t-test\n        if len(scores_a) == len(scores_b):\n            t_stat, p_value_paired = stats.ttest_rel(scores_a, scores_b)\n        else:\n            t_stat, p_value_paired = None, None\n        \n        # Independent t-test\n        t_stat_ind, p_value_ind = stats.ttest_ind(scores_a, scores_b)\n        \n        # Wilcoxon signed-rank test (non-parametric)\n        if len(scores_a) == len(scores_b):\n            w_stat, p_value_wilcoxon = stats.wilcoxon(scores_a, scores_b)\n        else:\n            w_stat, p_value_wilcoxon = None, None\n        \n        return {\n            \"paired_ttest\": {\"t_statistic\": t_stat, \"p_value\": p_value_paired},\n            \"independent_ttest\": {\"t_statistic\": t_stat_ind, \"p_value\": p_value_ind},\n            \"wilcoxon\": {\"w_statistic\": w_stat, \"p_value\": p_value_wilcoxon},\n            \"mean_difference\": np.mean(scores_a) - np.mean(scores_b),\n            \"effect_size_cohen_d\": self.cohens_d(scores_a, scores_b)\n        }\n    \n    def cohens_d(self, group1: List[float], group2: List[float]) -> float:\n        \"\"\"Calcola Cohen's d per effect size.\"\"\"\n        n1, n2 = len(group1), len(group2)\n        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)\n        \n        # Pooled standard deviation\n        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))\n        \n        return (np.mean(group1) - np.mean(group2)) / pooled_std\n    \n    def error_pattern_analysis(self, all_errors: Dict) -> Dict:\n        \"\"\"Analizza pattern comuni negli errori.\"\"\"\n        error_patterns = {\n            \"common_false_positives\": defaultdict(int),\n            \"common_false_negatives\": defaultdict(int),\n            \"error_by_entity_length\": defaultdict(list),\n            \"context_analysis\": defaultdict(list)\n        }\n        \n        for entity_type in [\"people\", \"places\", \"dates\"]:\n            errors_key = f\"{entity_type}_errors\"\n            if errors_key in all_errors:\n                for error in all_errors[errors_key]:\n                    # Conta errori comuni\n                    for fp in error[\"false_positives\"]:\n                        error_patterns[\"common_false_positives\"][fp] += 1\n                        error_patterns[\"error_by_entity_length\"][len(fp.split())].append((\"FP\", fp))\n                    \n                    for fn in error[\"false_negatives\"]:\n                        error_patterns[\"common_false_negatives\"][fn] += 1\n                        error_patterns[\"error_by_entity_length\"][len(fn.split())].append((\"FN\", fn))\n                    \n                    # Analisi contesto\n                    if error[\"false_positives\"] or error[\"false_negatives\"]:\n                        error_patterns[\"context_analysis\"][entity_type].append({\n                            \"document_length\": len(error[\"document\"].split()),\n                            \"entity_density\": len(error[\"true_positives\"]) / len(error[\"document\"].split()),\n                            \"has_errors\": True\n                        })\n        \n        return dict(error_patterns)\n    \n    def generate_evaluation_report(self, cv_results: Dict, baseline_comparison: Dict = None) -> str:\n        \"\"\"Genera report completo di valutazione.\"\"\"\n        report = []\n        report.append(\"# 📊 Robust Evaluation Report\")\n        report.append(\"\")\n        report.append(f\"**Cross-Validation:** {cv_results['cv_folds']}-fold\")\n        report.append(\"\")\n        \n        # Risultati principali con confidence intervals\n        report.append(\"## 🎯 Main Results (with 95% Confidence Intervals)\")\n        report.append(\"\")\n        report.append(\"| Entity Type | F1 Score | 95% CI | Std Dev |\")\n        report.append(\"|-------------|----------|--------|---------|\")\n        \n        for entity_type in [\"people\", \"places\", \"dates\"]:\n            key = f\"{entity_type}_f1\"\n            if key in cv_results[\"detailed_results\"]:\n                result = cv_results[\"detailed_results\"][key]\n                report.append(f\"| {entity_type.title()} | {result['mean']:.3f} | [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] | {result['std']:.3f} |\")\n        \n        # Macro F1\n        if \"macro_f1\" in cv_results[\"detailed_results\"]:\n            macro = cv_results[\"detailed_results\"][\"macro_f1\"]\n            report.append(f\"| **Overall (Macro)** | **{macro['mean_f1']:.3f}** | **[{macro['ci_lower']:.3f}, {macro['ci_upper']:.3f}]** | **{macro['std']:.3f}** |\")\n        \n        report.append(\"\")\n        \n        # Interpretazione dei risultati\n        report.append(\"## 🔍 Statistical Interpretation\")\n        report.append(\"\")\n        \n        if \"macro_f1\" in cv_results[\"detailed_results\"]:\n            macro = cv_results[\"detailed_results\"][\"macro_f1\"]\n            ci_width = macro['ci_upper'] - macro['ci_lower']\n            \n            if ci_width < 0.1:\n                stability = \"**Very stable**\"\n            elif ci_width < 0.2:\n                stability = \"**Moderately stable**\"\n            else:\n                stability = \"**Highly variable**\"\n            \n            report.append(f\"- **Model Stability:** {stability} (CI width: {ci_width:.3f})\")\n            \n            if macro['mean_f1'] > 0.7:\n                performance = \"**Good performance**\"\n            elif macro['mean_f1'] > 0.5:\n                performance = \"**Moderate performance**\"\n            else:\n                performance = \"**Needs improvement**\"\n            \n            report.append(f\"- **Performance Level:** {performance} (F1: {macro['mean_f1']:.3f})\")\n        \n        report.append(\"\")\n        \n        # Raccomandazioni basate sui risultati\n        report.append(\"## 💡 Evidence-Based Recommendations\")\n        report.append(\"\")\n        \n        if \"macro_f1\" in cv_results[\"detailed_results\"]:\n            macro = cv_results[\"detailed_results\"][\"macro_f1\"]\n            \n            if macro['std'] > 0.15:\n                report.append(\"1. **High variance detected** → Need larger, more diverse dataset\")\n            \n            if macro['ci_lower'] < 0.4:\n                report.append(\"2. **Lower confidence bound concerning** → Model may fail in some scenarios\")\n            \n            if cv_results['cv_folds'] < 10:\n                report.append(\"3. **Small sample warning** → Results may not generalize, need more data\")\n        \n        return \"\\n\".join(report)\n    \n    def save_results(self, cv_results: Dict, report: str, additional_data: Dict = None):\n        \"\"\"Salva tutti i risultati della valutazione.\"\"\"\n        # Salva risultati JSON\n        with open(self.output_dir / \"cv_results.json\", 'w', encoding='utf-8') as f:\n            json.dump(cv_results, f, ensure_ascii=False, indent=2)\n        \n        # Salva report\n        with open(self.output_dir / \"evaluation_report.md\", 'w', encoding='utf-8') as f:\n            f.write(report)\n        \n        # Salva dati aggiuntivi se forniti\n        if additional_data:\n            with open(self.output_dir / \"additional_analysis.json\", 'w', encoding='utf-8') as f:\n                json.dump(additional_data, f, ensure_ascii=False, indent=2)\n        \n        print(f\"Results saved to {self.output_dir}\")\n\ndef main():\n    \"\"\"Esegue valutazione robusta completa.\"\"\"\n    # Carica dataset\n    dataset = []\n    for data_file in [\"data/train.jsonl\", \"data/val.jsonl\"]:\n        try:\n            with open(data_file, 'r', encoding='utf-8') as f:\n                dataset.extend([json.loads(line) for line in f])\n        except FileNotFoundError:\n            print(f\"File {data_file} not found\")\n    \n    if len(dataset) < 10:\n        print(\"Warning: Dataset very small, results will be unreliable\")\n    \n    evaluator = RobustEvaluator()\n    \n    # Placeholder per model inference function\n    def dummy_inference(data):\n        return [{\"people\": [], \"places\": [], \"dates\": []} for _ in data]\n    \n    # Esegui cross-validation\n    print(\"Running cross-validation evaluation...\")\n    cv_results = evaluator.cross_validation_evaluation(dataset, dummy_inference, k=5)\n    \n    # Genera report\n    report = evaluator.generate_evaluation_report(cv_results)\n    \n    # Salva risultati\n    evaluator.save_results(cv_results, report)\n    \n    print(\"\\n\" + report)\n\nif __name__ == \"__main__\":\n    main()