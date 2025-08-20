#!/usr/bin/env python3
"""
Confronto sistematico con baseline per rispondere alle critiche.
Implementa confronti con spaCy, Flair, e regex-based approaches.
"""

import json
import re
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
import spacy
from datasets import load_dataset

class BaselineComparison:
    """Confronta il nostro modello con baseline standard."""
    
    def __init__(self):
        self.results = {}
        self.setup_models()
    
    def setup_models(self):
        """Inizializza i modelli baseline."""
        # spaCy italiano
        try:
            self.spacy_model = spacy.load("it_core_news_sm")
            print("✅ spaCy Italian model loaded")
        except OSError:
            print("❌ spaCy model not found. Install: python -m spacy download it_core_news_sm")
            self.spacy_model = None
        
        # Regex patterns per date italiane
        self.date_patterns = [
            r'\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'(?:dal|entro il|fino al)\s+\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}',
            r'dal\s+\d{1,2}\s+al\s+\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}'
        ]
        
        # Pattern per luoghi italiani comuni
        self.place_patterns = [
            r'\b(?:Milano|Roma|Napoli|Torino|Palermo|Genova|Bologna|Firenze|Bari|Catania|Venezia|Verona|Messina|Padova|Trieste|Brescia|Reggio Calabria|Modena|Prato|Parma)\b',
            r'\b(?:Università|Politecnico|Teatro|Palazzo|Piazza|Via)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:di|della|del)\s+[A-Z][a-z]+\b'
        ]
        
        # Pattern per nomi italiani
        self.people_patterns = [
            r'\b(?:Prof\.|Dott\.|Ing\.|Avv\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?=\s+(?:ha|è|terrà|presenta|dichiara))',
            r"(?:l'assessor[ae]|il sindaco|la sindaca|il direttore|la direttrice)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"
        ]
    
    def spacy_baseline(self, texts: List[str]) -> List[Dict]:
        """Baseline con spaCy italiano."""
        if not self.spacy_model:
            results = [{"people": [], "places": [], "dates": []} for _ in texts]
            self.results["spacy"] = {
                "predictions": results,
                "inference_time": 0.0,
                "avg_time_per_doc": 0.0
            }
            return results
        
        results = []
        start_time = time.time()
        
        for text in texts:
            doc = self.spacy_model(text)
            entities = {"people": [], "places": [], "dates": []}
            
            for ent in doc.ents:
                if ent.label_ == "PER":
                    entities["people"].append(ent.text)
                elif ent.label_ in ["LOC", "ORG", "GPE"]:
                    entities["places"].append(ent.text)
                # spaCy non ha buon supporto per date in italiano
            
            results.append(entities)
        
        inference_time = time.time() - start_time
        self.results["spacy"] = {
            "predictions": results,
            "inference_time": inference_time,
            "avg_time_per_doc": inference_time / len(texts)
        }
        
        return results
    
    def regex_baseline(self, texts: List[str]) -> List[Dict]:
        """Baseline con regex patterns."""
        results = []
        start_time = time.time()
        
        for text in texts:
            entities = {"people": [], "places": [], "dates": []}
            
            # Estrai date
            for pattern in self.date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["dates"].extend(matches)
            
            # Estrai luoghi
            for pattern in self.place_patterns:
                matches = re.findall(pattern, text)
                entities["places"].extend(matches)
            
            # Estrai persone
            for pattern in self.people_patterns:
                matches = re.findall(pattern, text)
                entities["people"].extend(matches)
            
            # Rimuovi duplicati
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            results.append(entities)
        
        inference_time = time.time() - start_time
        self.results["regex"] = {
            "predictions": results,
            "inference_time": inference_time,
            "avg_time_per_doc": inference_time / len(texts)
        }
        
        return results
    
    def compute_entity_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calcola metriche F1 per entità."""
        metrics = {}
        
        for entity_type in ["people", "places", "dates"]:
            pred_entities = []
            true_entities = []
            
            for pred, true in zip(predictions, ground_truth):
                pred_set = set(pred.get(entity_type, []))
                true_set = set(true.get(entity_type, []))
                
                # Crea binary labels per ogni entità possibile
                all_entities = pred_set | true_set
                for entity in all_entities:
                    pred_entities.append(1 if entity in pred_set else 0)
                    true_entities.append(1 if entity in true_set else 0)
            
            if len(pred_entities) > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_entities, pred_entities, average='binary', zero_division=0
                )
                metrics[entity_type] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            else:
                metrics[entity_type] = {"precision": 0, "recall": 0, "f1": 0}
        
        # Calcola F1 macro
        f1_scores = [metrics[t]["f1"] for t in metrics]
        metrics["macro_f1"] = np.mean(f1_scores)
        
        return metrics
    
    def run_comparison(self, test_data: List[Dict]) -> Dict:
        """Esegue confronto completo con tutti i baseline."""
        texts = [item["document"] for item in test_data]
        ground_truth = []
        
        # Parse ground truth
        for item in test_data:
            try:
                gt = json.loads(item["output"])
                ground_truth.append(gt)
            except:
                ground_truth.append({"people": [], "places": [], "dates": []})
        
        results = {
            "test_size": len(test_data),
            "baselines": {}
        }
        
        # Test spaCy
        print("Testing spaCy baseline...")
        spacy_preds = self.spacy_baseline(texts)
        spacy_metrics = self.compute_entity_metrics(spacy_preds, ground_truth)
        results["baselines"]["spacy"] = {
            "metrics": spacy_metrics,
            "inference_time": self.results["spacy"]["avg_time_per_doc"]
        }
        
        # Test Regex
        print("Testing regex baseline...")
        regex_preds = self.regex_baseline(texts)
        regex_metrics = self.compute_entity_metrics(regex_preds, ground_truth)
        results["baselines"]["regex"] = {
            "metrics": regex_metrics,
            "inference_time": self.results["regex"]["avg_time_per_doc"]
        }
        
        return results
    
    def generate_comparison_report(self, comparison_results: Dict, model_results: Dict = None) -> str:
        """Genera report dettagliato del confronto."""
        report = []
        report.append("# 📊 Baseline Comparison Report")
        report.append("")
        report.append(f"**Test dataset size:** {comparison_results['test_size']} examples")
        report.append("")
        
        # Tabella riassuntiva
        report.append("## 🏆 Performance Summary")
        report.append("")
        report.append("| Model | People F1 | Places F1 | Dates F1 | Macro F1 | Inference Time |")
        report.append("|-------|-----------|-----------|----------|----------|----------------|")
        
        # spaCy baseline
        spacy = comparison_results["baselines"]["spacy"]
        report.append(f"| spaCy Italian | {spacy['metrics']['people']['f1']:.3f} | {spacy['metrics']['places']['f1']:.3f} | {spacy['metrics']['dates']['f1']:.3f} | {spacy['metrics']['macro_f1']:.3f} | {spacy['inference_time']*1000:.1f}ms |")
        
        # Regex baseline
        regex = comparison_results["baselines"]["regex"]
        report.append(f"| Regex Patterns | {regex['metrics']['people']['f1']:.3f} | {regex['metrics']['places']['f1']:.3f} | {regex['metrics']['dates']['f1']:.3f} | {regex['metrics']['macro_f1']:.3f} | {regex['inference_time']*1000:.1f}ms |")
        
        # Il nostro modello (se fornito)
        if model_results:
            report.append(f"| **Our Model** | **{model_results.get('people', 0):.3f}** | **{model_results.get('places', 0):.3f}** | **{model_results.get('dates', 0):.3f}** | **{model_results.get('macro_f1', 0):.3f}** | **{model_results.get('inference_time', 0)*1000:.1f}ms** |")
        
        report.append("")
        
        # Analisi dettagliata
        report.append("## 📈 Detailed Analysis")
        report.append("")
        
        report.append("### spaCy Italian Model")
        spacy_metrics = spacy["metrics"]
        report.append(f"- **Strengths:** Good person recognition (F1: {spacy_metrics['people']['f1']:.3f})")
        report.append(f"- **Weaknesses:** Limited date support (F1: {spacy_metrics['dates']['f1']:.3f}), location confusion")
        report.append(f"- **Speed:** Fast inference ({spacy['inference_time']*1000:.1f}ms per document)")
        report.append("")
        
        report.append("### Regex Patterns")
        regex_metrics = regex["metrics"]
        report.append(f"- **Strengths:** Very fast ({regex['inference_time']*1000:.1f}ms), reliable for standard patterns")
        report.append(f"- **Weaknesses:** Rigid patterns, many false positives/negatives")
        report.append(f"- **Use case:** Good for structured data with known formats")
        report.append("")
        
        # Raccomandazioni
        report.append("## 💡 Recommendations")
        report.append("")
        
        best_baseline = "spacy" if spacy["metrics"]["macro_f1"] > regex["metrics"]["macro_f1"] else "regex"
        baseline_f1 = comparison_results["baselines"][best_baseline]["metrics"]["macro_f1"]
        
        report.append(f"1. **Current best baseline:** {best_baseline.title()} (F1: {baseline_f1:.3f})")
        
        if model_results and model_results.get("macro_f1", 0) > baseline_f1:
            improvement = model_results["macro_f1"] - baseline_f1
            report.append(f"2. **Our model improvement:** +{improvement:.3f} F1 points over best baseline")
        else:
            report.append("2. **Model needs improvement** to beat production baselines")
        
        report.append("3. **Hybrid approach:** Consider combining regex for dates + spaCy for entities")
        report.append("4. **Error analysis:** Focus on false positives in place detection")
        
        return "\n".join(report)

def main():
    """Esegue confronto baseline completo."""
    # Carica test data
    test_data = []
    try:
        test_file = Path("data/val.jsonl")
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f]
        else:
            print("No test data found. Creating sample data...")
            test_data = [
                {
                    "document": "Il Prof. Mario Rossi terrà una conferenza a Milano il 15 marzo 2024.",
                    "output": json.dumps({"people": ["Prof. Mario Rossi"], "places": ["Milano"], "dates": ["15 marzo 2024"]})
                },
                {
                    "document": "L'assessora Giulia Bianchi ha visitato l'Università Bocconi ieri.",
                    "output": json.dumps({"people": ["Giulia Bianchi"], "places": ["Università Bocconi"], "dates": []})
                }
            ]
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Esegui confronto
    comparator = BaselineComparison()
    results = comparator.run_comparison(test_data)
    
    # Genera report
    report = comparator.generate_comparison_report(results)
    
    # Salva risultati
    output_dir = Path("outputs/baseline_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + report)
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()