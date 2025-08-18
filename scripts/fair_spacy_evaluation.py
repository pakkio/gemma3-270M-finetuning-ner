#!/usr/bin/env python3
"""
Evaluation corretta e ottimizzata di spaCy per confronto onesto.
"""

import json
import spacy
from pathlib import Path
from typing import List, Dict, Tuple
import time

class FairSpacyEvaluator:
    """Evaluator che testa spaCy nelle condizioni ottimali."""
    
    def __init__(self):
        self.models = {}
        self.load_spacy_models()
    
    def load_spacy_models(self):
        """Carica tutti i modelli spaCy italiani disponibili."""
        model_names = [
            "it_core_news_sm",
            "it_core_news_md", 
            "it_core_news_lg"
        ]
        
        for model_name in model_names:
            try:
                self.models[model_name] = spacy.load(model_name)
                print(f"✅ Loaded {model_name}")
            except OSError:
                print(f"❌ {model_name} not available")
        
        if not self.models:
            print("No spaCy Italian models found!")
    
    def extract_entities_spacy(self, text: str, model_name: str, 
                              fuzzy_matching: bool = False) -> Dict:
        """Estrae entità con spaCy ottimizzato."""
        if model_name not in self.models:
            return {"people": [], "places": [], "dates": []}
        
        nlp = self.models[model_name]
        doc = nlp(text)
        
        entities = {"people": [], "places": [], "dates": []}
        
        for ent in doc.ents:
            # Person entities
            if ent.label_ == "PER":
                entities["people"].append(ent.text.strip())
            
            # Location entities - più inclusivo
            elif ent.label_ in ["LOC", "GPE", "ORG"]:
                # ORG spesso include università, teatri, etc.
                entities["places"].append(ent.text.strip())
            
            # Date entities - cerchiamo di estrarre da temporal expressions
            elif ent.label_ in ["DATE", "TIME"]:
                entities["dates"].append(ent.text.strip())
        
        # Se fuzzy matching, pulisci e normalizza
        if fuzzy_matching:
            entities = self.normalize_entities(entities)
        
        return entities
    
    def normalize_entities(self, entities: Dict) -> Dict:
        """Normalizza entità per matching più flessibile."""
        normalized = {}
        
        for entity_type, entity_list in entities.items():
            normalized_list = []
            for entity in entity_list:
                # Rimuovi articoli e preposizioni comuni
                entity = entity.replace("il ", "").replace("la ", "").replace("lo ", "")
                entity = entity.replace("dell'", "").replace("del ", "").replace("della ", "")
                entity = entity.strip()
                
                if entity and len(entity) > 1:
                    normalized_list.append(entity)
            
            normalized[entity_type] = list(set(normalized_list))  # Remove duplicates
        
        return normalized
    
    def evaluate_on_test_cases(self, test_cases: List[Dict]) -> Dict:
        """Evalua spaCy su test cases con diverse configurazioni."""
        results = {}
        
        for model_name in self.models.keys():
            print(f"\\nTesting {model_name}...")
            
            # Test con matching esatto
            exact_results = self.run_evaluation(test_cases, model_name, fuzzy_matching=False)
            
            # Test con matching fuzzy
            fuzzy_results = self.run_evaluation(test_cases, model_name, fuzzy_matching=True)
            
            results[model_name] = {
                "exact_matching": exact_results,
                "fuzzy_matching": fuzzy_results
            }
        
        return results
    
    def run_evaluation(self, test_cases: List[Dict], model_name: str, 
                      fuzzy_matching: bool) -> Dict:
        """Esegue evaluation con configurazione specifica."""
        predictions = []
        ground_truth = []
        
        start_time = time.time()
        
        for case in test_cases:
            document = case["document"]
            pred = self.extract_entities_spacy(document, model_name, fuzzy_matching)
            predictions.append(pred)
            
            # Parse ground truth
            try:
                if isinstance(case.get("expected"), dict):
                    gt = case["expected"]
                else:
                    gt = json.loads(case["output"])
                ground_truth.append(gt)
            except:
                ground_truth.append({"people": [], "places": [], "dates": []})
        
        inference_time = time.time() - start_time
        
        # Calcola metriche
        metrics = self.compute_metrics(predictions, ground_truth, fuzzy_matching)
        metrics["inference_time_ms"] = (inference_time / len(test_cases)) * 1000
        metrics["model_name"] = model_name
        metrics["fuzzy_matching"] = fuzzy_matching
        
        return metrics
    
    def compute_metrics(self, predictions: List[Dict], ground_truth: List[Dict], 
                       fuzzy_matching: bool) -> Dict:
        """Calcola metriche con opzioni di matching."""
        metrics = {}
        
        for entity_type in ["people", "places", "dates"]:
            tp, fp, fn = 0, 0, 0
            
            for pred, true in zip(predictions, ground_truth):
                pred_entities = set(pred.get(entity_type, []))
                true_entities = set(true.get(entity_type, []))
                
                if fuzzy_matching:
                    # Matching più flessibile per spaCy
                    tp_fuzzy, fp_fuzzy, fn_fuzzy = self.fuzzy_match(pred_entities, true_entities)
                    tp += tp_fuzzy
                    fp += fp_fuzzy  
                    fn += fn_fuzzy
                else:
                    # Matching esatto
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
    
    def fuzzy_match(self, pred_set: set, true_set: set) -> Tuple[int, int, int]:
        """Matching fuzzy per confronto più equo."""
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        # Cerca matches fuzzy
        for pred_entity in pred_set:
            for true_entity in true_set:
                if self.entities_match_fuzzy(pred_entity, true_entity):
                    tp += 1
                    matched_true.add(true_entity)
                    matched_pred.add(pred_entity)
                    break
        
        fp = len(pred_set - matched_pred)
        fn = len(true_set - matched_true)
        
        return tp, fp, fn
    
    def entities_match_fuzzy(self, entity1: str, entity2: str) -> bool:
        """Determina se due entità sono match fuzzy."""
        # Normalizza
        e1 = entity1.lower().strip()
        e2 = entity2.lower().strip()
        
        # Match esatto
        if e1 == e2:
            return True
        
        # Uno contiene l'altro (per luoghi come "Milano" vs "Università di Milano")
        if e1 in e2 or e2 in e1:
            return True
        
        # Match su parole chiave per nomi composti
        words1 = set(e1.split())
        words2 = set(e2.split())
        
        # Se condividono parole significative (>2 caratteri)
        significant_words1 = {w for w in words1 if len(w) > 2}
        significant_words2 = {w for w in words2 if len(w) > 2}
        
        if significant_words1 & significant_words2:
            return True
        
        return False

def main():
    """Esegue evaluation completa di spaCy."""
    print("🔧 FAIR spaCy EVALUATION")
    print("=" * 40)
    
    evaluator = FairSpacyEvaluator()
    
    # Carica test cases originali
    original_test = []
    try:
        with open("data/expanded/test_expanded.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_test.append(data)
    except FileNotFoundError:
        print("Original test data not found")
    
    # Carica test cases ambigui
    ambiguous_test = []
    try:
        with open("data/challenging_tests/ambiguous_test_cases.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                ambiguous_test.append(data)
    except FileNotFoundError:
        print("Ambiguous test data not found")
    
    # Test entrambi i dataset
    test_datasets = {
        "original": original_test[:20],  # Prima 20 per velocità
        "ambiguous": ambiguous_test
    }
    
    all_results = {}
    
    for dataset_name, test_data in test_datasets.items():
        if not test_data:
            continue
            
        print(f"\\n📊 Testing on {dataset_name} dataset ({len(test_data)} examples)")
        results = evaluator.evaluate_on_test_cases(test_data)
        all_results[dataset_name] = results
        
        # Stampa risultati migliori
        best_score = 0
        best_config = None
        
        for model_name, model_results in results.items():
            for match_type, metrics in model_results.items():
                score = metrics["macro_f1"]
                if score > best_score:
                    best_score = score
                    best_config = f"{model_name} ({match_type})"
        
        print(f"Best spaCy config: {best_config} - F1: {best_score:.3f}")
    
    # Salva risultati completi
    output_dir = Path("outputs/fair_spacy_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "spacy_comprehensive_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\\nResults saved to {output_dir}")
    return all_results

if __name__ == "__main__":
    results = main()