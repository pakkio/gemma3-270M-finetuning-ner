#!/usr/bin/env python3
"""
Dataset expansion strategy per rispondere alle critiche sui dataset piccoli.
Implementa multiple strategie per espandere il dataset in modo sistematico.
"""

import json
import random
import requests
from typing import List, Dict, Any
from pathlib import Path
import spacy
from datasets import load_dataset

class DatasetExpander:
    """Espande il dataset usando multiple strategie."""
    
    def __init__(self, output_dir: str = "data/expanded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Carica modello spaCy italiano per baseline comparison
        try:
            self.nlp = spacy.load("it_core_news_sm")
        except OSError:
            print("spaCy Italian model not found. Install with: python -m spacy download it_core_news_sm")
            self.nlp = None
    
    def load_evalita_icab_data(self) -> List[Dict]:
        """
        Carica dataset I-CAB/EVALITA se disponibile.
        Fallback su dataset pubblici alternativi.
        """
        expanded_data = []
        
        # 1. Prova a caricare WikiNER da HuggingFace
        try:
            dataset = load_dataset("mnaguib/WikiNER", "it", split="train")
            print(f"Loaded WikiNER Italian: {len(dataset)} examples")
            
            for example in dataset[:1000]:  # Primi 1000 esempi
                # Converte formato IOB in nostro formato JSON
                tokens = example['tokens']
                ner_tags = example['ner_tags']
                
                document = " ".join(tokens)
                entities = self._extract_entities_from_iob(tokens, ner_tags)
                
                if any(entities.values()):  # Solo se ci sono entità
                    expanded_data.append({
                        "document": document,
                        "output": json.dumps(entities, ensure_ascii=False)
                    })
                    
        except Exception as e:
            print(f"Could not load WikiNER: {e}")
        
        # 2. Genera esempi sintetici
        synthetic_examples = self._generate_synthetic_examples(500)
        expanded_data.extend(synthetic_examples)
        
        return expanded_data
    
    def _extract_entities_from_iob(self, tokens: List[str], tags: List[int]) -> Dict[str, List[str]]:
        """Converte formato IOB in nostro formato entità."""
        tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        
        entities = {"people": [], "places": [], "dates": []}
        current_entity = ""
        current_type = ""
        
        for token, tag_id in zip(tokens, tags):
            if tag_id >= len(tag_names):
                continue
                
            tag = tag_names[tag_id]
            
            if tag.startswith("B-"):
                if current_entity:
                    self._add_entity(entities, current_entity.strip(), current_type)
                current_entity = token
                current_type = tag[2:]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_entity += " " + token
            else:
                if current_entity:
                    self._add_entity(entities, current_entity.strip(), current_type)
                current_entity = ""
                current_type = ""
        
        if current_entity:
            self._add_entity(entities, current_entity.strip(), current_type)
            
        return entities
    
    def _add_entity(self, entities: Dict, entity: str, entity_type: str):
        """Aggiunge entità al dizionario corretto."""
        if entity_type == "PER":
            entities["people"].append(entity)
        elif entity_type in ["LOC", "ORG"]:  # ORG spesso contiene luoghi
            entities["places"].append(entity)
        # MISC e altri tipi vengono ignorati per ora
    
    def _generate_synthetic_examples(self, count: int) -> List[Dict]:
        """Genera esempi sintetici con pattern italiani."""
        templates = [
            "Il {person} terrà una conferenza a {place} il {date}.",
            "{person} ha visitato {place} durante {date}.",
            "L'assessore {person} ha presentato il progetto a {place} il {date}.",
            "Il direttore {person} incontrerà la stampa presso {place} {date}.",
            "La professoressa {person} insegna all'università di {place} dal {date}.",
        ]
        
        people = [
            "Mario Rossi", "Giulia Bianchi", "Andrea Verdi", "Francesca Neri",
            "Prof. Luigi Romano", "Dott.ssa Maria Conti", "Ing. Paolo Ferrari"
        ]
        
        places = [
            "Milano", "Roma", "Napoli", "Firenze", "Bologna", "Torino",
            "Università Bocconi", "Politecnico di Milano", "Teatro alla Scala",
            "Palazzo Chigi", "Piazza San Marco", "Palazzo Reale"
        ]
        
        dates = [
            "15 marzo 2024", "28 novembre 2024", "12 gennaio 2025",
            "5 giugno 2024", "dal 10 al 15 settembre 2024", "entro il 30 aprile 2025"
        ]
        
        synthetic_data = []
        for _ in range(count):
            template = random.choice(templates)
            person = random.choice(people)
            place = random.choice(places)
            date = random.choice(dates)
            
            document = template.format(person=person, place=place, date=date)
            entities = {
                "people": [person],
                "places": [place],
                "dates": [date]
            }
            
            synthetic_data.append({
                "document": document,
                "output": json.dumps(entities, ensure_ascii=False)
            })
        
        return synthetic_data
    
    def create_baseline_comparison(self, test_data: List[Dict]) -> Dict:
        """Crea confronto con baseline spaCy."""
        if not self.nlp:
            return {"error": "spaCy model not available"}
        
        spacy_results = []
        for example in test_data:
            doc = self.nlp(example["document"])
            
            entities = {"people": [], "places": [], "dates": []}
            for ent in doc.ents:
                if ent.label_ == "PER":
                    entities["people"].append(ent.text)
                elif ent.label_ in ["LOC", "ORG", "GPE"]:
                    entities["places"].append(ent.text)
                # spaCy non rileva date in italiano molto bene
            
            spacy_results.append({
                "document": example["document"],
                "spacy_output": json.dumps(entities, ensure_ascii=False),
                "expected_output": example["output"]
            })
        
        return {"spacy_baseline": spacy_results}
    
    def save_expanded_dataset(self, data: List[Dict], split_name: str):
        """Salva dataset espanso con split appropriati."""
        output_file = self.output_dir / f"{split_name}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(data)} examples to {output_file}")

def main():
    """Esegue espansione dataset completa."""
    expander = DatasetExpander()
    
    # 1. Carica dataset esistente
    original_train = []
    original_val = []
    
    try:
        with open("data/train.jsonl", 'r', encoding='utf-8') as f:
            original_train = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Original train.jsonl not found")
    
    try:
        with open("data/val.jsonl", 'r', encoding='utf-8') as f:
            original_val = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Original val.jsonl not found")
    
    print(f"Original dataset: {len(original_train)} train, {len(original_val)} val")
    
    # 2. Espandi dataset
    print("Expanding dataset with public data and synthetic examples...")
    expanded_data = expander.load_evalita_icab_data()
    
    # 3. Crea split più robusti
    total_data = original_train + original_val + expanded_data
    random.shuffle(total_data)
    
    train_size = int(0.7 * len(total_data))
    val_size = int(0.15 * len(total_data))
    
    new_train = total_data[:train_size]
    new_val = total_data[train_size:train_size + val_size]
    new_test = total_data[train_size + val_size:]
    
    # 4. Salva nuovi dataset
    expander.save_expanded_dataset(new_train, "train_expanded")
    expander.save_expanded_dataset(new_val, "val_expanded")
    expander.save_expanded_dataset(new_test, "test_expanded")
    
    print(f"\nNew splits: {len(new_train)} train, {len(new_val)} val, {len(new_test)} test")
    
    # 5. Crea baseline comparison
    print("Creating spaCy baseline comparison...")
    baseline_results = expander.create_baseline_comparison(new_test[:50])
    
    with open(expander.output_dir / "spacy_baseline.json", 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)
    
    print("Dataset expansion completed!")

if __name__ == "__main__":
    main()