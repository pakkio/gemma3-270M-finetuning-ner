#!/usr/bin/env python3
"""
Comprehensive data expansion to create 300+ validation examples
and balanced training data with adequate examples per entity type.
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import itertools

def load_jsonl(file_path):
    """Load JSONL data from file."""
    data = []
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def generate_additional_examples():
    """Generate additional Italian NER examples to expand dataset."""
    
    # Italian names, places, and dates for generation
    italian_names = [
        "Mario Rossi", "Giulia Bianchi", "Francesco Romano", "Elena Sicilia", 
        "Marco Marino", "Chiara Lombardi", "Andrea Veneto", "Silvia Toscana",
        "Roberto Romano", "Francesca Sicilia", "Giuseppe Marino", "Anna Romano",
        "Prof. Carlo Neri", "Dott.ssa Maria Conti", "Ing. Paolo Ferrari",
        "Dott. Luigi Romano", "Prof.ssa Elena Bianchi", "Avv. Marco Rossi"
    ]
    
    italian_places = [
        "Roma", "Milano", "Napoli", "Torino", "Palermo", "Genova", "Bologna",
        "Firenze", "Catania", "Venezia", "Verona", "Messina", "Padova", "Trieste",
        "Teatro alla Scala", "Colosseo", "Duomo di Milano", "Palazzo Pitti",
        "Università di Bologna", "Università Bocconi", "Politecnico di Milano",
        "Piazza San Marco", "Fontana di Trevi", "Ponte Vecchio", "Castel Sant'Angelo",
        "Palazzo della Pilotta", "Castello Sforzesco", "Teatro San Carlo",
        "Biblioteca Nazionale", "Museo Uffizi", "Galleria dell'Accademia"
    ]
    
    italian_dates = [
        "15 marzo 2024", "3 aprile 2025", "22 settembre 2023", "18 dicembre 2024",
        "dal 10 al 15 maggio 2024", "il 25 gennaio 2025", "tra il 5 e il 10 giugno 2024",
        "30 novembre 2023", "14 febbraio 2025", "7 agosto 2024"
    ]
    
    # Templates for generating examples
    templates = [
        {
            "template": "Il convegno organizzato da {person} si terrà presso {place} {date}.",
            "entities": {"people": 1, "places": 1, "dates": 1}
        },
        {
            "template": "{person} presenterà il suo nuovo libro a {place} e {place2} {date}.",
            "entities": {"people": 1, "places": 2, "dates": 1}
        },
        {
            "template": "La mostra fotografica avrà luogo a {place} {date}. L'evento è curato da {person}.",
            "entities": {"people": 1, "places": 1, "dates": 1}
        },
        {
            "template": "{place}, {date} — {person} inaugurerà la nuova sede presso {place2}.",
            "entities": {"people": 1, "places": 2, "dates": 1}
        },
        {
            "template": "Il festival della musica vedrà la partecipazione di {person} e {person2}. Gli eventi si svolgeranno a {place} {date}.",
            "entities": {"people": 2, "places": 1, "dates": 1}
        },
        {
            "template": "La conferenza stampa di {person} è programmata per {date}. L'incontro avrà luogo presso {place}.",
            "entities": {"people": 1, "places": 1, "dates": 1}
        },
        {
            "template": "Il symposium internazionale si terrà tra {place} e {place2} {date}.",
            "entities": {"people": 0, "places": 2, "dates": 1}
        },
        {
            "template": "{person} e {person2} terranno una lezione magistrale presso {place} {date}.",
            "entities": {"people": 2, "places": 1, "dates": 1}
        },
        {
            "template": "L'inaugurazione del nuovo museo è prevista per {date} a {place}.",
            "entities": {"people": 0, "places": 1, "dates": 1}
        },
        {
            "template": "Il Premio Nobel {person} visiterà {place} e {place2} {date}.",
            "entities": {"people": 1, "places": 2, "dates": 1}
        }
    ]
    
    # Generate examples
    generated_examples = []
    
    for template_info in templates:
        template = template_info["template"]
        entity_counts = template_info["entities"]
        
        # Generate multiple variations of each template
        for _ in range(50):  # 50 variations per template
            used_names = random.sample(italian_names, min(entity_counts["people"], len(italian_names)))
            used_places = random.sample(italian_places, min(entity_counts["places"], len(italian_places))) 
            used_dates = random.sample(italian_dates, min(entity_counts["dates"], len(italian_dates)))
            
            # Fill template
            replacements = {}
            if entity_counts["people"] >= 1:
                replacements["person"] = used_names[0] if used_names else ""
            if entity_counts["people"] >= 2:
                replacements["person2"] = used_names[1] if len(used_names) > 1 else ""
            if entity_counts["places"] >= 1:
                replacements["place"] = used_places[0] if used_places else ""
            if entity_counts["places"] >= 2:
                replacements["place2"] = used_places[1] if len(used_places) > 1 else ""
            if entity_counts["dates"] >= 1:
                replacements["date"] = used_dates[0] if used_dates else ""
            
            # Create document
            try:
                document = template.format(**replacements)
                
                # Create output
                output = {
                    "people": used_names[:entity_counts["people"]],
                    "places": used_places[:entity_counts["places"]],
                    "dates": used_dates[:entity_counts["dates"]]
                }
                
                example = {
                    "document": document,
                    "output": json.dumps(output, ensure_ascii=False)
                }
                
                generated_examples.append(example)
                
            except KeyError:
                continue  # Skip if template couldn't be filled
    
    return generated_examples

def create_ambiguous_test_cases():
    """Create ambiguous test cases for evaluation."""
    ambiguous_cases = [
        {
            "document": "Romano Prodi, ex Presidente del Consiglio, discuterà dell'impero romano presso l'Università di Bologna il 15 marzo 2024.",
            "output": json.dumps({
                "people": ["Romano Prodi"],
                "places": ["Università di Bologna"],
                "dates": ["15 marzo 2024"]
            }, ensure_ascii=False)
        },
        {
            "document": "Il sindaco Marino ha visitato la città di Marino per l'inaugurazione del nuovo parco il 20 aprile 2024.",
            "output": json.dumps({
                "people": ["Marino"],
                "places": ["Marino"],
                "dates": ["20 aprile 2024"]
            }, ensure_ascii=False)
        },
        {
            "document": "Sicilia è la regione più grande d'Italia. Maria Sicilia, professoressa di geografia, terrà una conferenza a Palermo il 10 maggio 2024.",
            "output": json.dumps({
                "people": ["Maria Sicilia"],
                "places": ["Sicilia", "Palermo"],
                "dates": ["10 maggio 2024"]
            }, ensure_ascii=False)
        },
        {
            "document": "La famiglia Romano gestisce il ristorante Romano da tre generazioni. Giuseppe Romano inaugurerà la nuova sede a Roma il 3 giugno 2024.",
            "output": json.dumps({
                "people": ["Giuseppe Romano"],
                "places": ["Roma"],
                "dates": ["3 giugno 2024"]
            }, ensure_ascii=False)
        },
        {
            "document": "Venezia accoglierà la Biennale d'Arte. Il curatore Franco Veneto presenterà le opere presso Ca' Pesaro dal 15 al 30 settembre 2024.",
            "output": json.dumps({
                "people": ["Franco Veneto"],
                "places": ["Venezia", "Ca' Pesaro"],
                "dates": ["dal 15 al 30 settembre 2024"]
            }, ensure_ascii=False)
        }
    ]
    
    return ambiguous_cases

def balance_dataset(train_data, val_data, target_train_size=800, target_val_size=350):
    """Balance dataset ensuring adequate validation size and entity distribution."""
    
    # Generate additional examples
    print("Generating additional examples...")
    generated = generate_additional_examples()
    ambiguous = create_ambiguous_test_cases()
    
    # Combine all data
    all_data = train_data + val_data + generated + ambiguous
    random.shuffle(all_data)
    
    print(f"Total available examples: {len(all_data)}")
    
    # Ensure we have enough data
    if len(all_data) < target_train_size + target_val_size:
        print(f"Warning: Only {len(all_data)} examples available, need {target_train_size + target_val_size}")
        target_val_size = min(target_val_size, len(all_data) // 3)
        target_train_size = len(all_data) - target_val_size
    
    # Split data
    val_data = all_data[:target_val_size]
    train_data = all_data[target_val_size:target_val_size + target_train_size]
    
    return train_data, val_data

def print_stats(data, name):
    """Print dataset statistics."""
    people_count = 0
    places_count = 0
    dates_count = 0
    
    for example in data:
        output = json.loads(example['output'])
        if output.get('people', []):
            people_count += 1
        if output.get('places', []):
            places_count += 1
        if output.get('dates', []):
            dates_count += 1
    
    print(f"\n{name} statistics:")
    print(f"  Total examples: {len(data)}")
    print(f"  Examples with people: {people_count} ({people_count/len(data)*100:.1f}%)")
    print(f"  Examples with places: {places_count} ({places_count/len(data)*100:.1f}%)")
    print(f"  Examples with dates: {dates_count} ({dates_count/len(data)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive data expansion')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--target-train-size', type=int, default=800, help='Target training set size')
    parser.add_argument('--target-val-size', type=int, default=350, help='Target validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    
    # Load existing data
    print("Loading existing data...")
    existing_train = load_jsonl('data/expanded/train_expanded.jsonl')
    existing_val = load_jsonl('data/expanded/val_expanded.jsonl')
    
    # Balance and expand dataset
    new_train, new_val = balance_dataset(
        existing_train, 
        existing_val,
        target_train_size=args.target_train_size,
        target_val_size=args.target_val_size
    )
    
    # Save new datasets
    train_file = output_dir / 'train_comprehensive.jsonl'
    val_file = output_dir / 'val_comprehensive.jsonl'
    
    save_jsonl(new_train, train_file)
    save_jsonl(new_val, val_file)
    
    # Print statistics
    print_stats(new_train, "New Training Set")
    print_stats(new_val, "New Validation Set")
    
    print(f"\nFiles saved:")
    print(f"  Training: {train_file} ({len(new_train)} examples)")
    print(f"  Validation: {val_file} ({len(new_val)} examples)")
    
    # Create test set with ambiguous cases
    test_cases = create_ambiguous_test_cases()
    test_file = output_dir / 'test_ambiguous.jsonl'
    save_jsonl(test_cases, test_file)
    print(f"  Ambiguous test cases: {test_file} ({len(test_cases)} examples)")

if __name__ == '__main__':
    main()