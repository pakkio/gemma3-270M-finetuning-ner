#!/usr/bin/env python3
"""
Analyze current dataset balance and entity distribution.
"""
import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path

def analyze_dataset(file_path):
    """Analyze entity distribution in a JSONL dataset file."""
    entity_counts = defaultdict(int)
    examples_with_entities = defaultdict(int)
    total_examples = 0
    empty_examples = defaultdict(int)
    
    # Initialize all entity types
    for entity_type in ['people', 'places', 'dates']:
        entity_counts[entity_type] = 0
        examples_with_entities[entity_type] = 0
        empty_examples[entity_type] = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_examples += 1
                data = json.loads(line)
                output = json.loads(data['output'])
                
                for entity_type in ['people', 'places', 'dates']:
                    entities = output.get(entity_type, [])
                    entity_counts[entity_type] += len(entities)
                    if entities:
                        examples_with_entities[entity_type] += 1
                    else:
                        empty_examples[entity_type] += 1
    
    return {
        'total_examples': total_examples,
        'entity_counts': dict(entity_counts),
        'examples_with_entities': dict(examples_with_entities),
        'empty_examples': dict(empty_examples)
    }

def print_analysis(file_path, analysis):
    """Print detailed analysis of dataset."""
    print(f"\n=== Analysis of {file_path} ===")
    print(f"Total examples: {analysis['total_examples']}")
    
    print(f"\nEntity Distribution:")
    for entity_type in ['people', 'places', 'dates']:
        count = analysis['entity_counts'][entity_type]
        with_entities = analysis['examples_with_entities'][entity_type]
        empty = analysis['empty_examples'][entity_type]
        pct_with = (with_entities / analysis['total_examples']) * 100
        avg_per_example = count / analysis['total_examples']
        
        print(f"  {entity_type.capitalize()}:")
        print(f"    Total entities: {count}")
        print(f"    Examples with {entity_type}: {with_entities} ({pct_with:.1f}%)")
        print(f"    Examples without {entity_type}: {empty} ({100-pct_with:.1f}%)")
        print(f"    Average per example: {avg_per_example:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset balance')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Analyze all dataset files
    files_to_analyze = [
        'train.jsonl',
        'val.jsonl', 
        'train_expanded.jsonl',
        'val_expanded.jsonl',
        'expanded/train_expanded.jsonl',
        'expanded/val_expanded.jsonl'
    ]
    
    total_analysis = {}
    
    for file_name in files_to_analyze:
        file_path = data_dir / file_name
        if file_path.exists():
            analysis = analyze_dataset(file_path)
            total_analysis[file_name] = analysis
            print_analysis(file_name, analysis)
    
    # Summary comparison
    print(f"\n=== SUMMARY COMPARISON ===")
    print(f"{'File':<25} {'Examples':<10} {'People':<8} {'Places':<8} {'Dates':<8}")
    print("-" * 65)
    
    for file_name, analysis in total_analysis.items():
        people = analysis['entity_counts']['people']
        places = analysis['entity_counts']['places'] 
        dates = analysis['entity_counts']['dates']
        examples = analysis['total_examples']
        print(f"{file_name:<25} {examples:<10} {people:<8} {places:<8} {dates:<8}")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    
    # Check validation size
    val_files = [f for f in total_analysis.keys() if 'val' in f]
    max_val_size = max([total_analysis[f]['total_examples'] for f in val_files]) if val_files else 0
    
    if max_val_size < 300:
        print(f"⚠️  VALIDATION SIZE TOO SMALL: {max_val_size} examples (need 300+ minimum)")
    
    # Check entity balance
    for file_name, analysis in total_analysis.items():
        if 'train' in file_name:
            people_pct = (analysis['examples_with_entities']['people'] / analysis['total_examples']) * 100
            places_pct = (analysis['examples_with_entities']['places'] / analysis['total_examples']) * 100
            dates_pct = (analysis['examples_with_entities']['dates'] / analysis['total_examples']) * 100
            
            print(f"\n{file_name} entity coverage:")
            if people_pct < 70:
                print(f"⚠️  People coverage low: {people_pct:.1f}% (target: 70%+)")
            if places_pct < 70:
                print(f"⚠️  Places coverage low: {places_pct:.1f}% (target: 70%+)")
            if dates_pct < 70:
                print(f"⚠️  Dates coverage low: {dates_pct:.1f}% (target: 70%+)")

if __name__ == '__main__':
    main()