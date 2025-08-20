#!/usr/bin/env python3
"""
Calculate precision, recall, F1 for spaCy from baseline data
"""

import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def extract_entities_from_json_string(json_str):
    """Extract entities from JSON string, handling malformed JSON"""
    try:
        data = json.loads(json_str)
        return data.get('people', []), data.get('places', []), data.get('dates', [])
    except:
        return [], [], []

def calculate_spacy_metrics():
    """Calculate spaCy metrics from baseline data"""
    
    with open('data/expanded/spacy_baseline.json', 'r') as f:
        data = json.load(f)
    
    all_predictions = []
    all_ground_truth = []
    
    # Process each test case
    for item in data['spacy_baseline']:
        spacy_people, spacy_places, spacy_dates = extract_entities_from_json_string(item['spacy_output'])
        expected_people, expected_places, expected_dates = extract_entities_from_json_string(item['expected_output'])
        
        # Convert to sets for comparison
        spacy_all = set(spacy_people + spacy_places + spacy_dates)
        expected_all = set(expected_people + expected_places + expected_dates)
        
        # For each expected entity, check if spaCy found it
        for entity in expected_all:
            all_ground_truth.append(1)  # True positive expected
            if entity in spacy_all:
                all_predictions.append(1)  # Correctly predicted
            else:
                all_predictions.append(0)  # Missed
        
        # For each spaCy prediction not in ground truth, it's a false positive
        for entity in spacy_all:
            if entity not in expected_all:
                all_ground_truth.append(0)  # False positive
                all_predictions.append(1)   # spaCy predicted it
    
    # Calculate metrics
    if len(all_ground_truth) > 0:
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average='binary', zero_division=0
        )
    else:
        accuracy = precision = recall = f1 = 0.0
    
    # Calculate per-entity type metrics
    people_tp = people_fp = people_fn = 0
    places_tp = places_fp = places_fn = 0
    dates_tp = dates_fp = dates_fn = 0
    
    for item in data['spacy_baseline']:
        spacy_people, spacy_places, spacy_dates = extract_entities_from_json_string(item['spacy_output'])
        expected_people, expected_places, expected_dates = extract_entities_from_json_string(item['expected_output'])
        
        # People metrics
        spacy_people_set = set(spacy_people)
        expected_people_set = set(expected_people)
        people_tp += len(spacy_people_set & expected_people_set)
        people_fp += len(spacy_people_set - expected_people_set)
        people_fn += len(expected_people_set - spacy_people_set)
        
        # Places metrics
        spacy_places_set = set(spacy_places)
        expected_places_set = set(expected_places)
        places_tp += len(spacy_places_set & expected_places_set)
        places_fp += len(spacy_places_set - expected_places_set)
        places_fn += len(expected_places_set - spacy_places_set)
        
        # Dates metrics
        spacy_dates_set = set(spacy_dates)
        expected_dates_set = set(expected_dates)
        dates_tp += len(spacy_dates_set & expected_dates_set)
        dates_fp += len(spacy_dates_set - expected_dates_set)
        dates_fn += len(expected_dates_set - spacy_dates_set)
    
    def calc_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    people_p, people_r, people_f1 = calc_metrics(people_tp, people_fp, people_fn)
    places_p, places_r, places_f1 = calc_metrics(places_tp, places_fp, places_fn)
    dates_p, dates_r, dates_f1 = calc_metrics(dates_tp, dates_fp, dates_fn)
    
    # Weighted average (by frequency in dataset)
    total_entities = people_tp + people_fn + places_tp + places_fn + dates_tp + dates_fn
    if total_entities > 0:
        people_weight = (people_tp + people_fn) / total_entities
        places_weight = (places_tp + places_fn) / total_entities  
        dates_weight = (dates_tp + dates_fn) / total_entities
        
        weighted_precision = (people_p * people_weight + places_p * places_weight + dates_p * dates_weight)
        weighted_recall = (people_r * people_weight + places_r * places_weight + dates_r * dates_weight)
        weighted_f1 = (people_f1 * people_weight + places_f1 * places_weight + dates_f1 * dates_weight)
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    return {
        'overall': {
            'accuracy': accuracy,
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'people': {
            'precision': people_p,
            'recall': people_r,
            'f1': people_f1,
            'tp': people_tp,
            'fp': people_fp,
            'fn': people_fn
        },
        'places': {
            'precision': places_p,
            'recall': places_r,
            'f1': places_f1,
            'tp': places_tp,
            'fp': places_fp,
            'fn': places_fn
        },
        'dates': {
            'precision': dates_p,
            'recall': dates_r,
            'f1': dates_f1,
            'tp': dates_tp,
            'fp': dates_fp,
            'fn': dates_fn
        },
        'test_cases': len(data['spacy_baseline'])
    }

if __name__ == "__main__":
    metrics = calculate_spacy_metrics()
    print("🔍 spaCy Baseline Metrics Analysis")
    print("=" * 50)
    print(f"Test cases: {metrics['test_cases']}")
    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {metrics['overall']['accuracy']:.3f}")
    print(f"  Precision: {metrics['overall']['precision']:.3f}")
    print(f"  Recall:    {metrics['overall']['recall']:.3f}")
    print(f"  F1-Score:  {metrics['overall']['f1']:.3f}")
    
    print(f"\nPer-Entity Performance:")
    for entity_type in ['people', 'places', 'dates']:
        m = metrics[entity_type]
        print(f"  {entity_type.title()}:")
        print(f"    Precision: {m['precision']:.3f}")
        print(f"    Recall:    {m['recall']:.3f}")
        print(f"    F1-Score:  {m['f1']:.3f}")
        print(f"    TP/FP/FN:  {m['tp']}/{m['fp']}/{m['fn']}")
    
    # Save results
    with open('outputs/spacy_detailed_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Results saved to: outputs/spacy_detailed_metrics.json")