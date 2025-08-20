#!/usr/bin/env python3
"""
Test KeyBERT for hashtag generation
"""
import json
import time
from pathlib import Path
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

def load_test_data(file_path):
    """Load test data."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append({
                    'document': data['document'],
                    'ground_truth_hashtags': data['hashtags'].split()
                })
    return examples

def generate_keybert_hashtags(keybert_model, text):
    """Generate hashtags using KeyBERT."""
    start_time = time.time()
    
    # Extract keywords using KeyBERT
    keywords = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='italian',
        nr_candidates=20
    )
    
    # Take top 8 keywords
    keywords = keywords[:8]
    
    # Convert keywords to hashtags
    hashtags = []
    for keyword, score in keywords:
        hashtag = keyword.lower().replace(' ', '').replace('-', '')
        hashtag = ''.join(c for c in hashtag if c.isalnum())
        
        if len(hashtag) > 2:
            hashtags.append(f"#{hashtag}")
    
    generation_time = time.time() - start_time
    
    return hashtags, generation_time

def evaluate_hashtags(predicted, ground_truth):
    """Evaluate predicted hashtags against ground truth."""
    # Normalize hashtags (remove #, lowercase)
    pred_normalized = set(tag.lower().replace('#', '') for tag in predicted)
    true_normalized = set(tag.lower().replace('#', '') for tag in ground_truth)
    
    if not true_normalized and not pred_normalized:
        return 1.0, 1.0, 1.0
    elif not true_normalized:
        return 0.0, 0.0, 0.0
    elif not pred_normalized:
        return 0.0, 0.0, 0.0
    
    # Calculate metrics
    intersection = pred_normalized.intersection(true_normalized)
    
    precision = len(intersection) / len(pred_normalized) if pred_normalized else 0.0
    recall = len(intersection) / len(true_normalized) if true_normalized else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def main():
    print("🔑 KEYBERT HASHTAG GENERATION TEST")
    print("=" * 50)
    
    # Load KeyBERT
    print("🔄 Loading KeyBERT...")
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    keybert_model = KeyBERT(model=sentence_model)
    print("✅ KeyBERT loaded successfully")
    
    # Load test data
    test_data_path = "data/val_hashtagger.jsonl"
    examples = load_test_data(test_data_path)
    print(f"📊 Loaded {len(examples)} test examples")
    
    # Process examples
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_time = 0.0
    
    print("\n🔄 Processing examples...")
    for i, example in enumerate(examples, 1):
        print(f"\rExample {i}/{len(examples)}", end="", flush=True)
        
        text = example['document']
        ground_truth = example['ground_truth_hashtags']
        
        hashtags, gen_time = generate_keybert_hashtags(keybert_model, text)
        precision, recall, f1 = evaluate_hashtags(hashtags, ground_truth)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_time += gen_time
    
    # Calculate averages
    avg_precision = total_precision / len(examples)
    avg_recall = total_recall / len(examples)
    avg_f1 = total_f1 / len(examples)
    avg_time = total_time / len(examples)
    examples_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
    
    print(f"\n\n🏆 KEYBERT RESULTS")
    print("=" * 50)
    print(f"Precision: {avg_precision:.3f}")
    print(f"Recall: {avg_recall:.3f}")
    print(f"F1 Score: {avg_f1:.3f}")
    print(f"Avg time per example: {avg_time:.3f}s")
    print(f"Examples per second: {examples_per_sec:.1f}")

if __name__ == "__main__":
    main()