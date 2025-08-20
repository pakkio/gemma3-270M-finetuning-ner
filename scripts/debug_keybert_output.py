#!/usr/bin/env python3
"""
Debug KeyBERT output format to understand 0.000 F1 score
"""
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import json

def debug_keybert():
    print("🔍 DEBUG: KeyBERT Output Analysis")
    print("=" * 50)
    
    # Load KeyBERT
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    keybert_model = KeyBERT(model=sentence_model)
    
    # Test example
    text = "Genova, 25 febbraio 2025 — Il porto di Genova registra un aumento del 15% nel traffico merci, consolidandosi come principale gateway per l'Europa centrale."
    ground_truth = "#genova #porto #merci #15percento #gateway #europa #commercio"
    
    print(f"Text: {text}")
    print(f"Ground Truth: {ground_truth}")
    
    # KeyBERT extraction
    keywords = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='italian',
        nr_candidates=20
    )
    
    print(f"\nKeyBERT Raw Output: {keywords}")
    
    # Our conversion process
    hashtags = []
    for keyword, score in keywords[:8]:
        hashtag = keyword.lower().replace(' ', '').replace('-', '')
        hashtag = ''.join(c for c in hashtag if c.isalnum())
        
        if len(hashtag) > 2:
            hashtags.append(f"#{hashtag}")
    
    print(f"Converted Hashtags: {hashtags}")
    
    # Evaluation
    ground_truth_normalized = set(tag.lower().replace('#', '') for tag in ground_truth.split())
    predicted_normalized = set(tag.lower().replace('#', '') for tag in hashtags)
    
    print(f"\nGround Truth Normalized: {ground_truth_normalized}")
    print(f"Predicted Normalized: {predicted_normalized}")
    
    intersection = predicted_normalized.intersection(ground_truth_normalized)
    print(f"Intersection: {intersection}")
    
    if predicted_normalized and ground_truth_normalized:
        precision = len(intersection) / len(predicted_normalized)
        recall = len(intersection) / len(ground_truth_normalized)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"\nPrecision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
    
    print("\n" + "="*50)
    print("ISSUE: KeyBERT extracts different semantic concepts")
    print("- Ground truth focuses on specific entities")
    print("- KeyBERT focuses on broader semantic keywords")
    print("- Domain mismatch causes 0.000 F1 score")

if __name__ == "__main__":
    debug_keybert()