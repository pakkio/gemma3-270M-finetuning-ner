#!/usr/bin/env python3
"""Test DistilBERT Intent Classifier"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def test_distilbert_classifier():
    """Test the trained DistilBERT classifier"""
    print("Loading DistilBERT Intent Classifier...")
    
    try:
        model_path = "outputs/distilbert-intent-classifier-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        def classify_intent(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions.max().item()
            
            intent = model.config.id2label[predicted_class_id]
            return intent, confidence, predictions[0].tolist()
        
        # Test cases with expected intents
        test_cases = [
            ("Non riesco ad accedere al mio account", "account_access"),
            ("Voglio cancellare il mio ordine", "order_cancellation"),
            ("Dove è il mio pacco?", "order_tracking"),
            ("Voglio restituire questo prodotto", "return_refund"),
            ("La carta non funziona", "payment_issues"),
            ("Quanto costa la spedizione?", "shipping_info"),
            ("È disponibile in magazzino?", "product_availability"),
            ("Caratteristiche del prodotto", "product_info"),
            ("Ci sono sconti?", "promotions_discounts"),
            ("Il sito non funziona", "technical_support"),
            ("Voglio cambiare la mia email", "account_management"),
            ("Buongiorno, ho una domanda", "general_inquiry")
        ]
        
        print("\n=== Testing DistilBERT Intent Classification ===")
        correct = 0
        total = len(test_cases)
        
        for text, expected in test_cases:
            intent, confidence, all_probs = classify_intent(text)
            
            print(f"\nInput: '{text}'")
            print(f"Expected: {expected}")
            print(f"Predicted: {intent} (confidence: {confidence:.3f})")
            
            # Show top 3 predictions
            prob_with_labels = [(model.config.id2label[i], prob) for i, prob in enumerate(all_probs)]
            prob_with_labels.sort(key=lambda x: x[1], reverse=True)
            
            print("Top 3 predictions:")
            for i, (label, prob) in enumerate(prob_with_labels[:3]):
                print(f"  {i+1}. {label}: {prob:.3f}")
            
            match = intent == expected
            print(f"Match: {'✓' if match else '✗'}")
            if match:
                correct += 1
            print("-" * 50)
        
        accuracy = correct / total
        print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Show available intents
        print(f"\nAvailable intents ({len(model.config.id2label)}):")
        for i, intent in model.config.id2label.items():
            print(f"  {i}: {intent}")
            
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_distilbert_classifier()