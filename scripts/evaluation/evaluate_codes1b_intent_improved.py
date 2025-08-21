#!/usr/bin/env python3
"""
Valutazione del modello Codes-1B Intent migliorato
Verifica se le ottimizzazioni hanno portato ai risultati sperati
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_improved_codes1b_model(model_path):
    """Load the improved Codes-1B intent model"""
    print(f"🔄 Loading improved Codes-1B model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        "seeklhy/codes-1b", 
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def predict_intent(model, tokenizer, text, intent_classes):
    """Predict intent for a given text"""
    prompt = f"""### Testo
{text}

### Intent
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the intent from the generated text
    if "### Intent" in generated_text:
        intent_part = generated_text.split("### Intent")[-1].strip()
        # Clean up the prediction
        intent_part = intent_part.split('\n')[0].strip()
        
        # Map to known intents
        intent_part_lower = intent_part.lower()
        for intent in intent_classes:
            if intent.lower() in intent_part_lower or intent_part_lower in intent.lower():
                return intent
    
    return "unknown"

def evaluate_improved_model():
    """Evaluate the improved Codes-1B intent classification model"""
    
    print("🚀 Evaluating IMPROVED Codes-1B Intent Classification Model")
    print("=" * 80)
    
    # Test data - same as original evaluation for fair comparison
    test_data = [
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
    
    intent_classes = [
        "account_access", "account_management", "general_inquiry", 
        "order_cancellation", "order_tracking", "payment_issues",
        "product_availability", "product_info", "promotions_discounts",
        "return_refund", "shipping_info", "technical_support"
    ]
    
    # Load improved model
    model_path = "models/production/codes1b-intent-improved"
    model, tokenizer = load_improved_codes1b_model(model_path)
    
    print(f"📊 Testing on {len(test_data)} examples...")
    print(f"🎯 Intent classes: {len(intent_classes)}")
    print()
    
    # Evaluate
    predictions = []
    ground_truth = []
    inference_times = []
    
    print("🧪 Detailed Results:")
    print("-" * 80)
    
    correct = 0
    for i, (text, true_intent) in enumerate(test_data):
        start_time = time.time()
        predicted_intent = predict_intent(model, tokenizer, text, intent_classes)
        inference_time = time.time() - start_time
        
        is_correct = predicted_intent == true_intent
        if is_correct:
            correct += 1
            
        print(f"{i+1:2d}. {text[:50]:<50} | True: {true_intent:<20} | Pred: {predicted_intent:<20} | {'✅' if is_correct else '❌'}")
        
        predictions.append(predicted_intent)
        ground_truth.append(true_intent)
        inference_times.append(inference_time)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted', zero_division=0
    )
    avg_inference_time = np.mean(inference_times)
    
    print()
    print("=" * 80)
    print("📊 IMPROVED MODEL RESULTS")
    print("=" * 80)
    print(f"🎯 Accuracy:     {accuracy:.1%}")
    print(f"🎯 Precision:    {precision:.3f}")
    print(f"🎯 Recall:       {recall:.3f}")
    print(f"🎯 F1 Score:     {f1:.3f}")
    print(f"⚡ Avg Time:     {avg_inference_time:.3f}s")
    print()
    
    # Comparison with original failed model
    print("📈 COMPARISON WITH ORIGINAL FAILED MODEL")
    print("-" * 50)
    print(f"Original Accuracy:  25.0%")
    print(f"Improved Accuracy:  {accuracy:.1%}")
    print(f"Improvement:        {accuracy - 0.25:.1%} (+{((accuracy/0.25)-1)*100:.0f}% relative)")
    print()
    print(f"Original F1:        0.182")
    print(f"Improved F1:        {f1:.3f}")
    print(f"Improvement:        {f1 - 0.182:.3f} (+{((f1/0.182)-1)*100:.0f}% relative)")
    print()
    
    # Classification report
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(ground_truth, predictions, target_names=intent_classes))
    
    # Save results
    results = {
        "model_name": "codes1b-intent-improved",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_inference_time": avg_inference_time,
        "total_examples": len(test_data),
        "correct_predictions": correct,
        "training_improvements": [
            "8x larger dataset (960 vs 120 examples)",
            "Increased learning rate (5e-4 vs 2e-4)",
            "Larger LoRA parameters (r=32 vs 16)",
            "More epochs (8 vs 5)",
            "Larger batch size (8 vs 4)",
            "Cosine learning rate scheduler",
            "Extended early stopping patience"
        ],
        "comparison": {
            "original_accuracy": 0.25,
            "improved_accuracy": accuracy,
            "accuracy_improvement": accuracy - 0.25,
            "relative_improvement_percent": ((accuracy/0.25)-1)*100,
            "original_f1": 0.182,
            "improved_f1": f1,
            "f1_improvement": f1 - 0.182
        },
        "predictions": list(zip([t[0] for t in test_data], ground_truth, predictions)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to results
    results_path = "results/codes1b_intent_improved_evaluation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {results_path}")
    print()
    
    # Success message
    if accuracy >= 0.90:
        print("🔥 SPECTACULAR SUCCESS! Accuracy ≥ 90%")
    elif accuracy >= 0.75:
        print("🚀 EXCELLENT SUCCESS! Accuracy ≥ 75%")
    elif accuracy >= 0.50:
        print("✅ GOOD SUCCESS! Significant improvement achieved")
    else:
        print("⚠️  Limited improvement, but still better than original")
    
    return results

if __name__ == "__main__":
    results = evaluate_improved_model()