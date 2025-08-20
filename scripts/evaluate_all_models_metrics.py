#!/usr/bin/env python3
"""
Comprehensive evaluation script to get precision, recall, F1, accuracy, and inference speed
for all models including Codes-1B
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def load_codes_1b_model(model_path):
    """Load Codes-1B fine-tuned model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            "seeklhy/codes-1b", 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Codes-1B model from {model_path}: {e}")
        return None, None

def load_gemma3_model(model_path):
    """Load Gemma3-270M fine-tuned model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Gemma3 model from {model_path}: {e}")
        return None, None

def load_distilbert_model(model_path):
    """Load DistilBERT fine-tuned model"""
    try:
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except Exception as e:
        print(f"Error loading DistilBERT model from {model_path}: {e}")
        return None

def evaluate_intent_classification():
    """Evaluate intent classification models"""
    print("🧪 Evaluating Intent Classification Models...")
    
    # Test data
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
    
    results = {}
    
    # Intent classes for mapping
    intent_classes = [
        "account_access", "account_management", "general_inquiry", 
        "order_cancellation", "order_tracking", "payment_issues",
        "product_availability", "product_info", "promotions_discounts",
        "return_refund", "shipping_info", "technical_support"
    ]
    
    # Test Codes-1B Intent
    print("Testing Codes-1B Intent Classification...")
    codes_model, codes_tokenizer = load_codes_1b_model("outputs/codes-1b-intent-classifier")
    if codes_model:
        codes_predictions = []
        codes_times = []
        
        for text, expected in test_data:
            prompt = f"""### Task: Classify this Italian customer service message into an intent category
### Message: {text}
### Intent: """
            
            start_time = time.time()
            try:
                inputs = codes_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = codes_model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=codes_tokenizer.eos_token_id,
                        eos_token_id=codes_tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                result = codes_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                prediction = result.strip().split('\n')[0].split('###')[0].strip()
                
                codes_predictions.append(prediction)
                codes_times.append(time.time() - start_time)
                
            except Exception as e:
                print(f"Error with Codes-1B prediction: {e}")
                codes_predictions.append("error")
                codes_times.append(time.time() - start_time)
        
        # Calculate metrics for Codes-1B
        true_labels = [expected for _, expected in test_data]
        
        # Map predictions to standard labels
        mapped_predictions = []
        for pred in codes_predictions:
            if pred in intent_classes:
                mapped_predictions.append(pred)
            else:
                # Find closest match
                closest = min(intent_classes, key=lambda x: len(set(pred.lower().split()) - set(x.lower().split())))
                mapped_predictions.append(closest)
        
        accuracy = accuracy_score(true_labels, mapped_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, mapped_predictions, average='weighted', zero_division=0
        )
        
        results['codes_1b'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_inference_time': np.mean(codes_times),
            'predictions': list(zip(true_labels, mapped_predictions))
        }
    
    # Test Gemma3-270M Intent
    print("Testing Gemma3-270M Intent Classification...")
    gemma_model, gemma_tokenizer = load_gemma3_model("outputs/gemma3-intent-classification-stable")
    if gemma_model:
        gemma_predictions = []
        gemma_times = []
        
        for text, expected in test_data:
            prompt = f"Classifica questo messaggio: {text}\nCategoria:"
            
            start_time = time.time()
            try:
                inputs = gemma_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = gemma_model.generate(
                        **inputs,
                        max_new_tokens=15,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=gemma_tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                result = gemma_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                prediction = result.strip().split('\n')[0].split()[0].strip()
                
                gemma_predictions.append(prediction)
                gemma_times.append(time.time() - start_time)
                
            except Exception as e:
                print(f"Error with Gemma3 prediction: {e}")
                gemma_predictions.append("error")
                gemma_times.append(time.time() - start_time)
        
        # Calculate metrics for Gemma3
        mapped_predictions = []
        for pred in gemma_predictions:
            if pred in intent_classes:
                mapped_predictions.append(pred)
            else:
                closest = min(intent_classes, key=lambda x: len(set(pred.lower().split()) - set(x.lower().split())))
                mapped_predictions.append(closest)
        
        accuracy = accuracy_score(true_labels, mapped_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, mapped_predictions, average='weighted', zero_division=0
        )
        
        results['gemma3_270m'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_inference_time': np.mean(gemma_times),
            'predictions': list(zip(true_labels, mapped_predictions))
        }
    
    # Test DistilBERT Intent
    print("Testing DistilBERT Intent Classification...")
    distilbert_classifier = load_distilbert_model("outputs/distilbert-intent-classification-stable")
    if distilbert_classifier:
        distilbert_predictions = []
        distilbert_times = []
        
        for text, expected in test_data:
            start_time = time.time()
            try:
                result = distilbert_classifier(text)
                prediction = result[0]['label']
                distilbert_predictions.append(prediction)
                distilbert_times.append(time.time() - start_time)
            except Exception as e:
                print(f"Error with DistilBERT prediction: {e}")
                distilbert_predictions.append("error")
                distilbert_times.append(time.time() - start_time)
        
        # Calculate metrics for DistilBERT
        accuracy = accuracy_score(true_labels, distilbert_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, distilbert_predictions, average='weighted', zero_division=0
        )
        
        results['distilbert'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_inference_time': np.mean(distilbert_times),
            'predictions': list(zip(true_labels, distilbert_predictions))
        }
    
    return results

def evaluate_text_to_sql():
    """Evaluate Text-to-SQL models"""
    print("🧪 Evaluating Text-to-SQL Models...")
    
    test_data = [
        ("Mostra tutti i clienti di Milano", "SELECT * FROM clienti WHERE citta = 'Milano';"),
        ("Quali sono i prodotti con prezzo superiore a 100 euro?", "SELECT * FROM prodotti WHERE prezzo > 100;"),
        ("Conta quanti ordini sono stati effettuati nel 2024", "SELECT COUNT(*) FROM ordini WHERE YEAR(data_ordine) = 2024;"),
        ("Trova i dipendenti del dipartimento vendite", "SELECT * FROM dipendenti WHERE dipartimento = 'vendite';"),
        ("Elenca tutti i prodotti della categoria elettronica", "SELECT * FROM prodotti WHERE categoria = 'elettronica';"),
        ("Mostra il nome e cognome dei clienti di Roma", "SELECT nome, cognome FROM clienti WHERE citta = 'Roma';"),
        ("Calcola il totale degli ordini del 2024", "SELECT SUM(totale) FROM ordini WHERE YEAR(data_ordine) = 2024;"),
        ("Trova tutti i prodotti sotto i 50 euro", "SELECT * FROM prodotti WHERE prezzo < 50;")
    ]
    
    results = {}
    
    # Test Codes-1B Text-to-SQL
    print("Testing Codes-1B Text-to-SQL...")
    codes_model, codes_tokenizer = load_codes_1b_model("outputs/codes-1b-text2sql")
    if codes_model:
        codes_predictions = []
        codes_times = []
        codes_correct = 0
        
        for question, expected_sql in test_data:
            prompt = f"""### Task: Convert this Italian question to SQL query
### Question: {question}
### SQL: """
            
            start_time = time.time()
            try:
                inputs = codes_tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = codes_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=codes_tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                result = codes_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                prediction = result.strip().split('\n')[0].split('###')[0].strip()
                
                # Simple SQL comparison (normalize spaces)
                pred_normalized = ' '.join(prediction.replace(';', '').split()).upper()
                expected_normalized = ' '.join(expected_sql.replace(';', '').split()).upper()
                
                if pred_normalized in expected_normalized or expected_normalized in pred_normalized:
                    codes_correct += 1
                
                codes_predictions.append(prediction)
                codes_times.append(time.time() - start_time)
                
            except Exception as e:
                print(f"Error with Codes-1B SQL prediction: {e}")
                codes_predictions.append("error")
                codes_times.append(time.time() - start_time)
        
        accuracy = codes_correct / len(test_data)
        results['codes_1b'] = {
            'accuracy': accuracy,
            'precision': accuracy,  # For SQL, precision ≈ accuracy
            'recall': accuracy,     # For SQL, recall ≈ accuracy  
            'f1': accuracy,         # For SQL, F1 ≈ accuracy
            'avg_inference_time': np.mean(codes_times),
            'predictions': list(zip([q for q, _ in test_data], codes_predictions))
        }
    
    # Test Gemma3-270M Text-to-SQL
    print("Testing Gemma3-270M Text-to-SQL...")
    gemma_model, gemma_tokenizer = load_gemma3_model("outputs/gemma3-text2sql-stable")
    if gemma_model:
        gemma_predictions = []
        gemma_times = []
        gemma_correct = 0
        
        for question, expected_sql in test_data:
            prompt = f"Domanda: {question}\nSQL:"
            
            start_time = time.time()
            try:
                inputs = gemma_tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = gemma_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=gemma_tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                result = gemma_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                prediction = result.strip().split('\n')[0].strip()
                
                # Simple SQL comparison
                pred_normalized = ' '.join(prediction.replace(';', '').split()).upper()
                expected_normalized = ' '.join(expected_sql.replace(';', '').split()).upper()
                
                if pred_normalized in expected_normalized or expected_normalized in pred_normalized:
                    gemma_correct += 1
                
                gemma_predictions.append(prediction)
                gemma_times.append(time.time() - start_time)
                
            except Exception as e:
                print(f"Error with Gemma3 SQL prediction: {e}")
                gemma_predictions.append("error")
                gemma_times.append(time.time() - start_time)
        
        accuracy = gemma_correct / len(test_data)
        results['gemma3_270m'] = {
            'accuracy': accuracy,
            'precision': accuracy,
            'recall': accuracy,
            'f1': accuracy,
            'avg_inference_time': np.mean(gemma_times),
            'predictions': list(zip([q for q, _ in test_data], gemma_predictions))
        }
    
    return results

def main():
    print("🚀 Starting Comprehensive Model Evaluation with Detailed Metrics")
    print("=" * 80)
    
    # Evaluate Intent Classification
    intent_results = evaluate_intent_classification()
    
    # Evaluate Text-to-SQL
    sql_results = evaluate_text_to_sql()
    
    # Combine results
    all_results = {
        'intent_classification': intent_results,
        'text_to_sql': sql_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save detailed results
    output_path = Path("outputs/detailed_model_metrics.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("📊 COMPREHENSIVE MODEL COMPARISON WITH DETAILED METRICS")
    print("=" * 120)
    
    print("\n🎯 INTENT CLASSIFICATION RESULTS:")
    print("-" * 120)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Avg Time (ms)':<15}")
    print("-" * 120)
    
    for model_name, metrics in intent_results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<11.3f} {metrics['precision']:<11.3f} "
              f"{metrics['recall']:<11.3f} {metrics['f1']:<11.3f} {metrics['avg_inference_time']*1000:<14.1f}")
    
    print("\n🗄️ TEXT-TO-SQL RESULTS:")
    print("-" * 120)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Avg Time (ms)':<15}")
    print("-" * 120)
    
    for model_name, metrics in sql_results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<11.3f} {metrics['precision']:<11.3f} "
              f"{metrics['recall']:<11.3f} {metrics['f1']:<11.3f} {metrics['avg_inference_time']*1000:<14.1f}")
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()