#!/usr/bin/env python3
"""
Evaluate Text-to-SQL models with comprehensive metrics
Compares CodeT5 vs Gemma3 performance on Italian business queries
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from peft import PeftModel
import sqlparse
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLEvaluator:
    """Comprehensive SQL query evaluator"""
    
    def __init__(self):
        self.exact_matches = 0
        self.syntax_valid = 0
        self.semantic_matches = 0
        self.total_evaluated = 0
        
    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison"""
        # Remove extra whitespace and normalize case
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Parse and format with sqlparse
        try:
            parsed = sqlparse.parse(sql)[0]
            formatted = sqlparse.format(
                str(parsed), 
                reindent=True, 
                keyword_case='upper',
                identifier_case='lower'
            )
            return formatted.strip()
        except:
            return sql.upper().strip()
    
    def check_syntax_validity(self, sql: str) -> bool:
        """Check if SQL has valid syntax"""
        try:
            parsed = sqlparse.parse(sql)
            return len(parsed) > 0 and parsed[0].tokens
        except:
            return False
    
    def extract_sql_components(self, sql: str) -> Dict:
        """Extract key components from SQL query"""
        components = {
            'select_fields': [],
            'tables': [],
            'where_conditions': [],
            'joins': [],
            'group_by': [],
            'order_by': [],
            'aggregations': []
        }
        
        sql_upper = sql.upper()
        
        # Extract SELECT fields
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if select_match:
            fields = [f.strip() for f in select_match.group(1).split(',')]
            components['select_fields'] = fields
        
        # Extract tables
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            components['tables'].append(from_match.group(1))
        
        # Extract WHERE conditions
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', sql_upper, re.DOTALL)
        if where_match:
            conditions = where_match.group(1).strip()
            components['where_conditions'].append(conditions)
        
        # Extract JOINs
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        components['joins'] = join_matches
        
        # Extract GROUP BY
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|$)', sql_upper)
        if group_match:
            components['group_by'] = [f.strip() for f in group_match.group(1).split(',')]
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', sql_upper)
        if order_match:
            components['order_by'] = [f.strip() for f in order_match.group(1).split(',')]
        
        # Check for aggregations
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
        for func in agg_functions:
            if func in sql_upper:
                components['aggregations'].append(func)
        
        return components
    
    def semantic_similarity(self, pred_sql: str, true_sql: str) -> float:
        """Calculate semantic similarity between SQL queries"""
        pred_components = self.extract_sql_components(pred_sql)
        true_components = self.extract_sql_components(true_sql)
        
        scores = []
        
        # Compare each component
        for component in pred_components:
            pred_items = set(pred_components[component])
            true_items = set(true_components[component])
            
            if not pred_items and not true_items:
                scores.append(1.0)  # Both empty
            elif not pred_items or not true_items:
                scores.append(0.0)  # One empty, one not
            else:
                intersection = len(pred_items & true_items)
                union = len(pred_items | true_items)
                scores.append(intersection / union if union > 0 else 0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def evaluate_prediction(self, predicted: str, ground_truth: str) -> Dict:
        """Evaluate a single prediction"""
        result = {
            'exact_match': False,
            'syntax_valid': False,
            'semantic_similarity': 0.0,
            'normalized_match': False
        }
        
        # Clean predictions
        predicted = predicted.strip()
        ground_truth = ground_truth.strip()
        
        # Check syntax validity
        result['syntax_valid'] = self.check_syntax_validity(predicted)
        
        # Check exact match
        result['exact_match'] = predicted.lower() == ground_truth.lower()
        
        # Check normalized match
        try:
            pred_norm = self.normalize_sql(predicted)
            true_norm = self.normalize_sql(ground_truth)
            result['normalized_match'] = pred_norm == true_norm
        except:
            pass
        
        # Calculate semantic similarity
        try:
            result['semantic_similarity'] = self.semantic_similarity(predicted, ground_truth)
        except:
            result['semantic_similarity'] = 0.0
        
        return result

class ModelPredictor:
    """Base class for model predictions"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        raise NotImplementedError
        
    def predict(self, question: str) -> str:
        """Generate SQL prediction for a question"""
        raise NotImplementedError

class CodeT5Predictor(ModelPredictor):
    """CodeT5 model predictor"""
    
    def load_model(self):
        logger.info(f"Loading CodeT5 model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    def predict(self, question: str) -> str:
        prompt = f"Converti la seguente domanda italiana in una query SQL: {question}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=3,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction.strip()

class Gemma3Predictor(ModelPredictor):
    """Gemma3 model predictor"""
    
    def load_model(self):
        logger.info(f"Loading Gemma3 model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def predict(self, question: str) -> str:
        # Match the exact training format
        prompt = f"""### Istruzione:
Converti la seguente domanda italiana in una query SQL.

### Domanda:
{question}

### SQL:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use working parameters from hashtag model test
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part (after the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Debug log
        logger.info(f"Raw prediction: '{prediction}' for question: '{question[:50]}...'")
        
        # Clean up the prediction - extract SQL until "### Fine" or newline
        prediction = prediction.strip()
        
        # Split on ### markers and take the first part
        if '###' in prediction:
            prediction = prediction.split('###')[0].strip()
        
        # Remove newlines and extra whitespace
        prediction = ' '.join(prediction.split())
        
        # If still empty, return a basic SELECT statement
        if not prediction:
            logger.warning(f"Empty prediction after cleaning for question: {question}")
            return "SELECT 1;"
        
        logger.info(f"Final prediction: '{prediction}'")
        return prediction

def evaluate_model(predictor: ModelPredictor, eval_data: List[Dict], model_name: str) -> Dict:
    """Evaluate a model on the evaluation dataset"""
    logger.info(f"Evaluating {model_name}...")
    
    # Check for existing checkpoint
    checkpoint_path = f"outputs/{model_name.lower()}_eval_checkpoint.json"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading existing checkpoint for {model_name}")
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    
    predictor.load_model()
    evaluator = SQLEvaluator()
    
    results = []
    start_time = time.time()
    
    for i, example in enumerate(eval_data):
        if i % 10 == 0:
            logger.info(f"Processing example {i+1}/{len(eval_data)}")
        
        question = example['question']
        ground_truth = example['sql']
        
        try:
            prediction = predictor.predict(question)
            eval_result = evaluator.evaluate_prediction(prediction, ground_truth)
            
            # Clean eval_result of any non-serializable objects
            clean_result = {}
            for key, value in eval_result.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    clean_result[key] = value
                else:
                    clean_result[key] = str(value)
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                **clean_result
            })
            
        except Exception as e:
            logger.warning(f"Error evaluating example {i}: {e}")
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'prediction': "",
                'exact_match': False,
                'syntax_valid': False,
                'semantic_similarity': 0.0,
                'normalized_match': False,
                'error': str(e)
            })
        
        # Save intermediate checkpoint every 10 examples
        if (i + 1) % 10 == 0 or i == len(eval_data) - 1:
            temp_results = {
                'model_name': model_name,
                'partial_results': results,
                'processed_examples': i + 1,
                'total_examples': len(eval_data)
            }
            with open(f"outputs/{model_name.lower()}_temp_checkpoint.json", 'w') as f:
                json.dump(temp_results, f, indent=2, ensure_ascii=False)
    
    evaluation_time = time.time() - start_time
    
    # Calculate aggregate metrics
    total_examples = len(results)
    exact_matches = sum(1 for r in results if r.get('exact_match', False))
    syntax_valid = sum(1 for r in results if r.get('syntax_valid', False))
    normalized_matches = sum(1 for r in results if r.get('normalized_match', False))
    avg_semantic_similarity = sum(r.get('semantic_similarity', 0) for r in results) / total_examples
    
    metrics = {
        'model_name': model_name,
        'total_examples': total_examples,
        'exact_match_accuracy': exact_matches / total_examples,
        'syntax_validity_rate': syntax_valid / total_examples,
        'normalized_match_accuracy': normalized_matches / total_examples,
        'semantic_similarity_avg': avg_semantic_similarity,
        'evaluation_time_seconds': evaluation_time,
        'examples_per_second': total_examples / evaluation_time,
        'detailed_results': results
    }
    
    # Save final checkpoint
    checkpoint_path = f"outputs/{model_name.lower()}_eval_checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation checkpoint for {model_name}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL models")
    parser.add_argument("--val_path", type=str, default="data/val_text2sql.jsonl",
                       help="Path to validation data")
    parser.add_argument("--codet5_path", type=str, default="outputs/codet5-text2sql",
                       help="Path to fine-tuned CodeT5 model")
    parser.add_argument("--gemma3_path", type=str, default="outputs/gemma3-text2sql",
                       help="Path to fine-tuned Gemma3 model")
    parser.add_argument("--output_path", type=str, default="outputs/text2sql_comparison.json",
                       help="Path to save comparison results")
    
    args = parser.parse_args()
    
    logger.info("Starting Text-to-SQL model evaluation")
    
    # Load validation data
    logger.info(f"Loading validation data from {args.val_path}")
    val_dataset = load_dataset("json", data_files=args.val_path, split="train")
    eval_data = list(val_dataset)
    
    logger.info(f"Loaded {len(eval_data)} validation examples")
    
    # Evaluate models
    results = {}
    
    # Evaluate CodeT5
    if Path(args.codet5_path).exists():
        codet5_predictor = CodeT5Predictor(args.codet5_path)
        results['codet5'] = evaluate_model(codet5_predictor, eval_data, "CodeT5")
    else:
        logger.warning(f"CodeT5 model not found at {args.codet5_path}")
    
    # Evaluate Gemma3
    if Path(args.gemma3_path).exists():
        gemma3_predictor = Gemma3Predictor(args.gemma3_path)
        results['gemma3'] = evaluate_model(gemma3_predictor, eval_data, "Gemma3")
    else:
        logger.warning(f"Gemma3 model not found at {args.gemma3_path}")
    
    # Create comparison summary
    if len(results) >= 2:
        comparison = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'validation_examples': len(eval_data),
            'models_compared': list(results.keys()),
            'metrics_comparison': {},
            'detailed_results': results
        }
        
        # Compare key metrics
        metrics_to_compare = [
            'exact_match_accuracy',
            'syntax_validity_rate', 
            'normalized_match_accuracy',
            'semantic_similarity_avg',
            'evaluation_time_seconds',
            'examples_per_second'
        ]
        
        for metric in metrics_to_compare:
            comparison['metrics_comparison'][metric] = {
                model: results[model].get(metric, 0) for model in results
            }
        
        # Determine winner
        codet5_score = results.get('codet5', {}).get('semantic_similarity_avg', 0)
        gemma3_score = results.get('gemma3', {}).get('semantic_similarity_avg', 0)
        
        if codet5_score > gemma3_score:
            winner = 'codet5'
            margin = codet5_score - gemma3_score
        elif gemma3_score > codet5_score:
            winner = 'gemma3'
            margin = gemma3_score - codet5_score
        else:
            winner = 'tie'
            margin = 0
        
        comparison['winner'] = {
            'model': winner,
            'margin': margin,
            'metric_used': 'semantic_similarity_avg'
        }
        
        # Save results
        logger.info(f"Saving comparison results to {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        with open(args.output_path, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
        for model_name, model_results in results.items():
            logger.info(f"\n{model_name.upper()} Results:")
            logger.info(f"  Exact Match Accuracy: {model_results['exact_match_accuracy']:.1%}")
            logger.info(f"  Syntax Validity Rate: {model_results['syntax_validity_rate']:.1%}")
            logger.info(f"  Semantic Similarity: {model_results['semantic_similarity_avg']:.3f}")
            logger.info(f"  Evaluation Time: {model_results['evaluation_time_seconds']:.1f}s")
        
        logger.info(f"\nWINNER: {winner.upper()} (margin: {margin:.3f})")
        
    else:
        logger.error("Need at least 2 models to compare")

if __name__ == "__main__":
    main()