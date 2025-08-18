#!/usr/bin/env python3
"""
Comprehensive hashtag generation comparison:
Gemma3 Fine-tuned vs BERT Fine-tuned vs KeyBERT Baseline
"""
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Import KeyBERT functionality
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

class HashtagComparison:
    """Comprehensive hashtag generation comparison framework."""
    
    def __init__(self, test_data_path):
        self.test_data_path = Path(test_data_path)
        self.test_examples = self._load_test_data()
        
        # Models will be loaded on demand
        self.gemma_model = None
        self.gemma_tokenizer = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.keybert_model = None
        
        print(f"📊 Loaded {len(self.test_examples)} test examples for comparison")
    
    def _load_test_data(self):
        """Load test data from JSONL."""
        examples = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append({
                        'document': data['document'],
                        'ground_truth_hashtags': data['hashtags'].split()
                    })
        return examples
    
    def load_gemma_hashtagger(self, model_path):
        """Load Gemma3 hashtagger model."""
        print("🔄 Loading Gemma3 hashtagger...")
        
        try:
            base_model_path = "google/gemma-3-270m"
            
            # Load base model - force CPU for stability
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map=None,  # Force CPU
                trust_remote_code=True,
                attn_implementation="eager"  # Use eager attention
            )
            
            # Load PEFT adapter
            self.gemma_model = PeftModel.from_pretrained(self.gemma_model, model_path)
            self.gemma_model.eval()
            
            # Load tokenizer
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            if self.gemma_tokenizer.pad_token is None:
                self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
            
            print("✅ Gemma3 hashtagger loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Gemma3 hashtagger: {e}")
            return False
    
    def load_bert_hashtagger(self, model_path):
        """Load BERT hashtagger model."""
        print("🔄 Loading BERT hashtagger...")
        
        try:
            self.bert_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.bert_model.eval()
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.bert_tokenizer.pad_token is None:
                self.bert_tokenizer.pad_token = self.bert_tokenizer.eos_token
            
            print("✅ BERT hashtagger loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load BERT hashtagger: {e}")
            return False
    
    def load_keybert(self):
        """Load KeyBERT baseline."""
        if not KEYBERT_AVAILABLE:
            print("❌ KeyBERT not available. Install with: pip install keybert sentence-transformers")
            return False
        
        print("🔄 Loading KeyBERT baseline...")
        
        try:
            sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.keybert_model = KeyBERT(model=sentence_model)
            
            print("✅ KeyBERT loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load KeyBERT: {e}")
            return False
    
    def generate_gemma_hashtags(self, text):
        """Generate hashtags using Gemma3 model."""
        if not self.gemma_model:
            return [], 0.0
        
        # Use simpler prompt format like training
        prompt = f"Document: {text}\nHashtags:"
        
        start_time = time.time()
        
        try:
            inputs = self.gemma_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,  # Shorter to avoid issues
                truncation=True,
                padding=True
            )
            
            # Force CPU for stability
            inputs = {k: v.cpu() for k, v in inputs.items()}
            self.gemma_model = self.gemma_model.cpu()
            
            with torch.no_grad():
                # Use greedy decoding for more consistent results
                outputs = self.gemma_model.generate(
                    **inputs,
                    max_new_tokens=40,  # Slightly longer
                    do_sample=False,    # Greedy decoding
                    pad_token_id=self.gemma_tokenizer.eos_token_id,
                    eos_token_id=self.gemma_tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Lower repetition penalty
                    early_stopping=True
                )
            
            # Decode response
            response = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            
            # Extract and clean hashtags
            hashtag_line = generated_text.split('\n')[0].strip()
            hashtags = []
            for tag in hashtag_line.split():
                # Clean and validate hashtag
                clean_tag = tag.strip()
                if clean_tag.startswith('#') and len(clean_tag) > 1:
                    # Remove emojis and invalid characters, keep only hashtag part
                    clean_tag = clean_tag.split()[0] if ' ' in clean_tag else clean_tag
                    # Remove trailing punctuation except for valid hashtag characters
                    clean_tag = ''.join(c for c in clean_tag if c.isalnum() or c == '#')
                    if len(clean_tag) > 1 and clean_tag != '#':
                        hashtags.append(clean_tag)
            
            return hashtags, generation_time
            
        except Exception as e:
            print(f"Gemma3 generation error: {e}")
            return [], time.time() - start_time
    
    def generate_bert_hashtags(self, text):
        """Generate hashtags using BERT model."""
        if not self.bert_model:
            return [], 0.0
        
        input_text = f"Genera hashtags per: {text}"
        
        start_time = time.time()
        
        inputs = self.bert_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.bert_tokenizer.pad_token_id,
                eos_token_id=self.bert_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.bert_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response.replace(input_text, "").strip()
        
        generation_time = time.time() - start_time
        
        # Extract hashtags
        hashtag_line = generated_text.split('\\n')[0].strip()
        hashtags = [tag.strip() for tag in hashtag_line.split() if tag.startswith('#')]
        
        return hashtags, generation_time
    
    def generate_keybert_hashtags(self, text):
        """Generate hashtags using KeyBERT."""
        if not self.keybert_model:
            return [], 0.0
        
        start_time = time.time()
        
        # Extract keywords using KeyBERT - fix parameter compatibility
        try:
            keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='italian',
                nr_candidates=20,
                use_mmr=True,
                diversity=0.7
            )
            # Take top 8
            keywords = keywords[:8]
        except TypeError:
            # Fallback for older KeyBERT versions
            keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='italian',
                use_mmr=True,
                diversity=0.7
            )[:8]
        
        # Convert keywords to hashtags
        hashtags = []
        for keyword, score in keywords:
            hashtag = keyword.lower().replace(' ', '').replace('-', '')
            hashtag = ''.join(c for c in hashtag if c.isalnum())
            
            if len(hashtag) > 2:
                hashtags.append(f"#{hashtag}")
        
        generation_time = time.time() - start_time
        
        return hashtags, generation_time
    
    def evaluate_hashtags(self, predicted, ground_truth):
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
    
    def run_comprehensive_comparison(self, gemma_path=None, bert_path=None, use_keybert=True):
        """Run comprehensive comparison across all models."""
        
        print("🏆 COMPREHENSIVE HASHTAG GENERATION COMPARISON")
        print("=" * 70)
        
        # Load models
        models_loaded = {}
        
        if gemma_path and Path(gemma_path).exists():
            models_loaded['gemma'] = self.load_gemma_hashtagger(gemma_path)
        else:
            models_loaded['gemma'] = False
            print("⚠️  Gemma3 model not available")
        
        if bert_path and Path(bert_path).exists():
            models_loaded['bert'] = self.load_bert_hashtagger(bert_path)
        else:
            models_loaded['bert'] = False
            print("⚠️  BERT model not available")
        
        if use_keybert:
            models_loaded['keybert'] = self.load_keybert()
        else:
            models_loaded['keybert'] = False
        
        # Initialize results
        results = {
            'gemma': {'predictions': [], 'times': [], 'metrics': []},
            'bert': {'predictions': [], 'times': [], 'metrics': []},
            'keybert': {'predictions': [], 'times': [], 'metrics': []}
        }
        
        print(f"\\n🔄 Processing {len(self.test_examples)} test examples...")
        
        # Process each test example
        for i, example in enumerate(self.test_examples, 1):
            text = example['document']
            ground_truth = example['ground_truth_hashtags']
            
            print(f"\\rExample {i}/{len(self.test_examples)}", end="", flush=True)
            
            # Test Gemma3
            if models_loaded['gemma']:
                hashtags, gen_time = self.generate_gemma_hashtags(text)
                precision, recall, f1 = self.evaluate_hashtags(hashtags, ground_truth)
                
                results['gemma']['predictions'].append(hashtags)
                results['gemma']['times'].append(gen_time)
                results['gemma']['metrics'].append((precision, recall, f1))
            
            # Test BERT
            if models_loaded['bert']:
                hashtags, gen_time = self.generate_bert_hashtags(text)
                precision, recall, f1 = self.evaluate_hashtags(hashtags, ground_truth)
                
                results['bert']['predictions'].append(hashtags)
                results['bert']['times'].append(gen_time)
                results['bert']['metrics'].append((precision, recall, f1))
            
            # Test KeyBERT
            if models_loaded['keybert']:
                hashtags, gen_time = self.generate_keybert_hashtags(text)
                precision, recall, f1 = self.evaluate_hashtags(hashtags, ground_truth)
                
                results['keybert']['predictions'].append(hashtags)
                results['keybert']['times'].append(gen_time)
                results['keybert']['metrics'].append((precision, recall, f1))
        
        print()  # New line after progress
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results, models_loaded)
        
        # Display results
        self._display_comparison_results(summary)
        
        return summary
    
    def _calculate_summary_stats(self, results, models_loaded):
        """Calculate summary statistics for all models."""
        summary = {}
        
        for model_name in ['gemma', 'bert', 'keybert']:
            if not models_loaded[model_name] or not results[model_name]['metrics']:
                continue
            
            metrics = results[model_name]['metrics']
            times = results[model_name]['times']
            
            # Calculate averages
            avg_precision = np.mean([m[0] for m in metrics])
            avg_recall = np.mean([m[1] for m in metrics])
            avg_f1 = np.mean([m[2] for m in metrics])
            
            total_time = sum(times)
            avg_time = np.mean(times)
            
            summary[model_name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'total_time': total_time,
                'avg_time_per_example': avg_time,
                'examples_processed': len(metrics),
                'examples_per_second': 1.0 / avg_time if avg_time > 0 else 0.0
            }
        
        return summary
    
    def _display_comparison_results(self, summary):
        """Display comparison results in a formatted table."""
        
        print("\\n🏆 HASHTAG GENERATION COMPARISON RESULTS")
        print("=" * 80)
        
        if not summary:
            print("❌ No models were successfully loaded and tested")
            return
        
        # Header
        print(f"{'Model':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Avg Time':<10} {'Ex/sec':<8}")
        print("-" * 80)
        
        # Sort by F1 score
        sorted_models = sorted(summary.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for model_name, stats in sorted_models:
            model_display = {
                'gemma': 'Gemma3',
                'bert': 'BERT',
                'keybert': 'KeyBERT'
            }.get(model_name, model_name)
            
            print(f"{model_display:<12} {stats['precision']:<10.3f} {stats['recall']:<10.3f} "
                  f"{stats['f1']:<10.3f} {stats['avg_time_per_example']:<10.3f} "
                  f"{stats['examples_per_second']:<8.1f}")
        
        # Winner announcement
        if sorted_models:
            winner = sorted_models[0]
            winner_name = {'gemma': 'Gemma3 Fine-tuned', 'bert': 'BERT Fine-tuned', 'keybert': 'KeyBERT'}.get(winner[0])
            print(f"\\n🥇 **WINNER: {winner_name}** with F1 Score: {winner[1]['f1']:.3f}")
        
        # Performance insights
        print("\\n📊 KEY INSIGHTS:")
        for model_name, stats in summary.items():
            model_display = {'gemma': 'Gemma3', 'bert': 'BERT', 'keybert': 'KeyBERT'}.get(model_name)
            print(f"   {model_display}: {stats['examples_per_second']:.1f} examples/sec, {stats['f1']:.3f} F1")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Hashtag Generation Comparison')
    parser.add_argument('--test-data', default='data/val_hashtagger.jsonl',
                       help='Test data file (JSONL format)')
    parser.add_argument('--gemma-model', default='outputs/simple-gemma3-hashtagger',
                       help='Path to Gemma3 hashtagger model')
    parser.add_argument('--bert-model', default='outputs/bert-hashtagger',
                       help='Path to BERT hashtagger model')
    parser.add_argument('--skip-keybert', action='store_true',
                       help='Skip KeyBERT baseline')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Check test data
    if not Path(args.test_data).exists():
        print(f"❌ Test data file not found: {args.test_data}")
        return False
    
    # Initialize comparison
    comparison = HashtagComparison(args.test_data)
    
    # Run comprehensive comparison
    summary = comparison.run_comprehensive_comparison(
        gemma_path=args.gemma_model,
        bert_path=args.bert_model,
        use_keybert=not args.skip_keybert
    )
    
    # Save results if requested
    if args.output and summary:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\\n💾 Results saved to: {args.output}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)