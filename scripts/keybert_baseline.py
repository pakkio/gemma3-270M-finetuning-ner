#!/usr/bin/env python3
"""
KeyBERT baseline for hashtag generation comparison.
Official competitor for automatic keyword/hashtag extraction.
"""
import json
import time
from pathlib import Path
import argparse
try:
    from keybert import KeyBERT
    import spacy
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Missing dependencies. Install with: pip install keybert spacy sentence-transformers")
    print(f"Error: {e}")
    exit(1)

class KeyBERTHashtagger:
    """KeyBERT-based hashtag generator for comparison."""
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"🔄 Loading KeyBERT with model: {model_name}")
        
        # Load sentence transformer for Italian support
        self.sentence_model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(model=self.sentence_model)
        
        # Load Italian spaCy model for preprocessing if available
        try:
            self.nlp = spacy.load("it_core_news_sm")
        except OSError:
            print("⚠️  Italian spaCy model not found, using basic preprocessing")
            self.nlp = None
        
        print("✅ KeyBERT hashtagger ready")
    
    def preprocess_text(self, text):
        """Preprocess text for better keyword extraction."""
        if self.nlp:
            doc = self.nlp(text)
            # Keep only meaningful tokens (nouns, proper nouns, adjectives)
            meaningful_tokens = [token.lemma_ for token in doc 
                               if not token.is_stop and not token.is_punct 
                               and token.pos_ in ['NOUN', 'PROPN', 'ADJ']
                               and len(token.text) > 2]
            return ' '.join(meaningful_tokens)
        else:
            # Basic preprocessing
            import re
            text = re.sub(r'[^\w\s]', ' ', text)
            words = [word for word in text.split() if len(word) > 2]
            return ' '.join(words)
    
    def generate_hashtags(self, text, max_hashtags=8):
        """Generate hashtags using KeyBERT."""
        start_time = time.time()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract keywords using KeyBERT
        keywords = self.kw_model.extract_keywords(
            processed_text,
            keyphrase_ngram_range=(1, 2),  # Single words and bigrams
            stop_words='italian',
            top_k=max_hashtags * 2,  # Extract more to filter best ones
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.7  # Balance between relevance and diversity
        )
        
        # Convert keywords to hashtags
        hashtags = []
        for keyword, score in keywords[:max_hashtags]:
            # Convert to hashtag format
            hashtag = keyword.lower().replace(' ', '').replace('-', '')
            # Remove non-alphanumeric characters
            hashtag = ''.join(c for c in hashtag if c.isalnum())
            
            if len(hashtag) > 2:  # Only meaningful hashtags
                hashtags.append(f"#{hashtag}")
        
        generation_time = time.time() - start_time
        
        return hashtags, generation_time
    
    def extract_keywords_with_scores(self, text, max_keywords=15):
        """Extract keywords with relevance scores for analysis."""
        processed_text = self.preprocess_text(text)
        
        keywords = self.kw_model.extract_keywords(
            processed_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='italian',
            top_k=max_keywords,
            use_mmr=True,
            diversity=0.7
        )
        
        return keywords

def evaluate_hashtags_against_ground_truth(predicted, ground_truth):
    """Evaluate predicted hashtags against ground truth."""
    # Normalize hashtags (remove #, lowercase)
    pred_normalized = set(tag.lower().replace('#', '') for tag in predicted)
    true_normalized = set(tag.lower().replace('#', '') for tag in ground_truth)
    
    if not true_normalized:
        return 0.0, 0.0, 0.0  # No ground truth
    
    # Calculate metrics
    intersection = pred_normalized.intersection(true_normalized)
    
    precision = len(intersection) / len(pred_normalized) if pred_normalized else 0.0
    recall = len(intersection) / len(true_normalized) if true_normalized else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def run_keybert_evaluation(test_data_path, output_path=None):
    """Run KeyBERT evaluation on test dataset."""
    
    print("🔬 KEYBERT HASHTAG EVALUATION")
    print("=" * 50)
    
    # Load test data
    test_examples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                test_examples.append(data)
    
    print(f"📊 Loaded {len(test_examples)} test examples")
    
    # Initialize KeyBERT
    keybert = KeyBERTHashtagger()
    
    results = []
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_time = 0.0
    
    print("\n🔄 Processing examples...")
    
    for i, example in enumerate(test_examples, 1):
        text = example['document']
        ground_truth_hashtags = example['hashtags'].split()
        
        # Generate hashtags
        predicted_hashtags, generation_time = keybert.generate_hashtags(text)
        total_time += generation_time
        
        # Evaluate
        precision, recall, f1 = evaluate_hashtags_against_ground_truth(
            predicted_hashtags, ground_truth_hashtags
        )
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        result = {
            "example": i,
            "document": text,
            "ground_truth": ground_truth_hashtags,
            "predicted": predicted_hashtags,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "generation_time": generation_time
        }
        
        results.append(result)
        
        print(f"Example {i}/{len(test_examples)}: F1={f1:.3f}, Time={generation_time:.3f}s")
    
    # Calculate averages
    avg_precision = total_precision / len(test_examples)
    avg_recall = total_recall / len(test_examples)
    avg_f1 = total_f1 / len(test_examples)
    avg_time = total_time / len(test_examples)
    
    # Summary
    summary = {
        "model": "KeyBERT",
        "test_examples": len(test_examples),
        "metrics": {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1
        },
        "timing": {
            "total_time_seconds": total_time,
            "average_time_per_document": avg_time,
            "documents_per_second": 1.0 / avg_time if avg_time > 0 else 0.0
        },
        "results": results
    }
    
    print(f"\n📊 KEYBERT EVALUATION RESULTS")
    print("=" * 50)
    print(f"📈 Average Precision: {avg_precision:.3f}")
    print(f"📈 Average Recall: {avg_recall:.3f}")
    print(f"📈 Average F1 Score: {avg_f1:.3f}")
    print(f"⏱️  Average Time per Document: {avg_time:.3f}s")
    print(f"🚀 Documents per Second: {1.0/avg_time:.1f}")
    
    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_path}")
    
    return summary

def interactive_keybert_demo():
    """Interactive demo of KeyBERT hashtag generation."""
    keybert = KeyBERTHashtagger()
    
    print("\n🎯 KEYBERT HASHTAG GENERATOR - Interactive Demo")
    print("Enter Italian text (empty line to quit):")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n📝 Text: ").strip()
            if not text:
                break
            
            # Generate hashtags
            hashtags, gen_time = keybert.generate_hashtags(text)
            
            # Also show keywords with scores for analysis
            keywords = keybert.extract_keywords_with_scores(text)
            
            print(f"\n🏷️  Generated Hashtags: {' '.join(hashtags)}")
            print(f"⏱️  Generation time: {gen_time:.3f}s")
            print(f"\n🔍 Top Keywords with Scores:")
            for keyword, score in keywords[:5]:
                print(f"   {keyword}: {score:.3f}")
        
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")

def main():
    parser = argparse.ArgumentParser(description='KeyBERT Hashtag Generator and Evaluator')
    parser.add_argument('--test-data', help='Test data file (JSONL format)')
    parser.add_argument('--output', help='Output file for evaluation results')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive demo mode')
    parser.add_argument('--text', help='Single text to process')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_keybert_demo()
    
    elif args.text:
        keybert = KeyBERTHashtagger()
        hashtags, gen_time = keybert.generate_hashtags(args.text)
        print(f"🏷️  Hashtags: {' '.join(hashtags)}")
        print(f"⏱️  Time: {gen_time:.3f}s")
    
    elif args.test_data:
        if not Path(args.test_data).exists():
            print(f"❌ Test data file not found: {args.test_data}")
            return False
        
        run_keybert_evaluation(args.test_data, args.output)
    
    else:
        print("❌ Please specify --interactive, --text, or --test-data")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)