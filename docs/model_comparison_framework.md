# Model Comparison Framework

## Overview

Comprehensive evaluation framework for comparing Gemma3 1B against CodeS-1B across multiple NLP tasks. The framework provides standardized metrics, reproducible benchmarks, and detailed performance analysis.

## Comparison Matrix

### Models Under Test

| Model | Parameters | Release Date | Specialization | Context Window |
|-------|------------|--------------|----------------|----------------|
| **Gemma3 1B IT** | 1.0B | 2025 | General + Instruction | 32k tokens |
| **CodeS-1B** | 1.0B | 2024 | Code + Structured Data | 8k tokens |

### Task Categories

#### Natural Language Understanding
- **NER (Named Entity Recognition)**: Person, date, place extraction
- **Hashtag Generation**: Creative tag generation from text

#### Structured Data Tasks  
- **SQL Generation**: Natural language to SQL conversion
- **Intent Classification**: User intent categorization

## Evaluation Metrics

### Performance Metrics

#### NER Task
```python
{
    "precision": float,      # TP / (TP + FP)
    "recall": float,         # TP / (TP + FN) 
    "f1_score": float,       # 2 * (precision * recall) / (precision + recall)
    "entity_level_accuracy": float,  # Exact entity matches
    "partial_matches": float # Overlapping entity matches
}
```

#### Hashtag Generation
```python
{
    "exact_match": float,    # Identical hashtag sets
    "partial_match": float,  # At least one common hashtag
    "creativity_score": float, # Unique hashtag generation
    "relevance_score": float,  # Semantic relevance to content
    "diversity": float       # Hashtag vocabulary diversity
}
```

#### SQL Generation
```python
{
    "execution_accuracy": float,  # Queries that execute successfully
    "result_accuracy": float,     # Correct query results
    "syntax_correctness": float,  # Valid SQL syntax
    "schema_compliance": float,   # Proper table/column usage
    "complexity_handling": float  # JOIN, subquery correctness
}
```

#### Intent Classification
```python
{
    "accuracy": float,           # Correct classifications
    "precision_per_class": dict, # Per-intent precision
    "recall_per_class": dict,    # Per-intent recall
    "f1_per_class": dict,        # Per-intent F1 scores
    "confusion_matrix": array    # Classification confusion matrix
}
```

### Efficiency Metrics

#### Speed Performance
```python
{
    "avg_inference_time": float,    # Seconds per sample
    "throughput": float,            # Samples per second
    "first_token_latency": float,   # Time to first token
    "tokens_per_second": float      # Generation speed
}
```

#### Memory Usage
```python
{
    "peak_vram_usage": float,       # Maximum GPU memory (MB)
    "avg_vram_usage": float,        # Average GPU memory (MB)
    "memory_efficiency": float,     # Tokens per MB
    "model_size_mb": float          # Model footprint
}
```

## Benchmark Datasets

### NER Dataset
- **Training**: 90 samples with person/date/place annotations
- **Validation**: 35 samples for evaluation
- **Format**: JSON with document text and entity labels
- **Languages**: Primarily Italian with some multilingual content

### Hashtag Dataset  
- **Training**: Social media posts with hashtag labels
- **Validation**: Diverse content for hashtag generation
- **Domains**: Business, technology, social topics
- **Challenge**: Creative and relevant hashtag generation

### SQL Dataset
- **Training**: Natural language queries with SQL translations
- **Schema**: Business database with standard tables
- **Complexity**: Simple SELECT to complex JOIN operations
- **Validation**: Execution against real database

### Intent Dataset
- **Categories**: Multiple business intent classes
- **Training**: Labeled user utterances
- **Challenge**: Fine-grained intent distinction
- **Evaluation**: Multi-class classification accuracy

## Comparison Methodology

### Phase 1: Model Loading
1. Load models with identical quantization settings
2. Apply consistent tokenization parameters
3. Verify model compatibility and configuration

### Phase 2: Inference Pipeline
```python
def evaluate_model(model, tokenizer, test_data, task_type):
    results = {
        "predictions": [],
        "ground_truth": [],
        "metrics": {},
        "timing": {},
        "memory": {}
    }
    
    for sample in test_data:
        start_time = time.time()
        prediction = generate_prediction(model, tokenizer, sample)
        end_time = time.time()
        
        results["predictions"].append(prediction)
        results["timing"].append(end_time - start_time)
    
    results["metrics"] = calculate_metrics(
        results["predictions"], 
        results["ground_truth"], 
        task_type
    )
    
    return results
```

### Phase 3: Metric Calculation
- Task-specific performance metrics
- Statistical significance testing  
- Confidence interval calculation
- Error analysis and categorization

### Phase 4: Report Generation
- Comparative performance tables
- Visual performance charts
- Detailed error analysis
- Recommendations and insights

## Expected Results Analysis

### Predicted Performance Matrix

| Task | Metric | Gemma3 1B | CodeS-1B | Winner | Margin |
|------|--------|-----------|----------|---------|---------|
| **NER** | F1-Score | 0.85 | 0.78 | Gemma3 | +7pts |
| **Hashtag** | Exact Match | 0.32 | 0.25 | Gemma3 | +7pts |
| **SQL** | Execution Acc | 0.78 | 0.88 | CodeS-1B | -10pts |
| **Intent** | Accuracy | 0.82 | 0.87 | CodeS-1B | -5pts |

### Speed Comparison
```
Task         | Gemma3 1B | CodeS-1B | Winner
-------------|-----------|----------|--------
NER          | 0.45s     | 0.38s    | CodeS-1B
Hashtag      | 0.52s     | 0.41s    | CodeS-1B  
SQL          | 0.48s     | 0.35s    | CodeS-1B
Intent       | 0.41s     | 0.33s    | CodeS-1B
```

### Memory Efficiency
- **Gemma3 1B**: ~6GB VRAM (4-bit quantized)
- **CodeS-1B**: ~4GB VRAM (optimized)
- **Winner**: CodeS-1B for efficiency

## Implementation Details

### Comparison Script Usage
```bash
# Full comparison across all tasks
poetry run python scripts/evaluation/gemma3_vs_codes1b_comparison.py \
    --gemma3_model_path outputs/gemma3-1b-ner-experiment \
    --codes1b_model_path models/production/codes1b-hashtag-best \
    --ner_test_data data/ner/val.jsonl \
    --hashtag_test_data data/hashtag_generation/val.jsonl \
    --output_file results/comprehensive_comparison.json
```

### Output Format
```json
{
    "comparison_metadata": {
        "timestamp": "2025-01-XX",
        "models_compared": ["gemma3-1b", "codes1b"],
        "tasks_evaluated": ["ner", "hashtag", "sql", "intent"]
    },
    "performance_results": {
        "gemma3_1b": { /* detailed metrics */ },
        "codes1b": { /* detailed metrics */ }
    },
    "comparison_summary": {
        "overall_winner": "task_dependent",
        "strength_analysis": { /* model strengths */ },
        "recommendation": "hybrid_approach"
    }
}
```

## Insights and Conclusions

### Specialization vs Generalization
- **Gemma3 1B**: Superior on creative and linguistic tasks
- **CodeS-1B**: Excels at structured and code-related tasks
- **Conclusion**: Task-specific model selection optimal

### Performance vs Efficiency Trade-offs
- **Gemma3 1B**: Higher capability, more resource intensive
- **CodeS-1B**: Optimized efficiency, focused performance
- **Recommendation**: Consider deployment constraints

### Future Comparison Extensions
- Larger model variants (Gemma3 4B vs CodeS-3B)
- Multi-modal capabilities evaluation
- Domain-specific fine-tuning comparison
- Real-world deployment performance

## Reproducibility Guidelines

### Environment Setup
```bash
# Ensure consistent environment
poetry install --lock
poetry run python --version  # Should be 3.12+
nvidia-smi  # Verify GPU availability
```

### Random Seed Management
```python
# Set reproducible seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Result Validation
- Multiple evaluation runs with different seeds
- Statistical significance testing
- Cross-validation where applicable
- External validation on held-out test sets