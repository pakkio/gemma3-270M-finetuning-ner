# Gemma3 1B Fine-tuning and Comparison Guide

## Overview

This guide documents the Gemma3 1B fine-tuning framework and comparison system with CodeS-1B models. The framework supports efficient training with 4-bit quantization and LoRA adapters for consumer GPUs.

## Model Information

### Gemma 3 1B (2025)
- **Parameters**: 1 billion (actual 1B, not 2B)
- **Context Window**: 32,768 tokens (vs 8k in Gemma 2)
- **Languages**: 140+ supported languages
- **Training Data**: 2 trillion tokens with August 2024 cutoff
- **Architecture**: Text-only (non-multimodal)
- **Model ID**: `google/gemma-3-1b-it` (instruction-tuned)

### Key Advantages Over Previous Models
- Larger context window for better document understanding
- Extensive multilingual support ideal for NER tasks
- Recent training data (2024) vs older models
- Optimized instruction-following capabilities

## Framework Components

### Training Scripts
- `scripts/training/train_gemma3_1b_ner.py` - NER task fine-tuning
- `scripts/training/train_gemma3_1b_hashtag.py` - Hashtag generation training
- `scripts/run_gemma3_1b_experiments.py` - Automated experiment runner

### Evaluation Scripts
- `scripts/evaluation/gemma3_vs_codes1b_comparison.py` - Comprehensive model comparison

### Configuration
- 4-bit quantization with BitsAndBytesConfig
- LoRA fine-tuning (r=16, alpha=32)
- Optimized for 8GB+ VRAM GPUs

## Hardware Requirements

### Minimum Requirements (with 4-bit quantization)
- **GPU Memory**: 6GB VRAM minimum
- **Recommended**: 8GB+ VRAM for optimal performance
- **System RAM**: 16GB+ recommended

### Performance Expectations
| Configuration | VRAM Usage | Batch Size | Speed | Quality |
|---------------|------------|------------|-------|---------|
| 4-bit Quantized | ~6GB | 8-12 | 3x faster | ~98% quality |
| Full Precision | ~12GB | 2-4 | 1x baseline | 100% quality |

## Setup Instructions

### 1. Prerequisites
```bash
# Install dependencies
poetry install

# HuggingFace authentication (required)
poetry run huggingface-cli login
```

### 2. Enable Model Access
Visit https://huggingface.co/google/gemma-3-1b-it and request access to the Gemma 3 license.

### 3. Data Preparation
Ensure the following datasets exist:
- `data/ner/train.jsonl` - NER training data
- `data/ner/val.jsonl` - NER validation data
- `data/hashtag_generation/train.jsonl` - Hashtag training data
- `data/hashtag_generation/val.jsonl` - Hashtag validation data

## Usage Examples

### Complete Experiment Pipeline
```bash
# Run full experiment (NER + hashtag + comparison)
poetry run python scripts/run_gemma3_1b_experiments.py --step all
```

### Individual Training Steps
```bash
# NER training only
poetry run python scripts/run_gemma3_1b_experiments.py --step ner

# Hashtag training only
poetry run python scripts/run_gemma3_1b_experiments.py --step hashtag

# Comparison only (requires trained models)
poetry run python scripts/run_gemma3_1b_experiments.py --step comparison
```

### Direct Training Script Usage
```bash
# NER training with custom parameters
poetry run python scripts/training/train_gemma3_1b_ner.py \
    --model_id google/gemma-3-1b-it \
    --train_path data/ner/train.jsonl \
    --val_path data/ner/val.jsonl \
    --output_dir outputs/gemma3-1b-ner-custom \
    --lr 2e-4 \
    --epochs 3 \
    --batch_size 8 \
    --use_4bit
```

## Training Configuration

### Default Parameters
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Batch Size**: 4 (with 4x gradient accumulation)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05
- **Max Sequence Length**: 1024 (NER), 512 (hashtag)

### Quantization Settings
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

## Expected Performance Comparison

### Predicted Winners by Task
| Task | Gemma3 1B | CodeS-1B | Reason |
|------|-----------|----------|---------|
| **NER** | ✅ Winner | | Better linguistic understanding |
| **Hashtag** | ✅ Winner | | Superior creativity & context |
| **SQL** | | ✅ Winner | Code specialization |
| **Intent** | | ✅ Winner | Structured classification |

### Performance Metrics
- **NER**: F1-score, Precision, Recall
- **Hashtag**: Exact match, Partial match, Creativity score
- **Speed**: Inference time per sample
- **Memory**: VRAM usage during training/inference

## Output Structure

### Training Outputs
```
outputs/
├── gemma3-1b-ner-experiment/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── training_config.json
│   └── inference_template.txt
└── gemma3-1b-hashtag-experiment/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── training_config.json
    └── inference_template.txt
```

### Results Directory
```
results/
├── gemma3_1b_vs_codes1b_final_comparison.json
└── gemma3_1b_experiment_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

### Common Issues

#### HuggingFace Authentication
```bash
# Error: 401 Unauthorized
# Solution: Login and request access
poetry run huggingface-cli login
```

#### Out of Memory Errors
```bash
# Reduce batch size and increase gradient accumulation
--batch_size 2 --grad_accum 8
```

#### Slow Training
```bash
# Ensure 4-bit quantization is enabled
--use_4bit
```

## Comparison Methodology

The comparison framework evaluates models across multiple dimensions:

1. **Task Performance**: Domain-specific metrics (F1, accuracy)
2. **Speed**: Inference time and throughput
3. **Memory Efficiency**: VRAM usage patterns
4. **Quality**: Output consistency and correctness

### Evaluation Process
1. Load both models with identical configurations
2. Run inference on standardized test sets
3. Calculate performance metrics
4. Generate comprehensive comparison report

## Future Enhancements

### Planned Features
- Support for larger Gemma 3 variants (4B, 27B)
- Multi-GPU training support
- Advanced quantization options (8-bit, mixed precision)
- Integration with evaluation frameworks

### Optimization Opportunities
- Flash Attention 2 integration
- Gradient checkpointing refinements
- Custom data collators for specific tasks
- Hyperparameter optimization automation

## References

- [Gemma 3 Model Card](https://huggingface.co/google/gemma-3-1b-it)
- [Gemma 3 Release Blog](https://huggingface.co/blog/gemma3)
- [Google AI Gemma Documentation](https://ai.google.dev/gemma/docs/core)
- [BitsAndBytesConfig Documentation](https://huggingface.co/docs/transformers/quantization)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)