# 🏋️ Training Guide - Gemma 3 270M Comprehensive Dataset

## ⏱️ **Training Time Expectations**

### **Comprehensive Dataset (690 training examples)**
- **Expected time**: 15-30 minutes
- **Previous**: 2-5 minutes (90 examples)
- **Increase**: ~6x longer due to larger dataset

### **Hardware Requirements**
- **Minimum**: 4GB VRAM (GTX 1650, RTX 3050) - ~30 minutes
- **Recommended**: 6-8GB VRAM (RTX 3060, RTX 4060) - ~15 minutes
- **High-end**: 12GB+ VRAM (RTX 4070+) - ~10 minutes

## 🚀 **Training Options**

### **Option 1: Interactive Training**
```bash
# Will prompt before starting 15-30min training
poetry run python scripts/unified_evaluation.py --action complete
```

### **Option 2: Direct Training**
```bash
# Start training immediately
poetry run python scripts/unified_evaluation.py --action train
```

### **Option 3: Manual Training**
```bash
# Full control over parameters
poetry run python scripts/finetune_gemma3.py \
    --train_path data/train_comprehensive.jsonl \
    --val_path data/val_comprehensive.jsonl \
    --output_dir outputs/gemma3-comprehensive \
    --epochs 8 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --bf16
```

## 📊 **Training Progress Monitoring**

### **Expected Loss Progression**
```
Epoch 1: Loss ~0.6-0.8 (high initial loss)
Epoch 2: Loss ~0.3-0.5 (rapid improvement)
Epoch 4: Loss ~0.1-0.2 (convergence starts)
Epoch 6: Loss ~0.05-0.1 (fine-tuning)
Epoch 8: Loss ~0.01-0.05 (final optimization)
```

### **Checkpoints Saved**
```
outputs/gemma3-comprehensive/
├── checkpoint-100/    # After ~2 minutes
├── checkpoint-200/    # After ~4 minutes  
├── checkpoint-300/    # After ~6 minutes
└── final model/       # After 15-30 minutes
```

## ⚡ **Speed Optimization Tips**

### **Faster Training**
```bash
# Reduce epochs for quicker results (may hurt performance)
--epochs 5

# Increase batch size if you have more VRAM
--batch_size 8

# Use smaller LoRA rank for speed
--lora_r 8
```

### **Quality Training (Recommended)**
```bash
# Current optimized settings
--epochs 8 --batch_size 4 --lr 2e-4 --lora_r 16 --bf16
```

## 🔄 **Background Training**

### **Using nohup (Linux/Mac)**
```bash
nohup poetry run python scripts/unified_evaluation.py --action train > training.log 2>&1 &
```

### **Using screen (Linux/Mac)**
```bash
screen -S training
poetry run python scripts/unified_evaluation.py --action train
# Ctrl+A, D to detach
# screen -r training to reattach
```

### **Monitoring Progress**
```bash
# Watch log file
tail -f training.log

# Check if training is running
ps aux | grep finetune_gemma3
```

## 📈 **Expected Results**

### **Performance Improvements**
- **Better recall**: Larger dataset should reduce precision=1.0 issue
- **More stable metrics**: 345 validation examples vs 93
- **Balanced entities**: 84%+ coverage for all entity types

### **Training Metrics**
- **Token accuracy**: Should reach 99%+
- **Final loss**: Expected <0.05
- **Validation loss**: Should track training loss closely

## 🛑 **If Training Fails**

### **Common Issues**
```bash
# CUDA OOM - reduce batch size
--batch_size 2

# Too slow - reduce epochs
--epochs 5

# Memory issues - use CPU
--device cpu (much slower)
```

### **Fallback Options**
```bash
# Use smaller dataset for testing
poetry run python scripts/finetune_gemma3.py \
    --train_path data/expanded/train_expanded.jsonl \
    --val_path data/expanded/val_expanded.jsonl \
    --epochs 5

# Use existing models for evaluation
poetry run python scripts/unified_evaluation.py --action complete --skip-training
```

## 🎯 **Quick Commands**

```bash
# Check if ready to train
poetry run python scripts/unified_analysis.py --action health

# Start interactive training (with prompt)
poetry run python scripts/unified_evaluation.py --action complete

# Skip training, evaluate existing models
poetry run python scripts/unified_evaluation.py --action complete --skip-training

# Baseline comparison only (no training)
poetry run python scripts/unified_evaluation.py --action baseline
```

**💡 Recommendation**: Start with baseline comparison to verify datasets, then proceed with training when ready for the 15-30 minute commitment.