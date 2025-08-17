# 🗺️ Gemma 3 270M Italian NER - Project Map

## 🚀 **Quick Start Guide**

### **1. Project Health Check**
```bash
poetry run python scripts/unified_analysis.py --action health
```

### **2. Complete Analysis**
```bash
poetry run python scripts/unified_analysis.py --action full
```

### **3. Complete Evaluation Pipeline**
```bash
poetry run python scripts/unified_evaluation.py --action complete
```

---

## 📁 **Simplified Project Structure**

```
gemma3-270M-finetuning-ner/
├── 📖 **Documentation**
│   ├── PROJECT_OVERVIEW.md           # 📋 Complete project documentation
│   ├── PROJECT_MAP.md               # 🗺️ This navigation guide
│   └── TRAINING_GUIDE.md            # 🏋️ Training time expectations & tips
│
├── 📊 **Ready-to-Use Datasets**
│   ├── data/train_comprehensive.jsonl    # 690 balanced training examples
│   ├── data/val_comprehensive.jsonl      # 345 validation examples  
│   └── data/test_ambiguous.jsonl         # 5 challenging test cases
│
├── 🛠️ **Core Tools**
│   ├── scripts/unified_analysis.py       # 🔍 All analysis functions
│   ├── scripts/unified_evaluation.py     # 📊 Complete evaluation pipeline
│   ├── scripts/finetune_gemma3.py       # 🏋️ Model training
│   └── scripts/inference.py             # 🧠 Model inference
│
└── 📤 **Results** (Generated)
    ├── outputs/unified_analysis/         # Analysis reports
    ├── outputs/unified_evaluation/       # Evaluation results
    └── outputs/gemma3-comprehensive/     # Trained models
```

---

## 🎯 **Main Use Cases**

### **🔍 Analysis & Health Check**
```bash
# Check if everything is ready
poetry run python scripts/unified_analysis.py --action health

# Full project analysis
poetry run python scripts/unified_analysis.py --action full

# Dataset balance analysis only
poetry run python scripts/unified_analysis.py --action balance
```

### **🏋️ Training & Evaluation**
```bash
# Complete pipeline (will prompt before 15-30min training)
poetry run python scripts/unified_evaluation.py --action complete

# Training only (15-30 minutes)
poetry run python scripts/unified_evaluation.py --action train

# Evaluation only (if model exists)
poetry run python scripts/unified_evaluation.py --action complete --skip-training

# Quick baseline comparison (no training needed)
poetry run python scripts/unified_evaluation.py --action baseline
```

### **🧪 Specific Testing**
```bash
# Test ambiguous cases
poetry run python scripts/unified_evaluation.py --action ambiguous

# Baseline comparison
poetry run python scripts/unified_evaluation.py --action baseline

# Confidence threshold tuning
poetry run python scripts/unified_evaluation.py --action tune
```

---

## 📊 **Key Datasets**

| Dataset | Examples | Purpose | Entity Coverage |
|---------|----------|---------|-----------------|
| `train_comprehensive.jsonl` | 690 | Training | 84%+ all types |
| `val_comprehensive.jsonl` | 345 | Validation | 84%+ all types |
| `test_ambiguous.jsonl` | 5 | Edge cases | 100% challenging |

---

## 🛠️ **Unified Tools**

### **`scripts/unified_analysis.py`** 
**All analysis functions in one tool:**
- Dataset balance analysis
- Project health check  
- Summary report generation
- Full analysis pipeline

### **`scripts/unified_evaluation.py`**
**Complete evaluation pipeline:**
- Model training
- Baseline comparison
- Model evaluation
- Ambiguous case testing
- Confidence threshold tuning
- Final report generation

---

## 🎯 **Issues Addressed**

| Issue | Solution | Status |
|-------|----------|--------|
| **Validation too small** (93 examples) | 345 examples (+271%) | ✅ **RESOLVED** |
| **Ultra-conservative model** (precision=1.0) | Confidence threshold tuning | ✅ **TOOL READY** |
| **Poor entity balance** | Balanced datasets (84%+ coverage) | ✅ **RESOLVED** |
| **Missing ambiguous cases** | 5 challenging test cases | ✅ **RESOLVED** |
| **Evaluation complexity** | Unified analysis/evaluation tools | ✅ **SIMPLIFIED** |

---

## 🚀 **Recommended Workflow**

### **For New Users:**
1. `poetry run python scripts/unified_analysis.py --action health` - Check readiness
2. `poetry run python scripts/unified_evaluation.py --action baseline` - Quick baseline (5min)
3. `poetry run python scripts/unified_evaluation.py --action complete` - Full pipeline (⚠️ 15-30min training)
4. Review results in `outputs/unified_evaluation/`

### **For Development:**
1. `poetry run python scripts/unified_analysis.py --action full` - Analysis
2. `poetry run python scripts/unified_evaluation.py --action train` - Training
3. `poetry run python scripts/unified_evaluation.py --action ambiguous` - Test edge cases

### **For Research:**
1. Check `PROJECT_OVERVIEW.md` for complete methodology
2. Use `scripts/unified_evaluation.py --action baseline` for comparisons
3. Generate reports with `scripts/unified_analysis.py --action summary`

---

## 📈 **Expected Results**

With the comprehensive datasets, expect:
- **More stable metrics** (345 validation examples)
- **Better recall** (threshold tuning addresses precision=1.0)
- **Fair spaCy comparison** (larger balanced test set)
- **Edge case evaluation** (ambiguous entity testing)

---

## 💡 **Key Commands Summary**

```bash
# Quick health check
poetry run python scripts/unified_analysis.py --action health

# Complete analysis
poetry run python scripts/unified_analysis.py --action full

# Complete evaluation (train + test everything)
poetry run python scripts/unified_evaluation.py --action complete

# Evaluation without retraining
poetry run python scripts/unified_evaluation.py --action complete --skip-training
```

**🎉 Project is ready for systematic evaluation addressing all methodological concerns!**