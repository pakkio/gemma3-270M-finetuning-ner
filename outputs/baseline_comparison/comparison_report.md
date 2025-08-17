# 📊 Baseline Comparison Report

**Test dataset size:** 35 examples

## 🏆 Performance Summary

| Model | People F1 | Places F1 | Dates F1 | Macro F1 | Inference Time |
|-------|-----------|-----------|----------|----------|----------------|
| spaCy Italian | 0.970 | 0.400 | 0.000 | 0.457 | 6.4ms |
| Regex Patterns | 0.182 | 0.368 | 0.500 | 0.350 | 0.1ms |

## 📈 Detailed Analysis

### spaCy Italian Model
- **Strengths:** Good person recognition (F1: 0.970)
- **Weaknesses:** Limited date support (F1: 0.000), location confusion
- **Speed:** Fast inference (6.4ms per document)

### Regex Patterns
- **Strengths:** Very fast (0.1ms), reliable for standard patterns
- **Weaknesses:** Rigid patterns, many false positives/negatives
- **Use case:** Good for structured data with known formats

## 💡 Recommendations

1. **Current best baseline:** Spacy (F1: 0.457)
2. **Model needs improvement** to beat production baselines
3. **Hybrid approach:** Consider combining regex for dates + spaCy for entities
4. **Error analysis:** Focus on false positives in place detection