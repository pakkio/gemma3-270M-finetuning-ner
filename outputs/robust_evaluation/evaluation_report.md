# 📊 Robust Evaluation Report

**Cross-Validation:** 5-fold

## 🎯 Main Results (with 95% Confidence Intervals)

| Entity Type | F1 Score | 95% CI | Std Dev |
|-------------|----------|--------|---------|
| People | 0.008 | [0.000, 0.016] | 0.010 |
| Places | 0.085 | [0.051, 0.120] | 0.040 |
| Dates | 0.012 | [0.000, 0.027] | 0.015 |
| **Overall (Macro)** | **0.035** | **[0.022, 0.048]** | **0.015** |

## 🔍 Statistical Interpretation

- **Model Stability:** **Very stable** (CI width: 0.026)
- **Performance Level:** **Needs improvement** (F1: 0.035)

## 💡 Evidence-Based Recommendations

2. **Lower confidence bound concerning** → Model may fail in some scenarios
3. **Small sample warning** → Results may not generalize, need more data