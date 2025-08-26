# 🎯 Gemma Models Training & Comparison Framework

## Panoramica del Progetto

Framework completo per il fine-tuning e confronto di modelli **Gemma3 (270M & 1B)** vs **CodeS-1B** su multiple task NLP. Include ottimizzazioni per GPU consumer (4-8GB VRAM) con quantizzazione 4-bit e supporto LoRA.

## 🆕 Nuove Funzionalità - Gemma3 1B

### Framework di Training Gemma3 1B
- **`google/gemma-3-1b-it`**: Modello instruction-tuned da 1B parametri
- **32k context window**: 4x superiore a modelli precedenti  
- **140+ lingue**: Supporto multilingue esteso
- **4-bit quantization**: Training efficiente su 8GB VRAM
- **LoRA fine-tuning**: Adattamento rapido e efficiente

## 🏆 Risultati Principali

### Text-to-SQL (Generazione Query SQL da Italiano)
| Modello | Exact Match | Semantic Similarity | Training Time | VRAM |
|---------|-------------|--------------------|--------------|----|
| **Codes-1B** 🥇 | **62.5%** | - | 113s | 4GB |
| CodeT5 | 0% | **56.9%** | 26s | 4GB |
| Gemma3-270M | 0% | 41.7% | 54s | 4GB |

### Named Entity Recognition (Estrazione Entità)
| Modello | F1 Score | Precision | Recall | Training Time |
|---------|----------|-----------|--------|----|
| **Gemma3-270M** 🥇 | **98.3%** | 98.4% | 98.3% | 7 min |
| spaCy Fine-tuned | 98.4% | - | - | 4 min |
| spaCy Generic | 53.3% | - | - | - |

### Intent Classification
| Modello | Accuracy | F1 Score | Training Time | Status |
|---------|----------|----------|---------------|--------|
| **Codes-1B Improved** 🥇 | **100%** | **1.000** | 8h | **Perfetto** |
| Codes-1B Original | 25% | 0.182 | 62s | Fallito |

### Hashtag Generation
| Modello | Training Loss | Training Time | Examples |
|---------|---------------|---------------|----------|
| Codes-1B | 1.306 | 94s | 153/52 |

## 🔧 Caratteristiche Tecniche

- **Quantizzazione**: Tutti i modelli ottimizzati per GPU 4GB
- **LoRA Fine-tuning**: Efficienza massima con parametri minimi
- **Evaluation Sistematica**: Metriche complete per tutti i modelli
- **4 Task NLP**: NER, Text2SQL, Intent Classification, Hashtag Generation

## 📁 Struttura del Progetto

```
├── docs/                  # Documentazione completa
├── data/                  # Dataset organizzati per task
│   ├── ner/              # Named Entity Recognition
│   ├── text2sql/         # Text-to-SQL generation  
│   ├── intent_classification/  # Classificazione intenti
│   └── hashtag_generation/     # Generazione hashtag
├── scripts/               # Script organizzati
│   ├── training/         # Training dei modelli
│   ├── evaluation/       # Valutazione e metriche
│   └── data_preparation/ # Preparazione dataset
├── models/               # Modelli finali
│   ├── production/       # Modelli migliori per produzione
│   └── experiments/      # Esperimenti
└── results/              # Risultati e metriche
```

## 🚀 Quick Start

### Training
```bash
# NER con Gemma3-270M
poetry run python scripts/training/train_gemma3_ner.py

# Text2SQL con Codes-1B  
poetry run python scripts/training/train_codes1b_text2sql.py
```

### Gemma3 1B Training & Comparison
```bash
# Training completo NER + Hashtag + Confronto
poetry run python scripts/run_gemma3_1b_experiments.py --step all

# Solo training NER
poetry run python scripts/run_gemma3_1b_experiments.py --step ner

# Confronto Gemma3 1B vs CodeS-1B
poetry run python scripts/evaluation/gemma3_vs_codes1b_comparison.py
```

### Traditional Evaluation
```bash
# Valutazione completa tutti i modelli (270M focus)
poetry run python scripts/evaluation/evaluate_all.py
```

## 📊 Conclusioni

### Gemma3 270M vs CodeS-1B (Risultati Consolidati)
1. **CodeS-1B è powerhouse multi-dominio**: 62.5% Text-to-SQL + 100% Intent Classification
2. **Gemma3-270M eccelle su NER linguistico** (98.3% F1)
3. **Quantizzazione efficace** per GPU 4GB-8GB
4. **Training optimization cruciale**: da 25% → 100% accuracy

### Gemma3 1B vs CodeS-1B (Predizioni)
1. **NER**: Gemma3 1B winner (+32k context, multilingue)
2. **Hashtag**: Gemma3 1B winner (creatività + comprensione)
3. **SQL**: CodeS-1B winner (specializzazione code)
4. **Intent**: CodeS-1B winner (classification precision)

## 📚 Documentazione

- [Gemma3 1B Training Guide](docs/gemma3_1b_training_guide.md)
- [Model Comparison Framework](docs/model_comparison_framework.md)
- [Documentazione completa](docs/)