# 🎯 Confronto Multi-Modello per NLP Italiano

## Panoramica del Progetto

Studio comparativo completo tra **Gemma3-270M** e **Codes-1B** per multiple task di NLP italiano, con focus su ottimizzazione per **GPU 4GB** e valutazioni sistematiche.

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

### Evaluation
```bash
# Valutazione completa tutti i modelli
poetry run python scripts/evaluation/evaluate_all.py
```

## 📊 Conclusioni

1. **Codes-1B è un powerhouse multi-dominio**: 62.5% Text-to-SQL + 100% Intent Classification
2. **Gemma3-270M eccelle su NER** (98.3% F1)
3. **Quantizzazione efficace** per GPU 4GB
4. **Training optimization cruciale**: da 25% → 100% accuracy con giusti parametri
5. **Breakthrough**: I modelli possono eccellere oltre la specializzazione originale

Per dettagli completi: [docs/](docs/)