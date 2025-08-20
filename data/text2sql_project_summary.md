# 🗄️ Italian Text-to-SQL Fine-tuning Project

## 📋 Project Overview

This is the **third fine-tuning experiment** in our comprehensive comparison study, adding **code generation capabilities** to complement our existing NER and hashtag generation tasks.

### 🎯 Task: Italian Business Queries → SQL
- **Input**: Natural language questions in Italian about business data
- **Output**: Structured SQL queries for database operations
- **Domain**: Italian business scenarios (customers, orders, products, employees)

### 🏆 Model Comparison: CodeT5 vs Gemma3
- **Baseline**: Fine-tuned CodeT5 (Salesforce/codet5p-220m)
- **Challenger**: Fine-tuned Gemma3 (google/gemma-2-2b)
- **Fair Fight**: Both models trained on identical Italian→SQL dataset

## 📊 Dataset Statistics

### Training Data (`train_text2sql.jsonl`)
- **Examples**: 128 Italian question-SQL pairs
- **Coverage**: Comprehensive business scenarios
- **Complexity**: Simple SELECT to complex JOINs with aggregations

### Validation Data (`val_text2sql.jsonl`)  
- **Examples**: 51 diverse test cases
- **Scenarios**: Edge cases, complex queries, multi-table operations

### Sample Training Examples
```json
{
  "question": "Mostra tutti i clienti di Milano con ordini superiori a 1000 euro",
  "sql": "SELECT DISTINCT c.* FROM clienti c JOIN ordini o ON c.id_cliente = o.id_cliente WHERE c.citta = 'Milano' AND o.totale > 1000;"
}

{
  "question": "Conta le fatture pagate per ogni mese",
  "sql": "SELECT MONTH(data_pagamento) as mese, COUNT(*) as fatture_pagate FROM fatture WHERE stato = 'pagata' GROUP BY MONTH(data_pagamento);"
}
```

## 🗃️ Database Schema

Comprehensive Italian business schema with:
- **10 tables**: clienti, prodotti, ordini, dipendenti, fornitori, fatture, etc.
- **Realistic relationships**: Foreign keys, business logic constraints
- **Italian field names**: Complete localization for authentic scenarios
- **Business domains**: E-commerce, HR, accounting, inventory management

## 🔧 Technical Implementation

### CodeT5 Fine-tuning (`finetune_codet5_text2sql.py`)
- **Model**: Salesforce/codet5p-220m
- **Architecture**: Sequence-to-sequence with T5 tokenizer
- **Prompt**: "Converti la seguente domanda italiana in una query SQL: {question}"
- **Training**: LoRA with r=16, 3 epochs, batch_size=4

### Gemma3 Fine-tuning (`finetune_gemma3_text2sql.py`)
- **Model**: google/gemma-2-2b  
- **Architecture**: Causal language model with instruction format
- **Prompt**: Structured instruction template with Italian context
- **Training**: LoRA with r=16, 3 epochs, batch_size=2

### Training Configuration
```python
# Optimized for Text-to-SQL task
{
  "learning_rate": 1e-4,  # Conservative for SQL syntax
  "epochs": 3,            # Prevent overfitting
  "max_length": 1024,     # Handle complex queries
  "lora_r": 16,           # Balanced parameter efficiency
  "bf16": True            # Memory optimization
}
```

## 📏 Evaluation Metrics

### Comprehensive SQL Assessment (`evaluate_text2sql.py`)
1. **Exact Match**: Character-perfect SQL matching
2. **Syntax Validity**: SQLparse validation of generated queries
3. **Normalized Match**: Formatted SQL comparison (consistent styling)
4. **Semantic Similarity**: Component-based analysis
   - SELECT fields matching
   - Table references accuracy
   - WHERE conditions similarity
   - JOIN operations correctness
   - Aggregation functions alignment

### Performance Metrics
- **Accuracy**: Multiple levels of correctness measurement
- **Speed**: Inference time per query generation
- **Reliability**: Percentage of syntactically valid outputs

## 🎪 Expected Capabilities

### Basic Queries
```sql
-- Italian: "Mostra tutti i clienti di Roma"
SELECT * FROM clienti WHERE citta = 'Roma';

-- Italian: "Conta gli ordini del 2024"
SELECT COUNT(*) FROM ordini WHERE YEAR(data_ordine) = 2024;
```

### Complex Operations
```sql
-- Italian: "Trova i clienti VIP con spesa totale > 10000 euro"
SELECT c.nome, c.cognome, SUM(o.totale) as spesa_totale 
FROM clienti c JOIN ordini o ON c.id_cliente = o.id_cliente 
GROUP BY c.id_cliente, c.nome, c.cognome 
HAVING SUM(o.totale) > 10000 
ORDER BY spesa_totale DESC;
```

### Business Intelligence
```sql
-- Italian: "Mostra il trend mensile vendite per categoria sport"
SELECT MONTH(o.data_ordine) as mese, SUM(d.subtotale) as vendite 
FROM ordini o JOIN dettagli_ordine d ON o.id_ordine = d.id_ordine 
JOIN prodotti p ON d.id_prodotto = p.id_prodotto 
WHERE p.categoria = 'sport' AND YEAR(o.data_ordine) = YEAR(CURDATE()) 
GROUP BY MONTH(o.data_ordine) ORDER BY mese;
```

## 🚀 Project Significance

### Why Text-to-SQL?
1. **Practical Value**: Business users can query databases in natural language
2. **Different Domain**: Code generation vs. natural language processing
3. **Structured Output**: SQL syntax requires precise formatting
4. **Italian Context**: Localized business terminology and scenarios

### Comparison Context
This project completes our **three-domain evaluation**:
1. **NER**: Entity extraction (vs spaCy) → spaCy wins 98.4% vs 97.6%
2. **Hashtags**: Content generation (vs BERT) → Results pending
3. **Text-to-SQL**: Code generation (vs CodeT5) → Currently training

### Expected Insights
- How do general-purpose LLMs handle structured code generation?
- Does Italian language capability transfer to SQL generation?
- Can Gemma3 compete with specialized code models like CodeT5?
- What are the trade-offs between model size and task-specific training?

## 📁 File Structure
```
data/
├── italian_business_schema.sql      # Database schema definition
├── train_text2sql.jsonl           # Training questions & SQL pairs
├── val_text2sql.jsonl             # Validation dataset
└── text2sql_project_summary.md    # This summary

scripts/
├── finetune_codet5_text2sql.py    # CodeT5 fine-tuning script
├── finetune_gemma3_text2sql.py    # Gemma3 fine-tuning script
└── evaluate_text2sql.py           # Comprehensive evaluation

outputs/
├── codet5-text2sql/               # Fine-tuned CodeT5 model
├── gemma3-text2sql/               # Fine-tuned Gemma3 model
└── text2sql_comparison.json       # Head-to-head results
```

## ⚡ Quick Start

```bash
# 1. Train both models (running in background)
poetry run python scripts/finetune_codet5_text2sql.py --epochs 3
poetry run python scripts/finetune_gemma3_text2sql.py --epochs 3

# 2. Evaluate and compare
poetry run python scripts/evaluate_text2sql.py

# 3. View results
cat outputs/text2sql_comparison.json
```

## 🎯 Success Criteria

### Minimum Viable Performance
- **Syntax Validity**: >80% of generated SQL should be parseable
- **Basic Queries**: >90% accuracy on simple SELECT statements
- **Complex Queries**: >60% semantic similarity on JOINs/aggregations

### Comparative Analysis
- Fair evaluation using identical training data
- Statistical significance with 51 validation examples
- Multiple evaluation metrics for comprehensive assessment
- Honest reporting of which model performs better

---

**Status**: ⏳ Models currently training in background
**Next**: Evaluation and head-to-head comparison
**Goal**: Complete third domain comparison for comprehensive ML study