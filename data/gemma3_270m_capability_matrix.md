# Gemma3-270M Capability Matrix: When to Use vs Avoid

Based on extensive testing with hashtag generation, NER, SQL generation, and intent classification.

## 🏆 SUCCESS CASES (Gemma3-270M Competitive)

| Use Case | Description | Gemma3-270M Performance | Alternative Model | Why Gemma3 Works |
|----------|-------------|-------------------------|-------------------|------------------|
| **Hashtag Generation** | Generate relevant hashtags from social media text | ✅ **85%+ accuracy** | DistilBERT (90%) | Short outputs, pattern matching, creative but bounded |
| **Named Entity Recognition** | Extract person/place/org from text | ✅ **80%+ F1 score** | spaCy (95%) | Token-level tagging, leverages pre-training |
| **Keyword Extraction** | Extract key terms from documents | ✅ **Expected good** | KeyBERT | Similar to hashtags, pattern recognition |
| **Text Summarization (1-2 sentences)** | Very short summaries of articles | ✅ **Likely works** | BART-large | Constrained generation, pre-trained knowledge |
| **Simple Translation** | Common phrase translation (En↔It) | ✅ **Likely works** | MarianMT | Leverages multilingual pre-training |
| **Sentiment Classification** | Positive/negative/neutral detection | ✅ **Expected good** | DistilBERT | Simple classification with generation approach |
| **Text Completion** | Complete partial sentences/phrases | ✅ **Likely works** | GPT-2 small | Core language modeling capability |
| **Style Transfer** | Formal↔casual text conversion | ✅ **Possible** | T5-small | Pattern-based transformation |

## ❌ FAILURE CASES (Gemma3-270M Worthless)

| Use Case | Description | Gemma3-270M Performance | Better Alternative | Why Gemma3 Fails |
|----------|-------------|-------------------------|-------------------|------------------|
| **SQL Generation** | Convert natural language to SQL queries | ❌ **0% - only pad tokens** | CodeT5 (95%) | Complex syntax, logical reasoning required |
| **Intent Classification** | Classify user intents from text | ❌ **0% - empty outputs** | DistilBERT (75%) | Better with classification head than generation |
| **Code Generation** | Generate Python/Java/etc from descriptions | ❌ **Expected failure** | CodeLlama | Requires precise syntax, complex logic |
| **Long-form Content** | Blog posts, articles, documentation | ❌ **Expected failure** | GPT-3.5/Claude | Context length, coherence over long text |
| **Math Problem Solving** | Solve arithmetic or algebra problems | ❌ **Expected failure** | GPT-4 | Multi-step reasoning, symbol manipulation |
| **Complex Reasoning** | Multi-step logical problems | ❌ **Expected failure** | GPT-4 | Chain-of-thought reasoning beyond capacity |
| **Technical Documentation** | API docs, code comments, manuals | ❌ **Expected failure** | CodeT5 | Domain expertise, technical accuracy needed |
| **Creative Writing** | Stories, poems, creative content | ❌ **Expected poor** | GPT-3.5 | Lacks creativity, coherence for long form |

## 🔄 MIXED/CONDITIONAL CASES

| Use Case | Description | Gemma3-270M Performance | Better Alternative | Conditions for Success |
|----------|-------------|-------------------------|-------------------|----------------------|
| **Question Answering** | Answer factual questions | 🟡 **Depends on complexity** | BERT-large | Simple facts only, not reasoning |
| **Text Classification** | Categorize documents by topic | 🟡 **Generation approach suboptimal** | DistilBERT | Works but classification models better |
| **Chatbot Responses** | Generate conversational replies | 🟡 **Very simple only** | DialoGPT | Basic greetings/FAQs, not complex dialogue |
| **Email Auto-replies** | Generate email responses | 🟡 **Template-based only** | T5-base | Standard templates, not personalized |
| **Data Validation** | Check if text matches patterns | 🟡 **Simple rules only** | Regex/Rule-based | Basic format checking |

## 📊 CAPABILITY PATTERNS

### ✅ Gemma3-270M Sweet Spot:
- **Input**: 10-100 words
- **Output**: 1-20 tokens
- **Task**: Pattern recognition, tagging, short generation
- **Domain**: General knowledge, no specialized expertise
- **Examples**: Hashtags, NER, keywords, simple sentiment

### ❌ Gemma3-270M Limitations:
- **Complex syntax** (SQL, code)
- **Multi-step reasoning** (math, logic)
- **Long-form generation** (articles, stories)
- **Specialized domains** (medical, legal, technical)
- **High accuracy requirements** (>95%)

## 🚀 RECOMMENDED MODEL ALTERNATIVES

| Task Category | Small Model (≤1B) | Medium Model (2-7B) | Large Model (7B+) |
|---------------|-------------------|--------------------|--------------------|
| **Classification** | DistilBERT | DeBERTa-v3 | Llama-3-8B |
| **Code Generation** | CodeT5-small | CodeLlama-7B | CodeLlama-34B |
| **Text Generation** | T5-small | Gemma-2B | Llama-3-8B |
| **Reasoning** | N/A | Llama-3-8B | GPT-4 |
| **Multilingual** | mBERT | XLM-R | Llama-3-8B |

## 💡 KEY INSIGHTS

1. **Architecture Matters More Than Size**: DistilBERT (66M) > Gemma3-270M for classification
2. **Task Alignment Critical**: Gemma3 works for tasks similar to pre-training (language modeling)
3. **Fine-tuning Stability Issues**: Gemma3-270M shows training collapse (NaN gradients, zero loss)
4. **Sweet Spot**: Short input → short output tasks with pattern recognition
5. **Avoid**: Complex reasoning, structured output, specialized domains

## 🎯 RECOMMENDATION FRAMEWORK

**Use Gemma3-270M when:**
- Task involves short, creative outputs
- Pattern recognition from text
- Speed/size constraints critical
- Similar to pre-training objectives

**Avoid Gemma3-270M when:**
- Need >90% accuracy
- Complex logical reasoning required
- Structured outputs (JSON, SQL, code)
- Long-form generation needed
- Domain expertise required

**Bottom Line**: Gemma3-270M has a narrow but valuable niche in lightweight NLP tasks requiring fast inference and creative pattern matching.