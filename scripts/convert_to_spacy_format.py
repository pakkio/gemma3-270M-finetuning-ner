#!/usr/bin/env python3
"""
Convert our JSONL dataset to spaCy training format.
Transforms entity lists into character-level span annotations.
"""
import json
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import re
from pathlib import Path
import argparse

def find_entity_spans(text, entities):
    """
    Find character spans for entities in text.
    Returns list of (start, end, label) tuples.
    """
    spans = []
    text_lower = text.lower()
    
    # Entity type mapping
    entity_labels = {
        'people': 'PERSON',
        'places': 'GPE',  # Geopolitical entity 
        'dates': 'DATE'
    }
    
    for entity_type, entity_list in entities.items():
        if entity_type not in entity_labels:
            continue
            
        label = entity_labels[entity_type]
        
        for entity in entity_list:
            entity_clean = entity.strip()
            if not entity_clean:
                continue
                
            # Try exact match first
            entity_lower = entity_clean.lower()
            start_pos = text_lower.find(entity_lower)
            
            if start_pos != -1:
                end_pos = start_pos + len(entity_clean)
                # Check if it's a word boundary match
                if (start_pos == 0 or not text[start_pos-1].isalnum()) and \
                   (end_pos >= len(text) or not text[end_pos].isalnum()):
                    spans.append((start_pos, end_pos, label))
                    continue
            
            # Try fuzzy matching for complex entities
            # Handle cases like "Prof.ssa Elena Bianchi" where formatting might vary
            pattern = re.escape(entity_clean).replace(r'\ ', r'\s+')
            pattern = pattern.replace(r'\.', r'\.?')  # Optional periods
            
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                spans.append((match.start(), match.end(), label))
    
    return spans

def convert_jsonl_to_spacy(input_file, output_file, lang_model="it_core_news_sm"):
    """Convert JSONL format to spaCy binary format."""
    
    # Load Italian spaCy model for tokenization
    try:
        nlp = spacy.load(lang_model)
    except OSError:
        print(f"❌ {lang_model} not found. Using blank Italian model.")
        nlp = spacy.blank("it")
    
    # Create DocBin to store training examples
    doc_bin = DocBin()
    examples = []
    conversion_stats = {"total": 0, "successful": 0, "entities_found": 0, "entities_total": 0}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text = data['document']
                entities_data = json.loads(data['output'])
                
                conversion_stats["total"] += 1
                
                # Find entity spans
                spans = find_entity_spans(text, entities_data)
                
                # Count entities
                total_entities = sum(len(ents) for ents in entities_data.values())
                conversion_stats["entities_total"] += total_entities
                conversion_stats["entities_found"] += len(spans)
                
                # Create spaCy doc
                doc = nlp.make_doc(text)
                
                # Create Example with annotations
                example_entities = [(start, end, label) for start, end, label in spans]
                example_dict = {"entities": example_entities}
                
                example = Example.from_dict(doc, example_dict)
                examples.append(example)
                doc_bin.add(example.reference)
                
                conversion_stats["successful"] += 1
                
                if line_num <= 3:  # Show first few examples
                    print(f"📝 Example {line_num}:")
                    print(f"   Text: {text[:60]}...")
                    print(f"   Entities: {example_entities}")
                    print()
                    
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
                continue
    
    # Save to binary format
    doc_bin.to_disk(output_file)
    
    # Print conversion statistics
    print("🔄 CONVERSION COMPLETED")
    print("=" * 50)
    print(f"📊 Total examples: {conversion_stats['total']}")
    print(f"✅ Successfully converted: {conversion_stats['successful']}")
    print(f"🎯 Entities found: {conversion_stats['entities_found']}/{conversion_stats['entities_total']}")
    print(f"📈 Entity detection rate: {conversion_stats['entities_found']/conversion_stats['entities_total']*100:.1f}%")
    print(f"💾 Saved to: {output_file}")
    
    return examples

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL to spaCy format')
    parser.add_argument('--input', default='data/train_comprehensive.jsonl', 
                       help='Input JSONL file')
    parser.add_argument('--output', default='data/train_spacy.spacy',
                       help='Output spaCy file')
    parser.add_argument('--lang', default='it_core_news_sm',
                       help='spaCy language model')
    
    args = parser.parse_args()
    
    print("🔄 CONVERTING JSONL TO SPACY FORMAT")
    print("=" * 50)
    print(f"📁 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print(f"🌍 Language model: {args.lang}")
    print()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    examples = convert_jsonl_to_spacy(args.input, args.output, args.lang)
    
    print(f"\n✅ Conversion complete! Ready for spaCy training.")

if __name__ == "__main__":
    main()