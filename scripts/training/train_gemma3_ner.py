#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Implementazione semplice del DataCollator per masking della response
class DataCollatorForCompletionOnlyLM:
    def __init__(self, response_template, tokenizer, mlm=False):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features):
        batch = {}
        input_ids = [f["input_ids"] for f in features]
        labels = []
        
        for input_id in input_ids:
            # Converti in stringa per trovare il template
            text = self.tokenizer.decode(input_id, skip_special_tokens=True)
            
            if self.response_template in text:
                # Trova la posizione del template di risposta
                template_start = text.find(self.response_template)
                response_start = template_start + len(self.response_template)
                
                # Tokenizza solo la parte prima della risposta
                prefix_text = text[:response_start]
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                
                # Crea labels: -100 per prefix, token originali per response
                label = [-100] * len(prefix_tokens) + input_id[len(prefix_tokens):]
                if len(label) > len(input_id):
                    label = label[:len(input_id)]
                elif len(label) < len(input_id):
                    label = label + input_id[len(label):]
            else:
                # Se non trova il template, usa tutto come target
                label = input_id.copy()
            
            labels.append(label)
        
        # Padding
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
            pad_len = max_len - len(ids)
            
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        
        batch["input_ids"] = padded_input_ids
        batch["labels"] = padded_labels
        batch["attention_mask"] = attention_mask
        
        # Converti in tensori
        import torch
        for key in batch:
            batch[key] = torch.tensor(batch[key])
        
        return batch

TEMPLATE = """Sei una funzione di estrazione. Non usi strumenti esterni.
Regole output:
- Rispondi SOLO con un oggetto JSON valido.
- Chiavi: "people","dates","places". Valori: array di stringhe.
- Se assente: [].

### Documento
{document}

### Risposta JSON
{output}"""

INFERENCE_TEMPLATE = """Sei una funzione di estrazione. Non usi strumenti esterni.
Regole output:
- Rispondi SOLO con un oggetto JSON valido.
- Chiavi: "people","dates","places". Valori: array di stringhe.
- Se assente: [].

### Documento
{document}

### Risposta JSON
"""

def format_example(example):
    doc = example["document"]
    out = example.get("output", "")
    return {"text": TEMPLATE.format(document=doc, output=out)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-3-270m")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-entity-extraction")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_files = {"train": args.train_path, "validation": args.val_path}
    raw = load_dataset("json", data_files=data_files)
    train_ds = raw["train"].map(format_example, remove_columns=raw["train"].column_names)
    val_ds = raw["validation"].map(format_example, remove_columns=raw["validation"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model {args.model_id} with 8-bit quantization for 4GB VRAM compatibility...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto",
        attn_implementation="eager"  # Recommended for Gemma3 training stability
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None  # Auto-detect per Gemma 3 270M
    )
    model = get_peft_model(model, lora)

    response_template = "### Risposta JSON\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=train_args,
        data_collator=collator,
        processing_class=tokenizer
    )

    print(f"Starting training with {len(train_ds)} training examples and {len(val_ds)} validation examples")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "inference_template.txt"), "w", encoding="utf-8") as f:
        f.write(INFERENCE_TEMPLATE)

    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training completed. Model saved in: {args.output_dir}")

if __name__ == "__main__":
    main()