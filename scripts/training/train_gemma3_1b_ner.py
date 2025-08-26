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
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

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
            text = self.tokenizer.decode(input_id, skip_special_tokens=True)
            
            if self.response_template in text:
                template_start = text.find(self.response_template)
                response_start = template_start + len(self.response_template)
                
                prefix_text = text[:response_start]
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                
                label = [-100] * len(prefix_tokens) + input_id[len(prefix_tokens):]
                if len(label) > len(input_id):
                    label = label[:len(input_id)]
                elif len(label) < len(input_id):
                    label = label + input_id[len(label):]
            else:
                label = input_id.copy()
            
            labels.append(label)
        
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
    parser.add_argument("--model_id", type=str, default="unsloth/gemma-3-1b-it-bnb-4bit")
    parser.add_argument("--train_path", type=str, default="data/ner/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/ner/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3-1b-ner")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true", default=True)
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

    print(f"Loading Gemma3 1B model {args.model_id}...")
    
    # 4-bit quantization config for 1B model
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
    )

    # LoRA configuration optimized for 1B model
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        fp16=args.fp16,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        group_by_length=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=train_args,
        data_collator=collator,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_len
    )

    print(f"🚀 Starting Gemma3 1B training with {len(train_ds)} training examples")
    print(f"⚡ Config: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"💻 Using: {'4-bit quantization' if args.use_4bit else 'full precision'}")
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "inference_template.txt"), "w", encoding="utf-8") as f:
        f.write(INFERENCE_TEMPLATE)

    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"✅ Gemma3 1B training completed. Model saved in: {args.output_dir}")

if __name__ == "__main__":
    main()