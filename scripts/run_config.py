#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import subprocess
import sys
from pathlib import Path

def load_config(config_file, config_name):
    """Carica una configurazione specifica dal file JSON"""
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    if config_name not in configs['model_configs']:
        available = list(configs['model_configs'].keys())
        raise ValueError(f"Config '{config_name}' non trovata. Disponibili: {available}")
    
    return configs['model_configs'][config_name]

def run_training_with_config(config, train_path, val_path, output_dir):
    """Esegue il training con i parametri della configurazione"""
    cmd = [
        sys.executable, "scripts/finetune_gemma3.py",
        "--model_id", config["model_id"],
        "--train_path", train_path,
        "--val_path", val_path,
        "--output_dir", output_dir,
        "--lr", str(config["lr"]),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--grad_accum", str(config["grad_accum"]),
        "--lora_r", str(config["lora_r"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--lora_dropout", str(config["lora_dropout"]),
        "--max_seq_len", str(config["max_seq_len"]),
        "--bf16"
    ]
    
    print(f"Eseguendo training con configurazione: {config['description']}")
    print(f"Comando: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Esegui training con configurazioni predefinite")
    parser.add_argument("--config", type=str, default="gemma3_270m_balanced", 
                       help="Nome configurazione (gemma3_270m_fast/balanced/quality)")
    parser.add_argument("--config_file", type=str, default="configs/gemma3_270m_optimized.json",
                       help="File delle configurazioni")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/val.jsonl")
    parser.add_argument("--output_dir", type=str, help="Directory output (auto se non specificata)")
    parser.add_argument("--list_configs", action="store_true", help="Mostra configurazioni disponibili")
    
    args = parser.parse_args()
    
    if args.list_configs:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
        print("Configurazioni disponibili:")
        for name, config in configs['model_configs'].items():
            print(f"  {name}: {config['description']}")
        return
    
    try:
        config = load_config(args.config_file, args.config)
        
        if not args.output_dir:
            args.output_dir = f"outputs/gemma3-270m-{args.config}"
        
        # Crea directory se non esiste
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Salva configurazione usata
        with open(f"{args.output_dir}/config_used.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        returncode = run_training_with_config(config, args.train_path, args.val_path, args.output_dir)
        
        if returncode == 0:
            print(f"\nTraining completato con successo!")
            print(f"Modello salvato in: {args.output_dir}")
            print(f"\nPer testare:")
            print(f"python scripts/inference.py --model_path {args.output_dir} --interactive")
        else:
            print(f"\nTraining fallito con codice di errore: {returncode}")
            
    except Exception as e:
        print(f"Errore: {e}")
        return 1

if __name__ == "__main__":
    main()