#!/usr/bin/env python3

# Quick start script to begin optimized Gemma3-270M training for 8GB VRAM
# This bypasses Poetry installation delays and gets training started immediately

import subprocess
import sys
import os

def install_minimal_deps():
    """Install minimal dependencies needed for training"""
    print("🚀 Installing minimal dependencies for immediate training...")
    
    deps = [
        "torch>=2.0.0",
        "transformers>=4.36.0", 
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "accelerate>=0.24.0"
    ]
    
    for dep in deps:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])

def start_training():
    """Start the optimized training"""
    print("✅ Dependencies installed. Starting optimized Gemma3-270M training...")
    
    cmd = [
        sys.executable,
        "scripts/training/train_gemma3_ner_8gb.py",
        "--train_path", "data/ner/train_expanded.jsonl",
        "--val_path", "data/ner/val_expanded.jsonl", 
        "--epochs", "3",
        "--batch_size", "16",
        "--lr", "5e-4",
        "--fp16"
    ]
    
    print(f"🎯 Command: {' '.join(cmd)}")
    print("⚡ Optimizations for 8GB VRAM:")
    print("  - NO quantization (faster training)")
    print("  - FP16 precision")
    print("  - Batch size 16 (vs 8 with quantization)")
    print("  - Learning rate 5e-4 (higher for faster convergence)")
    print("  - Only 3 epochs needed")
    print("  - Expected time: 15-30 minutes (vs 8 hours)")
    
    # Run in background
    process = subprocess.Popen(cmd)
    print(f"🏃 Training started in background (PID: {process.pid})")
    return process

if __name__ == "__main__":
    if not os.path.exists("data/ner/train_expanded.jsonl"):
        print("❌ Training data not found. Please ensure data/ner/train_expanded.jsonl exists")
        sys.exit(1)
        
    install_minimal_deps()
    process = start_training()
    
    print("\n🔍 To monitor training:")
    print(f"   ps aux | grep {process.pid}")
    print("   nvidia-smi  # Check GPU usage")
    print("\n📊 Training will save to: outputs/gemma3-270m-8gb-optimized/")