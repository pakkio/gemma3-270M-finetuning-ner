#!/usr/bin/env python3
"""
Train spaCy NER model on our Italian dataset.
Measures training time and performance for fair comparison with Gemma3.
"""
import subprocess
import time
import sys
from pathlib import Path
import argparse
import json

def train_spacy_model(config_path, output_dir, gpu=False):
    """Train spaCy model and measure time."""
    
    print("🚀 SPACY NER TRAINING")
    print("=" * 50)
    print(f"📁 Config: {config_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  GPU: {'Yes' if gpu else 'No'}")
    print()
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build training command
    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--verbose"
    ]
    
    if gpu:
        cmd.append("--gpu-id")
        cmd.append("0")
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print("⏱️  Starting timer...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        if result.returncode == 0:
            print("✅ Training successful!")
            print("\nTraining output:")
            print(result.stdout)
            
            # Save timing info
            timing_info = {
                "training_time_seconds": training_time,
                "training_time_minutes": training_time / 60,
                "success": True,
                "gpu_used": gpu
            }
            
            with open(Path(output_dir) / "training_time.json", 'w') as f:
                json.dump(timing_info, f, indent=2)
                
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 30 minutes!")
        return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train spaCy NER model')
    parser.add_argument('--config', default='configs/spacy_ner_config.cfg',
                       help='spaCy config file')
    parser.add_argument('--output', default='outputs/spacy-ner-italian',
                       help='Output directory')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return False
    
    # Check if training data exists
    if not Path("data/train_spacy.spacy").exists():
        print("❌ Training data not found. Run convert_to_spacy_format.py first")
        return False
    
    success = train_spacy_model(args.config, args.output, args.gpu)
    
    if success:
        print(f"\n🎉 spaCy model ready at: {args.output}")
        print("📊 Check training_time.json for timing details")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)