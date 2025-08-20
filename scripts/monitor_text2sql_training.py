#!/usr/bin/env python3
"""
Monitor Text-to-SQL training progress
"""

import json
import time
from pathlib import Path
import argparse

def check_training_status():
    """Check the status of both training processes"""
    codet5_path = Path("outputs/codet5-text2sql")
    gemma3_path = Path("outputs/gemma3-text2sql")
    
    print("=== TEXT-TO-SQL TRAINING STATUS ===\n")
    
    # Check CodeT5
    print("📊 CodeT5 Training:")
    if codet5_path.exists():
        config_file = codet5_path / "training_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            print(f"  ✅ Training completed!")
            print(f"  📈 Training time: {config.get('training_time_seconds', 0):.1f}s")
            print(f"  📉 Final loss: {config.get('final_train_loss', 0):.4f}")
        else:
            checkpoints = list(codet5_path.glob("checkpoint-*"))
            print(f"  🔄 Training in progress ({len(checkpoints)} checkpoints)")
    else:
        print("  ⏳ Not started yet")
    
    print()
    
    # Check Gemma3
    print("🤖 Gemma3 Training:")
    if gemma3_path.exists():
        config_file = gemma3_path / "training_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            print(f"  ✅ Training completed!")
            print(f"  📈 Training time: {config.get('training_time_seconds', 0):.1f}s")
            print(f"  📉 Final loss: {config.get('final_train_loss', 0):.4f}")
        else:
            checkpoints = list(gemma3_path.glob("checkpoint-*"))
            print(f"  🔄 Training in progress ({len(checkpoints)} checkpoints)")
    else:
        print("  ⏳ Not started yet")
    
    print()
    
    # Check if ready for evaluation
    both_complete = (
        (codet5_path / "training_config.json").exists() and 
        (gemma3_path / "training_config.json").exists()
    )
    
    if both_complete:
        print("🎉 Both models trained! Ready for evaluation:")
        print("   poetry run python scripts/evaluate_text2sql.py")
    else:
        print("⏳ Waiting for training to complete...")
    
    return both_complete

def main():
    parser = argparse.ArgumentParser(description="Monitor Text-to-SQL training")
    parser.add_argument("--watch", action="store_true", help="Watch continuously")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    
    args = parser.parse_args()
    
    if args.watch:
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                complete = check_training_status()
                if complete:
                    print("Training complete! Exiting monitor.")
                    break
                print(f"\nRefreshing in {args.interval}s... (Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        check_training_status()

if __name__ == "__main__":
    main()