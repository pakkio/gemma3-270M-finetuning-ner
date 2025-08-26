#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import json
import argparse
from datetime import datetime
import time

class Gemma3ExperimentRunner:
    def __init__(self, base_dir="/home/pakkio/IdeaProjects/gemma3-270M-finetuning-ner"):
        self.base_dir = base_dir
        self.results = {}
        
    def run_command(self, command, description):
        """Run a command and capture output"""
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"{'='*60}")
        print(f"Command: {command}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS ({duration:.1f}s)")
                if result.stdout:
                    print("Output:", result.stdout[-500:])  # Last 500 chars
                return True, duration, result.stdout
            else:
                print(f"❌ FAILED ({duration:.1f}s)")
                print("Error:", result.stderr)
                return False, duration, result.stderr
                
        except subprocess.TimeoutExpired:
            print("⏱️ TIMEOUT after 1 hour")
            return False, 3600, "Timeout"
        except Exception as e:
            print(f"💥 EXCEPTION: {e}")
            return False, 0, str(e)
    
    def train_gemma3_1b_ner(self):
        """Train Gemma3 1B on NER task"""
        command = """poetry run python scripts/training/train_gemma3_1b_ner.py \
            --model_id google/gemma-3-1b-it \
            --train_path data/ner/train.jsonl \
            --val_path data/ner/val.jsonl \
            --output_dir outputs/gemma3-1b-ner-experiment \
            --lr 2e-4 \
            --epochs 3 \
            --batch_size 4 \
            --grad_accum 4 \
            --use_4bit \
            --lora_r 16 \
            --lora_alpha 32"""
        
        success, duration, output = self.run_command(command, "Training Gemma3 1B on NER")
        self.results["gemma3_1b_ner_training"] = {
            "success": success,
            "duration": duration,
            "output": output[-1000:] if output else ""  # Last 1000 chars
        }
        return success
    
    def train_gemma3_1b_hashtag(self):
        """Train Gemma3 1B on hashtag generation"""
        command = """poetry run python scripts/training/train_gemma3_1b_hashtag.py \
            --model_id google/gemma-3-1b-it \
            --train_path data/hashtag_generation/train.jsonl \
            --val_path data/hashtag_generation/val.jsonl \
            --output_dir outputs/gemma3-1b-hashtag-experiment \
            --lr 2e-4 \
            --epochs 3 \
            --batch_size 4 \
            --grad_accum 4 \
            --use_4bit \
            --lora_r 16 \
            --lora_alpha 32"""
        
        success, duration, output = self.run_command(command, "Training Gemma3 1B on Hashtag Generation")
        self.results["gemma3_1b_hashtag_training"] = {
            "success": success,
            "duration": duration,
            "output": output[-1000:] if output else ""
        }
        return success
    
    def run_comparison(self):
        """Run comparison between Gemma3 1B and CodeS-1B"""
        command = """poetry run python scripts/evaluation/gemma3_vs_codes1b_comparison.py \
            --gemma3_model_path outputs/gemma3-1b-ner-experiment \
            --codes1b_model_path models/production/codes1b-hashtag-best \
            --ner_test_data data/ner/val.jsonl \
            --hashtag_test_data data/hashtag_generation/val.jsonl \
            --output_file results/gemma3_1b_vs_codes1b_final_comparison.json"""
        
        success, duration, output = self.run_command(command, "Running Gemma3 1B vs CodeS-1B Comparison")
        self.results["comparison"] = {
            "success": success,
            "duration": duration,
            "output": output[-1000:] if output else ""
        }
        return success
    
    def check_prerequisites(self):
        """Check if all required files exist"""
        required_files = [
            "data/ner/train.jsonl",
            "data/ner/val.jsonl", 
            "data/hashtag_generation/train.jsonl",
            "data/hashtag_generation/val.jsonl",
            "models/production/codes1b-hashtag-best",
            "scripts/training/train_gemma3_1b_ner.py",
            "scripts/training/train_gemma3_1b_hashtag.py",
            "scripts/evaluation/gemma3_vs_codes1b_comparison.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = os.path.join(self.base_dir, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        print("✅ All prerequisites found")
        return True
    
    def run_full_experiment(self):
        """Run the complete Gemma3 1B experiment"""
        print("🧪 GEMMA3 1B FINE-TUNING EXPERIMENT")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Aborting due to missing prerequisites")
            return False
        
        experiment_start = time.time()
        
        # Step 1: Train NER model
        print("\n📍 STEP 1: Training Gemma3 1B on NER task")
        ner_success = self.train_gemma3_1b_ner()
        
        if not ner_success:
            print("❌ NER training failed, aborting experiment")
            return False
        
        # Step 2: Train hashtag model
        print("\n📍 STEP 2: Training Gemma3 1B on hashtag generation")
        hashtag_success = self.train_gemma3_1b_hashtag()
        
        if not hashtag_success:
            print("⚠️ Hashtag training failed, continuing with comparison using NER model")
        
        # Step 3: Run comparison
        print("\n📍 STEP 3: Running model comparison")
        comparison_success = self.run_comparison()
        
        experiment_duration = time.time() - experiment_start
        
        # Generate summary
        self.results["experiment_summary"] = {
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_duration": experiment_duration,
            "ner_training_success": ner_success,
            "hashtag_training_success": hashtag_success,
            "comparison_success": comparison_success,
            "overall_success": ner_success and comparison_success
        }
        
        # Save results
        results_file = f"results/gemma3_1b_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🏁 EXPERIMENT COMPLETED")
        print("=" * 60)
        print(f"Total duration: {experiment_duration/60:.1f} minutes")
        print(f"NER training: {'✅' if ner_success else '❌'}")
        print(f"Hashtag training: {'✅' if hashtag_success else '❌'}")
        print(f"Model comparison: {'✅' if comparison_success else '❌'}")
        print(f"Results saved to: {results_file}")
        
        if comparison_success and os.path.exists("results/gemma3_1b_vs_codes1b_final_comparison.json"):
            print(f"Comparison results: results/gemma3_1b_vs_codes1b_final_comparison.json")
        
        return self.results["experiment_summary"]["overall_success"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, choices=["ner", "hashtag", "comparison", "all"], 
                        default="all", help="Which step to run")
    parser.add_argument("--base_dir", type=str, 
                        default="/home/pakkio/IdeaProjects/gemma3-270M-finetuning-ner")
    args = parser.parse_args()
    
    runner = Gemma3ExperimentRunner(args.base_dir)
    
    if args.step == "all":
        success = runner.run_full_experiment()
        exit(0 if success else 1)
    elif args.step == "ner":
        success = runner.train_gemma3_1b_ner()
    elif args.step == "hashtag":
        success = runner.train_gemma3_1b_hashtag()
    elif args.step == "comparison":
        success = runner.run_comparison()
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()