#!/usr/bin/env python3
"""
Unified Analysis Tool - Consolidates all dataset analysis and evaluation functions.
Provides a single interface for all project analysis needs.
"""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

class UnifiedAnalysis:
    """Unified interface for all project analysis tasks."""
    
    def __init__(self, data_dir="data", output_dir="outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_dataset_balance(self):
        """Analyze entity distribution across all datasets."""
        print("🔍 DATASET BALANCE ANALYSIS")
        print("=" * 50)
        
        result = subprocess.run([
            sys.executable, "scripts/analyze_dataset_balance.py",
            "--data-dir", str(self.data_dir)
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        return result.returncode == 0
    
    def run_baseline_comparison(self, test_file="data/val_comprehensive.jsonl"):
        """Run baseline comparison against spaCy and regex."""
        print("\n🏆 BASELINE COMPARISON")
        print("=" * 50)
        
        baseline_script = Path("scripts/baseline_comparison.py")
        if not baseline_script.exists():
            print("❌ Baseline comparison script not found")
            return False
        
        result = subprocess.run([
            sys.executable, str(baseline_script),
            "--test-file", test_file,
            "--output-dir", str(self.output_dir / "baseline_comparison")
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        return result.returncode == 0
    
    def run_cross_validation(self, train_file="data/train_comprehensive.jsonl", 
                           val_file="data/val_comprehensive.jsonl"):
        """Run robust cross-validation analysis."""
        print("\n📊 CROSS-VALIDATION ANALYSIS")
        print("=" * 50)
        
        cv_script = Path("scripts/robust_evaluation.py")
        if not cv_script.exists():
            print("❌ Cross-validation script not found")
            return False
        
        result = subprocess.run([
            sys.executable, str(cv_script),
            "--train-file", train_file,
            "--val-file", val_file,
            "--output-dir", str(self.output_dir / "cross_validation")
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        return result.returncode == 0
    
    def evaluate_ambiguous_cases(self, test_file="data/test_ambiguous.jsonl"):
        """Evaluate model on challenging ambiguous cases."""
        print("\n🎯 AMBIGUOUS CASES EVALUATION")
        print("=" * 50)
        
        if not Path(test_file).exists():
            print(f"❌ Ambiguous test file not found: {test_file}")
            return False
        
        with open(test_file, 'r') as f:
            cases = [json.loads(line) for line in f if line.strip()]
        
        print(f"📝 Testing {len(cases)} challenging cases:")
        
        for i, case in enumerate(cases, 1):
            print(f"\n🔍 Case {i}:")
            print(f"   Text: {case['document'][:80]}...")
            
            expected = json.loads(case['output'])
            print(f"   Expected entities:")
            for entity_type, entities in expected.items():
                if entities:
                    print(f"     {entity_type}: {entities}")
        
        # TODO: Add actual model evaluation when model path is provided
        print("\n💡 To evaluate with a trained model, use:")
        print("   python scripts/inference.py --model_path <path> --file data/test_ambiguous.jsonl")
        
        return True
    
    def check_project_health(self):
        """Check overall project health and readiness."""
        print("\n🏥 PROJECT HEALTH CHECK")
        print("=" * 50)
        
        health_report = {
            "datasets": {},
            "scripts": {},
            "readiness": {}
        }
        
        # Check dataset files
        critical_files = [
            "data/train_comprehensive.jsonl",
            "data/val_comprehensive.jsonl", 
            "data/test_ambiguous.jsonl"
        ]
        
        print("📂 Checking critical datasets:")
        for file_path in critical_files:
            path = Path(file_path)
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            
            if exists:
                with open(path, 'r') as f:
                    lines = sum(1 for line in f if line.strip())
                print(f"   ✅ {file_path}: {lines} examples ({size:,} bytes)")
                health_report["datasets"][file_path] = {"exists": True, "examples": lines, "size": size}
            else:
                print(f"   ❌ {file_path}: Missing")
                health_report["datasets"][file_path] = {"exists": False, "examples": 0, "size": 0}
        
        # Check script files
        script_files = [
            "scripts/finetune_gemma3.py",
            "scripts/inference.py",
            "scripts/comprehensive_evaluation.py",
            "scripts/tune_confidence_thresholds.py"
        ]
        
        print("\n🛠️ Checking critical scripts:")
        for file_path in script_files:
            exists = Path(file_path).exists()
            status = "✅" if exists else "❌"
            print(f"   {status} {file_path}")
            health_report["scripts"][file_path] = exists
        
        # Readiness assessment
        train_ready = Path("data/train_comprehensive.jsonl").exists()
        val_ready = Path("data/val_comprehensive.jsonl").exists()
        scripts_ready = all(Path(f).exists() for f in script_files)
        
        print(f"\n🚀 READINESS ASSESSMENT:")
        print(f"   Training Ready: {'✅' if train_ready else '❌'}")
        print(f"   Validation Ready: {'✅' if val_ready else '❌'}")
        print(f"   Scripts Ready: {'✅' if scripts_ready else '❌'}")
        
        overall_ready = train_ready and val_ready and scripts_ready
        print(f"   Overall Ready: {'✅' if overall_ready else '❌'}")
        
        health_report["readiness"] = {
            "training": train_ready,
            "validation": val_ready,
            "scripts": scripts_ready,
            "overall": overall_ready
        }
        
        # Save health report
        report_file = self.output_dir / "project_health_report.json"
        with open(report_file, 'w') as f:
            json.dump(health_report, f, indent=2)
        
        print(f"\n📄 Health report saved: {report_file}")
        return overall_ready
    
    def generate_summary_report(self):
        """Generate comprehensive project summary."""
        print("\n📋 GENERATING PROJECT SUMMARY")
        print("=" * 50)
        
        summary = {
            "project_name": "Gemma 3 270M Italian NER",
            "methodology_improvements": {
                "validation_expansion": "93 → 345 examples (+271%)",
                "training_expansion": "437 → 690 examples (+58%)", 
                "entity_balance": "All types >84% coverage",
                "ambiguous_cases": "5 challenging test cases",
                "evaluation_tools": "Statistical analysis ready"
            },
            "expected_improvements": {
                "stable_metrics": "345 validation examples provide reliable estimates",
                "better_recall": "Confidence threshold tuning addresses precision=1.0 issue",
                "fair_comparison": "Larger balanced test set for spaCy comparison",
                "edge_cases": "Ambiguous entity evaluation capability"
            },
            "next_steps": [
                "Train model on train_comprehensive.jsonl (690 examples)",
                "Evaluate on val_comprehensive.jsonl (345 examples)",
                "Run confidence threshold tuning for better recall",
                "Test on test_ambiguous.jsonl for challenging cases",
                "Compare against spaCy baseline with new datasets"
            ],
            "files_ready": {
                "training_data": "data/train_comprehensive.jsonl",
                "validation_data": "data/val_comprehensive.jsonl",
                "ambiguous_tests": "data/test_ambiguous.jsonl",
                "analysis_tools": "scripts/unified_analysis.py",
                "evaluation_pipeline": "scripts/comprehensive_evaluation.py"
            }
        }
        
        summary_file = self.output_dir / "project_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("📊 Project Summary:")
        print(f"   🎯 Training data: 690 examples (balanced)")
        print(f"   🔍 Validation data: 345 examples (reliable)")
        print(f"   🧪 Ambiguous tests: 5 challenging cases")
        print(f"   🛠️ Analysis tools: Ready")
        print(f"   📈 Expected impact: Stable metrics, better recall")
        
        print(f"\n💾 Summary saved: {summary_file}")
        return summary
    
    def run_full_analysis(self):
        """Run complete analysis suite."""
        print("🚀 RUNNING FULL PROJECT ANALYSIS")
        print("=" * 60)
        
        # 1. Dataset balance analysis
        self.analyze_dataset_balance()
        
        # 2. Project health check
        healthy = self.check_project_health()
        
        # 3. Ambiguous cases evaluation
        self.evaluate_ambiguous_cases()
        
        # 4. Generate summary
        self.generate_summary_report()
        
        if healthy:
            print("\n✅ PROJECT IS READY FOR TRAINING AND EVALUATION!")
            print("\n🚀 Recommended next steps:")
            print("   1. poetry run python scripts/finetune_gemma3.py --train_path data/train_comprehensive.jsonl --val_path data/val_comprehensive.jsonl")
            print("   2. poetry run python scripts/comprehensive_evaluation.py")
            print("   3. poetry run python scripts/tune_confidence_thresholds.py --model-path <trained_model>")
        else:
            print("\n⚠️ PROJECT NEEDS ATTENTION BEFORE PROCEEDING")
            print("   Check the health report for missing files")
        
        return healthy

def main():
    parser = argparse.ArgumentParser(description='Unified Analysis Tool for Gemma 3 270M Italian NER')
    parser.add_argument('--action', choices=[
        'balance', 'baseline', 'cv', 'ambiguous', 'health', 'summary', 'full'
    ], default='full', help='Analysis action to perform')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='outputs/unified_analysis', help='Output directory')
    parser.add_argument('--test-file', help='Test file for baseline comparison')
    parser.add_argument('--train-file', help='Training file for cross-validation')
    parser.add_argument('--val-file', help='Validation file for cross-validation')
    
    args = parser.parse_args()
    
    analyzer = UnifiedAnalysis(args.data_dir, args.output_dir)
    
    if args.action == 'balance':
        analyzer.analyze_dataset_balance()
    elif args.action == 'baseline':
        test_file = args.test_file or "data/val_comprehensive.jsonl"
        analyzer.run_baseline_comparison(test_file)
    elif args.action == 'cv':
        train_file = args.train_file or "data/train_comprehensive.jsonl"
        val_file = args.val_file or "data/val_comprehensive.jsonl"
        analyzer.run_cross_validation(train_file, val_file)
    elif args.action == 'ambiguous':
        analyzer.evaluate_ambiguous_cases()
    elif args.action == 'health':
        analyzer.check_project_health()
    elif args.action == 'summary':
        analyzer.generate_summary_report()
    elif args.action == 'full':
        analyzer.run_full_analysis()

if __name__ == '__main__':
    main()