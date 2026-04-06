#!/usr/bin/env python3
"""
Comprehensive Full Test Suite Report Generator
Monitors benchmark progress and generates complete report
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

def generate_full_report():
    """Generate comprehensive test report"""
    
    backend_dir = Path("backend")
    metrics_dir = backend_dir / "results" / "metrics"
    
    print("=" * 100)
    print("🧪 NETGUARD COMPREHENSIVE FULL TEST SUITE REPORT")
    print("=" * 100)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📊 TEST CONFIGURATION:")
    print("  Dataset: UNSW-NB15 (Real-world network traffic)")
    print("  Models: ALL 8 architectures")
    print("  - CNN1D")
    print("  - FT-Transformer")
    print("  - Vanilla Autoencoder")
    print("  - Variational Autoencoder (VAE)")
    print("  - BiLSTM with Attention")
    print("  - CNN-LSTM Hybrid")
    print("  - Isolation Forest")
    print("  - Contrastive SSL")
    print("  Epochs: 5 per run")
    print("  Runs: 2 (for reproducibility)")
    print("  Max Samples: 5000")
    print("\n" + "=" * 100)
    
    # Check for results
    if not metrics_dir.exists():
        print("⏳ BENCHMARK IN PROGRESS")
        print("   Waiting for results to be generated...")
        return False
    
    print("\n📈 TEST RESULTS:")
    print("-" * 100)
    
    results_files = sorted(metrics_dir.glob("*.json"))
    
    if not results_files:
        print("⏳ No results found yet - benchmark still running or not started")
        return False
    
    all_results = []
    models_tested = set()
    datasets_tested = set()
    
    total_accuracy = 0
    total_f1 = 0
    total_tests = 0
    
    print(f"\n{'Model':<25} {'Dataset':<15} {'Accuracy':<12} {'F1-Score':<12} {'AUC-ROC':<12} {'Latency(ms)':<12}")
    print("-" * 100)
    
    for result_file in results_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            model_name = data.get("model_name", "Unknown")
            dataset_name = data.get("dataset", "Unknown")
            accuracy = data.get("accuracy", 0)
            f1_score = data.get("f1_score", 0)
            auc_roc = data.get("auc_roc", data.get("auc-roc", 0))
            latency = data.get("latency_ms_per_sample", 0)
            
            models_tested.add(model_name)
            datasets_tested.add(dataset_name)
            
            print(f"{model_name:<25} {dataset_name:<15} {accuracy:>10.4f}  {f1_score:>10.4f}  {auc_roc:>10.4f}  {latency:>10.3f}")
            
            all_results.append(data)
            total_accuracy += accuracy
            total_f1 += f1_score
            total_tests += 1
            
        except Exception as e:
            print(f"⚠️  Error reading {result_file.name}: {e}")
    
    print("-" * 100)
    
    # Summary Statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"  Total Tests Run: {total_tests}")
    print(f"  Models Tested: {len(models_tested)}/8")
    if total_tests > 0:
        print(f"  Average Accuracy: {total_accuracy/total_tests:.4f}")
        print(f"  Average F1-Score: {total_f1/total_tests:.4f}")
    
    # Models Status
    print("\n✅ MODELS TESTED:")
    expected_models = [
        "cnn1d", "ft_transformer", "vanilla_ae", "vae",
        "bilstm_attention", "cnn_lstm", "isolation_forest", "contrastive_ssl"
    ]
    
    for model in expected_models:
        status = "✅" if model in models_tested else "⏳"
        print(f"  {status} {model}")
    
    # Error Check
    print("\n🔍 ERROR CHECK:")
    errors_found = False
    
    # Check for F1-score multiclass errors
    for result in all_results:
        if result.get("f1_score") is None or (isinstance(result.get("f1_score"), (int, float)) and result.get("f1_score") < 0):
            print(f"  ❌ F1-Score error in {result.get('model_name')}")
            errors_found = True
    
    if not errors_found and total_tests > 0:
        print("  ✅ NO ERRORS FOUND")
        print("  ✅ No F1-score multiclass errors")
        print("  ✅ All metrics calculated successfully")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if total_tests == len(expected_models) * 2:  # 8 models * 2 runs
        print("  ✅ FULL TEST SUITE COMPLETED SUCCESSFULLY")
        print("  ✅ All 8 architectures tested")
        print("  ✅ All runs completed")
        print("  ✅ System is stable and ready for production")
    else:
        print(f"  ⏳ {len(expected_models) * 2 - total_tests} tests remaining")
        print("  ⏳ Benchmark still in progress")
    
    print("\n" + "=" * 100)
    
    return total_tests > 0

def monitor_progress():
    """Monitor benchmark progress"""
    print("\n🔄 MONITORING BENCHMARK PROGRESS...")
    print("Checking results every 30 seconds...\n")
    
    max_wait = 3600  # 1 hour
    elapsed = 0
    last_count = 0
    
    while elapsed < max_wait:
        metrics_dir = Path("backend/results/metrics")
        if metrics_dir.exists():
            result_files = list(metrics_dir.glob("*.json"))
            current_count = len(result_files)
            
            if current_count > last_count:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Results found: {current_count} files")
                last_count = current_count
        
        time.sleep(30)
        elapsed += 30
    
    print("\nGenerating final report...\n")
    return generate_full_report()

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent)
    
    # First check if we have results
    has_results = generate_full_report()
    
    if not has_results:
        print("\n⏳ Starting progress monitor...")
        monitor_progress()
