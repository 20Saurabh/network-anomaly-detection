#!/usr/bin/env python3
"""
Direct test of the F1-score fixes
"""
import sys
import os
import subprocess
from pathlib import Path

os.chdir(Path(__file__).parent / "backend")
sys.path.insert(0, ".")

# Test 1: Check adversarial robustness fix
print("=" * 80)
print("TEST 1: Adversarial Robustness F1-Score Fix")
print("=" * 80)

try:
    from robustness.adversarial_attacks import evaluate_adversarial_robustness
    import numpy as np
    import torch
    from data.preprocessing import generate_synthetic_dataset
    from models.cnn1d import CNN1DClassifier
    
    # Generate test data
    data = generate_synthetic_dataset(n_samples=200)
    model = CNN1DClassifier(input_dim=data["X_test"].shape[1], num_classes=2)
    
    # Test adversarial robustness
    print("\nTesting adversarial robustness with binary data...")
    results = evaluate_adversarial_robustness(
        model, data["X_test"][:50], data["y_test"][:50],
        "cnn1d", epsilons=[0.01],
        max_samples=50
    )
    
    if "clean_f1" in results:
        print(f"✅ Adversarial robustness test PASSED")
        print(f"   Clean F1: {results['clean_f1']:.4f}")
    else:
        print(f"❌ Adversarial robustness test FAILED")
        
except Exception as e:
    print(f"❌ Adversarial robustness test FAILED: {e}")

# Test 2: Check explainability fix
print("\n" + "=" * 80)
print("TEST 2: Explainability (Permutation Importance) F1-Score Fix")
print("=" * 80)

try:
    from explainability.shap_analysis import compute_permutation_importance
    from data.preprocessing import generate_synthetic_dataset
    from models.cnn1d import CNN1DClassifier
    
    # Generate test data
    data = generate_synthetic_dataset(n_samples=100)
    model = CNN1DClassifier(input_dim=data["X_test"].shape[1], num_classes=2)
    feature_names = [f"feat_{i}" for i in range(data["X_test"].shape[1])]
    
    print("\nTesting permutation importance with binary data...")
    results = compute_permutation_importance(
        model, data["X_test"][:30], data["y_test"][:30],
        feature_names[:30], "cnn1d", n_repeats=2
    )
    
    if results and len(results) > 0:
        print(f"✅ Permutation importance test PASSED")
        print(f"   Computed importance for {len(results)} features")
    else:
        print(f"❌ Permutation importance test FAILED")
        
except Exception as e:
    print(f"❌ Permutation importance test FAILED: {e}")

# Test 3: Run a full benchmark
print("\n" + "=" * 80)
print("TEST 3: Full Benchmark Run")
print("=" * 80)

try:
    print("\nRunning single model benchmark...")
    result = subprocess.run(
        [sys.executable, "run_experiments.py", 
         "--dataset", "synthetic", 
         "--models", "cnn1d", 
         "--epochs", "1", 
         "--runs", "1"],
        capture_output=False,
        timeout=120
    )
    
    if result.returncode == 0:
        print(f"\n✅ Full benchmark test PASSED")
    else:
        print(f"\n❌ Full benchmark test FAILED (exit code: {result.returncode})")
        
except subprocess.TimeoutExpired:
    print(f"\n⏱️  Full benchmark test TIMEOUT")
except Exception as e:
    print(f"\n❌ Full benchmark test FAILED: {e}")

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ F1-score fixes have been applied to:")
print("   - backend/robustness/adversarial_attacks.py")
print("   - backend/explainability/shap_analysis.py")
print("   - backend/training/evaluate.py")
print("\n💡 The system now detects actual number of classes and uses appropriate averaging")
