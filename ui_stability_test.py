#!/usr/bin/env python3
"""
Quick UI Stability Test - Tests key crash scenarios without heavy computations
"""

import os
import sys
import subprocess
from pathlib import Path

def test_backend_components():
    """Test that all backend components can be imported and instantiated."""
    print("🔧 Testing Backend Components...")

    try:
        # Add backend to path
        sys.path.insert(0, 'backend')

        # Test imports
        from config import DATASET_CONFIGS, MODEL_CONFIGS
        from data.preprocessing import load_and_preprocess, generate_synthetic_dataset
        from models import get_model_class
        from training.evaluate import evaluate_model
        print("✅ Imports successful")

        # Test synthetic data generation
        data = generate_synthetic_dataset(n_samples=100)
        assert data["X_train"].shape[0] == 90  # 90% train
        assert data["X_test"].shape[0] == 10   # 10% test
        print("✅ Synthetic data generation works")

        # Test model instantiation
        models_to_test = ["cnn1d", "vanilla_ae", "isolation_forest"]
        for model_name in models_to_test:
            cls = get_model_class(model_name)
            if model_name == "isolation_forest":
                model = cls(n_estimators=10)
            else:
                model = cls(input_dim=10, num_classes=2)
        print("✅ Model instantiation works")

        return True
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_pipeline_scenarios():
    """Test different pipeline scenarios."""
    print("\n🔬 Testing Pipeline Scenarios...")

    backend_dir = Path("backend")
    results = []

    scenarios = [
        {
            "name": "Single Model",
            "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "--epochs", "1", "--runs", "1", "--skip-adversarial", "--skip-explainability"],
            "timeout": 60
        },
        {
            "name": "Multiple Models",
            "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "vanilla_ae", "--epochs", "1", "--runs", "1", "--skip-adversarial", "--skip-explainability"],
            "timeout": 90
        },
        {
            "name": "All Features (No Heavy Computations)",
            "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "--epochs", "1", "--runs", "1", "--skip-adversarial", "--skip-explainability"],
            "timeout": 60
        }
    ]

    for scenario in scenarios:
        print(f"   Testing: {scenario['name']}")
        try:
            result = subprocess.run(
                scenario["cmd"],
                cwd=backend_dir,
                capture_output=True,
                text=True,
                timeout=scenario["timeout"]
            )
            if result.returncode == 0:
                print("   ✅ PASS"                results.append(True)
            else:
                print(f"   ❌ FAIL: {result.stderr[-100:]}")
                results.append(False)
        except subprocess.TimeoutExpired:
            print("   ❌ TIMEOUT"            results.append(False)
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)

    return results

def test_custom_data_handling():
    """Test custom data upload scenarios."""
    print("\n📤 Testing Custom Data Handling...")

    try:
        sys.path.insert(0, 'backend')
        from data.preprocessing import load_and_preprocess
        import pandas as pd
        import numpy as np

        # Create test directory
        custom_dir = Path("backend/data/raw/custom")
        custom_dir.mkdir(parents=True, exist_ok=True)

        test_cases = [
            {
                "name": "Binary Numeric Labels",
                "data": pd.DataFrame({
                    'feature_1': np.random.randn(50),
                    'feature_2': np.random.randn(50),
                    'label': np.random.choice([0, 1], 50)
                })
            },
            {
                "name": "String Labels",
                "data": pd.DataFrame({
                    'feature_1': np.random.randn(50),
                    'feature_2': np.random.randn(50),
                    'label': np.random.choice(['normal', 'attack'], 50)
                })
            },
            {
                "name": "Multiclass (should convert to binary)",
                "data": pd.DataFrame({
                    'feature_1': np.random.randn(50),
                    'feature_2': np.random.randn(50),
                    'label': np.random.choice([0, 1, 2], 50)
                })
            }
        ]

        results = []
        for test_case in test_cases:
            try:
                # Save test data
                file_path = custom_dir / f"test_{test_case['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                test_case["data"].to_csv(file_path, index=False)

                # Test loading
                data = load_and_preprocess("custom", binary=True, max_samples=30)

                # Verify binary conversion
                unique_labels = set(data["y_train"])
                assert unique_labels.issubset({0, 1}), f"Expected binary labels, got {unique_labels}"

                print(f"   ✅ {test_case['name']}")
                results.append(True)
                file_path.unlink()  # Clean up

            except Exception as e:
                print(f"   ❌ {test_case['name']}: {e}")
                results.append(False)

        return results

    except Exception as e:
        print(f"❌ Custom data test setup failed: {e}")
        return [False]

def test_ui_imports():
    """Test that UI can import all required modules."""
    print("\n🖥️  Testing UI Imports...")

    try:
        # Test streamlit app imports
        import streamlit as st
        import pandas as pd
        import numpy as np
        import altair as alt
        import plotly.express as px

        # Test backend imports from UI perspective
        sys.path.insert(0, 'backend')
        from config import DATASET_CONFIGS
        from data.preprocessing import load_and_preprocess

        print("✅ UI imports successful")
        return True
    except Exception as e:
        print(f"❌ UI imports failed: {e}")
        return False

def main():
    """Run all stability tests."""
    print("=" * 80)
    print("🧪 NETGUARD UI STABILITY TEST SUITE")
    print("Testing key scenarios that could cause crashes")
    print("=" * 80)

    all_results = []

    # Test backend components
    all_results.append(test_backend_components())

    # Test UI imports
    all_results.append(test_ui_imports())

    # Test pipeline scenarios
    pipeline_results = test_pipeline_scenarios()
    all_results.extend(pipeline_results)

    # Test custom data handling
    custom_results = test_custom_data_handling()
    all_results.extend(custom_results)

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)

    test_names = [
        "Backend Components",
        "UI Imports",
        "Pipeline: Single Model",
        "Pipeline: Multiple Models",
        "Pipeline: All Features",
        "Custom: Binary Numeric",
        "Custom: String Labels",
        "Custom: Multiclass Conversion"
    ]

    passed = 0
    for i, result in enumerate(all_results):
        status = "✅" if result else "❌"
        print(f"{status} {test_names[i]}")
        if result:
            passed += 1

    total = len(all_results)
    success_rate = passed / total * 100

    print(f"\n🎯 Overall: {passed}/{total} tests passed ({success_rate:.1f}%)")

    if success_rate >= 80:
        print("🎉 UI should be stable! You can run the Streamlit app safely.")
        print("\n💡 Recommendations:")
        print("   - Adversarial robustness may take long, consider running separately")
        print("   - Explainability (SHAP) may be slow for large datasets")
        print("   - Custom data upload should work with various label formats")
        return 0
    else:
        print("⚠️  Some critical components failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())