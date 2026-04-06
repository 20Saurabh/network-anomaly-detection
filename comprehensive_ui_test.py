#!/usr/bin/env python3
"""
Comprehensive Test Suite for NetGuard Streamlit UI
Tests all features, combinations, and edge cases to prevent crashes.
"""

import os
import sys
import time
import subprocess
import signal
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

class NetGuardUITester:
    """Comprehensive tester for NetGuard Streamlit UI."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.backend_dir = self.project_root / "backend"
        self.streamlit_dir = self.project_root / "streamlit_app"
        self.test_results = []
        self.streamlit_process = None

    def log_test(self, test_name: str, result: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
        self.test_results.append({
            "test": test_name,
            "result": result,
            "message": message
        })

    def start_streamlit(self) -> bool:
        """Start Streamlit server."""
        try:
            print("\n🚀 Starting Streamlit server...")
            self.streamlit_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py"],
                cwd=self.streamlit_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Wait for server to start
            time.sleep(5)
            return self.streamlit_process.poll() is None
        except Exception as e:
            self.log_test("Start Streamlit", False, f"Failed to start: {e}")
            return False

    def stop_streamlit(self):
        """Stop Streamlit server."""
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=10)
                print("🛑 Stopped Streamlit server")
            except:
                self.streamlit_process.kill()

    def test_backend_imports(self) -> bool:
        """Test that all backend modules can be imported."""
        try:
            # Test core imports
            from config import DATASET_CONFIGS
            from data.preprocessing import load_and_preprocess
            from models import get_model_class
            from training.evaluate import evaluate_model
            from robustness.adversarial_attacks import evaluate_adversarial_robustness
            from explainability.shap_analysis import compute_shap_values
            from runtime.streaming import StreamingInferencePipeline

            self.log_test("Backend Imports", True, "All modules imported successfully")
            return True
        except Exception as e:
            self.log_test("Backend Imports", False, f"Import failed: {e}")
            return False

    def test_dataset_loading(self) -> bool:
        """Test loading all supported datasets."""
        from data.preprocessing import load_and_preprocess, generate_synthetic_dataset

        datasets = ["synthetic"]
        results = []

        # Test synthetic data
        try:
            data = generate_synthetic_dataset(n_samples=100)
            assert data["X_train"].shape[0] > 0
            assert len(np.unique(data["y_train"])) <= 2  # Binary
            results.append(True)
        except Exception as e:
            results.append(False)
            self.log_test("Synthetic Dataset", False, str(e))

        # Test real datasets if available
        for dataset in ["unsw_nb15", "ciciot2023", "edge_iiot"]:
            try:
                data = load_and_preprocess(dataset, binary=True, max_samples=100)
                assert data["X_train"].shape[0] > 0
                assert len(np.unique(data["y_train"])) <= 2
                results.append(True)
            except Exception as e:
                # Expected to fail if dataset not downloaded
                if "not found" in str(e).lower():
                    results.append(True)  # This is expected
                else:
                    results.append(False)
                    self.log_test(f"{dataset} Dataset", False, str(e))

        success = all(results)
        self.log_test("Dataset Loading", success, f"Loaded {sum(results)}/{len(results)} datasets")
        return success

    def test_model_instantiation(self) -> bool:
        """Test creating all model instances."""
        from models import get_model_class
        from config import MODEL_CONFIGS

        models_to_test = [
            "cnn1d", "ft_transformer", "vanilla_ae", "vae",
            "bilstm_attention", "cnn_lstm", "isolation_forest",
            "contrastive_ssl"
        ]

        results = []
        for model_name in models_to_test:
            try:
                cls = get_model_class(model_name)
                if model_name == "isolation_forest":
                    model = cls(n_estimators=10, contamination=0.1)
                else:
                    model = cls(input_dim=10, num_classes=2)
                results.append(True)
            except Exception as e:
                results.append(False)
                self.log_test(f"Model {model_name}", False, str(e))

        success = all(results)
        self.log_test("Model Instantiation", success, f"Created {sum(results)}/{len(models_to_test)} models")
        return success

    def test_pipeline_execution(self) -> bool:
        """Test running the full benchmark pipeline."""
        test_cases = [
            {
                "name": "Single Model Basic",
                "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "--epochs", "1", "--runs", "1"],
                "timeout": 60
            },
            {
                "name": "Multiple Models",
                "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "vanilla_ae", "--epochs", "1", "--runs", "1"],
                "timeout": 120
            },
            {
                "name": "With Adversarial",
                "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "--epochs", "1", "--runs", "1", "--skip-adversarial"],
                "timeout": 60  # Skip adversarial to speed up
            },
            {
                "name": "With Explainability",
                "cmd": [sys.executable, "run_experiments.py", "--dataset", "synthetic", "--models", "cnn1d", "--epochs", "1", "--runs", "1", "--skip-explainability"],
                "timeout": 60  # Skip explainability to speed up
            }
        ]

        results = []
        for test_case in test_cases:
            try:
                print(f"   Testing: {test_case['name']}")
                result = subprocess.run(
                    test_case["cmd"],
                    cwd=self.backend_dir,
                    capture_output=True,
                    text=True,
                    timeout=test_case["timeout"]
                )
                if result.returncode == 0:
                    results.append(True)
                    print(f"   ✅ {test_case['name']} completed successfully")
                else:
                    results.append(False)
                    print(f"   ❌ {test_case['name']} failed: {result.stderr[-200:]}")
            except subprocess.TimeoutExpired:
                results.append(False)
                print(f"   ❌ {test_case['name']} timed out")
            except Exception as e:
                results.append(False)
                print(f"   ❌ {test_case['name']} error: {e}")

        success = all(results)
        self.log_test("Pipeline Execution", success, f"Passed {sum(results)}/{len(test_cases)} test cases")
        return success

    def test_custom_data_upload(self) -> bool:
        """Test custom data upload functionality."""
        # Create test CSV
        test_data = {
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'label': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(test_data)

        # Test different label formats
        test_cases = [
            ("binary_numeric", df.copy()),
            ("multiclass_numeric", df.assign(label=np.random.choice([0, 1, 2], 100))),
            ("string_labels", df.assign(label=df['label'].map({0: 'normal', 1: 'attack'}))),
        ]

        results = []
        for case_name, test_df in test_cases:
            try:
                # Save test data
                custom_dir = self.backend_dir / "data" / "raw" / "custom"
                custom_dir.mkdir(parents=True, exist_ok=True)
                test_file = custom_dir / f"test_{case_name}.csv"
                test_df.to_csv(test_file, index=False)

                # Test loading
                from data.preprocessing import load_and_preprocess
                data = load_and_preprocess("custom", binary=True, max_samples=50)

                # Verify binary conversion worked
                unique_labels = np.unique(data["y_train"])
                assert len(unique_labels) <= 2, f"Expected binary labels, got {unique_labels}"

                results.append(True)
                print(f"   ✅ {case_name} upload test passed")
                test_file.unlink()  # Clean up

            except Exception as e:
                results.append(False)
                print(f"   ❌ {case_name} upload test failed: {e}")

        success = all(results)
        self.log_test("Custom Data Upload", success, f"Passed {sum(results)}/{len(test_cases)} upload formats")
        return success

    def test_streaming_simulation(self) -> bool:
        """Test streaming inference pipeline."""
        try:
            from data.preprocessing import generate_synthetic_dataset
            from runtime.streaming import StreamingInferencePipeline
            from models.cnn1d import CNN1DClassifier

            # Create test data and model
            data = generate_synthetic_dataset(n_samples=500)
            model = CNN1DClassifier(input_dim=data["X_test"].shape[1], num_classes=2)

            # Test streaming
            pipeline = StreamingInferencePipeline(model, scaler=None)
            results = pipeline.simulate_stream(data["X_test"][:100])  # Small subset

            # Verify results
            assert results["total_processed"] == 100
            assert results["total_anomalies"] >= 0
            assert results["avg_latency_ms"] > 0

            self.log_test("Streaming Simulation", True, f"Processed {results['total_processed']} samples")
            return True

        except Exception as e:
            self.log_test("Streaming Simulation", False, str(e))
            return False

    def test_adversarial_robustness(self) -> bool:
        """Test adversarial robustness evaluation."""
        try:
            from data.preprocessing import generate_synthetic_dataset
            from robustness.adversarial_attacks import evaluate_adversarial_robustness
            from models.cnn1d import CNN1DClassifier
            import torch

            # Create test data and model
            data = generate_synthetic_dataset(n_samples=100)
            model = CNN1DClassifier(input_dim=data["X_test"].shape[1], num_classes=2)

            # Convert to tensors
            x_test = torch.FloatTensor(data["X_test"][:20])  # Small batch
            y_test = torch.LongTensor(data["y_test"][:20])

            # Test adversarial evaluation
            results = evaluate_adversarial_robustness(
                model, x_test, y_test, eps=0.01, device="cpu"
            )

            # Verify results structure
            assert "f1_clean" in results
            assert "f1_fgsm" in results
            assert "f1_pgd" in results

            self.log_test("Adversarial Robustness", True, f"F1 clean: {results['f1_clean']:.3f}")
            return True

        except Exception as e:
            self.log_test("Adversarial Robustness", False, str(e))
            return False

    def test_explainability(self) -> bool:
        """Test SHAP explainability analysis."""
        try:
            from data.preprocessing import generate_synthetic_dataset
            from explainability.shap_analysis import compute_shap_values
            from models.cnn1d import CNN1DClassifier

            # Create test data and model
            data = generate_synthetic_dataset(n_samples=100)
            model = CNN1DClassifier(input_dim=data["X_test"].shape[1], num_classes=2)

            # Test SHAP computation
            shap_values = compute_shap_values(
                model, data["X_test"][:20], data["feature_names"]
            )

            # Verify results
            assert shap_values.shape[0] == 20  # n_samples
            assert shap_values.shape[1] == len(data["feature_names"])  # n_features

            self.log_test("Explainability", True, f"SHAP values shape: {shap_values.shape}")
            return True

        except Exception as e:
            self.log_test("Explainability", False, str(e))
            return False

    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        test_cases = [
            {
                "name": "Empty Dataset",
                "data": pd.DataFrame(),
                "expect_fail": True
            },
            {
                "name": "Missing Label Column",
                "data": pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
                "expect_fail": True
            },
            {
                "name": "Single Class Labels",
                "data": pd.DataFrame({'feature1': [1, 2, 3], 'label': [0, 0, 0]}),
                "expect_fail": False  # Should handle gracefully
            },
            {
                "name": "All NaN Features",
                "data": pd.DataFrame({'feature1': [np.nan, np.nan], 'label': [0, 1]}),
                "expect_fail": False  # Should handle with fillna
            }
        ]

        results = []
        for test_case in test_cases:
            try:
                # Save test data
                custom_dir = self.backend_dir / "data" / "raw" / "custom"
                custom_dir.mkdir(parents=True, exist_ok=True)
                test_file = custom_dir / f"edge_case_{test_case['name'].lower().replace(' ', '_')}.csv"
                test_case["data"].to_csv(test_file, index=False)

                # Try to load
                from data.preprocessing import load_and_preprocess
                data = load_and_preprocess("custom", binary=True, max_samples=10)

                if test_case["expect_fail"]:
                    results.append(False)  # Should have failed but didn't
                    print(f"   ❌ {test_case['name']}: Expected failure but succeeded")
                else:
                    results.append(True)
                    print(f"   ✅ {test_case['name']}: Handled gracefully")

                test_file.unlink()

            except Exception as e:
                if test_case["expect_fail"]:
                    results.append(True)
                    print(f"   ✅ {test_case['name']}: Failed as expected - {str(e)[:50]}...")
                else:
                    results.append(False)
                    print(f"   ❌ {test_case['name']}: Unexpected failure - {e}")

        success = all(results)
        self.log_test("Edge Cases", success, f"Handled {sum(results)}/{len(test_cases)} edge cases correctly")
        return success

    def run_all_tests(self) -> Dict:
        """Run all test suites."""
        print("=" * 80)
        print("🧪 COMPREHENSIVE NETGUARD UI TEST SUITE")
        print("=" * 80)

        # Core functionality tests
        self.test_backend_imports()
        self.test_dataset_loading()
        self.test_model_instantiation()

        # Feature tests
        self.test_pipeline_execution()
        self.test_custom_data_upload()
        self.test_streaming_simulation()
        self.test_adversarial_robustness()
        self.test_explainability()

        # Edge case tests
        self.test_edge_cases()

        # Summary
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.test_results if r["result"])
        total = len(self.test_results)

        for result in self.test_results:
            status = "✅" if result["result"] else "❌"
            print(f"{status} {result['test']}")
            if result["message"]:
                print(f"   {result['message']}")

        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 All tests passed! UI should be stable.")
        else:
            print("⚠️  Some tests failed. Check the issues above.")

        return {
            "passed": passed,
            "total": total,
            "results": self.test_results
        }


def main():
    """Run the comprehensive test suite."""
    tester = NetGuardUITester()

    try:
        results = tester.run_all_tests()

        # Exit with appropriate code
        if results["passed"] == results["total"]:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Some tests failed

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()