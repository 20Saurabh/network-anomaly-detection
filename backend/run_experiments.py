"""
Master Experiment Orchestrator — run_experiments.py
===================================================
Runs the full benchmark: data loading → model training → evaluation →
statistical tests → visualization → adversarial robustness → streaming.

Usage:
  python run_experiments.py                           # Synthetic data, all models
  python run_experiments.py --dataset unsw_nb15       # Real dataset
  python run_experiments.py --models cnn1d ft_transformer  # Specific models
  python run_experiments.py --epochs 10 --runs 1      # Quick test
  python run_experiments.py --skip-adversarial --skip-streaming  # Skip optional
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import (
    DEVICE, SEEDS, set_seed, TRAINING_CONFIG, MODEL_CONFIGS,
    TABULAR_MODELS, UNSUPERVISED_MODELS, update_results_dir
)
from data.preprocessing import load_and_preprocess, generate_synthetic_dataset
from data.dataloader import create_dataloaders
from training import train_model
from training.evaluate import evaluate_model, aggregate_results
from training.statistical_tests import compute_all_statistics
from visualization.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection Benchmark")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "ciciot2023", "edge_iiot", "unsw_nb15", "cicids2017", "custom"],
                        help="Dataset to use")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train (default: all tabular)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs for statistical analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Binary classification (default)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (for CPU)")
    parser.add_argument("--skip-adversarial", action="store_true",
                        help="Skip adversarial robustness evaluation")
    parser.add_argument("--skip-streaming", action="store_true",
                        help="Skip streaming simulation")
    parser.add_argument("--skip-explainability", action="store_true",
                        help="Skip SHAP/feature importance (slow)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Unique name/timestamp for the run to save history (e.g. 2026-04-05_120000)")
    return parser.parse_args()


def instantiate_model(model_name: str, input_dim: int, num_classes: int):
    """Create a model instance by name."""
    from models import get_model_class
    from models.base import BaseSklearnModel

    cls = get_model_class(model_name)

    # Check if it's an sklearn model
    if issubclass(cls, BaseSklearnModel):
        return cls(
            n_estimators=MODEL_CONFIGS.if_n_estimators,
            contamination=MODEL_CONFIGS.if_contamination,
        )

    # PyTorch model
    kwargs = {
        "input_dim": input_dim,
        "num_classes": num_classes,
    }

    if model_name == "vanilla_ae":
        kwargs["hidden_dims"] = MODEL_CONFIGS.ae_hidden_dims
        kwargs["dropout"] = MODEL_CONFIGS.ae_dropout
    elif model_name == "vae":
        kwargs["beta"] = MODEL_CONFIGS.vae_beta
    elif model_name == "cnn1d":
        kwargs["channels"] = MODEL_CONFIGS.cnn_channels
    elif model_name == "bilstm_attention":
        kwargs["hidden_dim"] = MODEL_CONFIGS.lstm_hidden
        kwargs["num_layers"] = MODEL_CONFIGS.lstm_layers
        kwargs["n_heads"] = MODEL_CONFIGS.attention_heads
    elif model_name == "cnn_lstm":
        kwargs["lstm_hidden"] = MODEL_CONFIGS.lstm_hidden
    elif model_name == "ft_transformer":
        kwargs["d_model"] = MODEL_CONFIGS.ft_d_model
        kwargs["n_heads"] = MODEL_CONFIGS.ft_n_heads
        kwargs["n_layers"] = MODEL_CONFIGS.ft_n_layers
        kwargs["dropout"] = MODEL_CONFIGS.ft_dropout
    elif model_name == "e_graphsage":
        kwargs["hidden_dim"] = MODEL_CONFIGS.gnn_hidden
        kwargs["out_dim"] = MODEL_CONFIGS.gnn_out
    elif model_name == "gnn_transformer":
        kwargs["gnn_hidden"] = MODEL_CONFIGS.gnn_hidden
        kwargs["gnn_out"] = MODEL_CONFIGS.gnn_out
    elif model_name == "contrastive_ssl":
        kwargs["projection_dim"] = MODEL_CONFIGS.ssl_projection_dim
        kwargs["temperature"] = MODEL_CONFIGS.ssl_temperature

    return cls(**kwargs)


def run_single_experiment(data, model_names, args):
    """Run one full experiment (train + evaluate all models)."""
    input_dim = data["X_train"].shape[1]
    num_classes = data["num_classes"]

    all_results = {}
    all_histories = {}
    all_models = {}
    all_scores = {}
    all_y_tests = {}

    for model_name in model_names:
        if model_name == "ensemble_stacking":
            continue  # Handle after all base models

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        try:
            model = instantiate_model(model_name, input_dim, num_classes)

            # Create appropriate dataloader
            from models.base import BaseSklearnModel
            if isinstance(model, BaseSklearnModel):
                history = train_model(model, {}, model_name, data)
                result = evaluate_model(model, data, model_name)
            else:
                model_type = "autoencoder" if model.model_type == "unsupervised" else "supervised"
                loaders = create_dataloaders(
                    data, model_type=model_type,
                    use_weighted_sampler=(model_type == "supervised")
                )
                epochs = args.epochs or TRAINING_CONFIG.epochs
                history = train_model(model, loaders, model_name, data, epochs=epochs)
                result = evaluate_model(model, data, model_name, loaders)

            all_results[model_name] = result
            all_histories[model_name] = history
            all_models[model_name] = model

            # Store scores for ROC/PR curves
            all_y_tests[model_name] = data["y_test"]
            if isinstance(model, BaseSklearnModel):
                all_scores[model_name] = model.get_anomaly_scores(data["X_test"])
            else:
                model.eval()
                with torch.no_grad():
                    x_t = torch.FloatTensor(data["X_test"]).to(DEVICE)
                    scores = model.get_anomaly_scores(x_t)
                    all_scores[model_name] = scores

        except Exception as e:
            print(f"  ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

    # ── Ensemble Stacking ─────────────────────────────────────────
    if "ensemble_stacking" in model_names and len(all_models) >= 2:
        print(f"\n{'='*60}")
        print(f"  MODEL: ensemble_stacking")
        print(f"{'='*60}")
        try:
            _run_ensemble(all_models, all_results, all_histories, data,
                         all_scores, all_y_tests)
        except Exception as e:
            print(f"  ERROR with ensemble: {e}")

    return all_results, all_histories, all_models, all_scores, all_y_tests


def _run_ensemble(all_models, all_results, all_histories, data,
                  all_scores, all_y_tests):
    """Build stacking ensemble from base model predictions."""
    from models.ensemble_stacking import EnsembleStacking
    from models.base import BaseSklearnModel

    base_names = [n for n in all_models if n != "ensemble_stacking"]

    # Collect val predictions from each model
    X_meta_val = []
    X_meta_test = []

    for name in base_names:
        model = all_models[name]
        if isinstance(model, BaseSklearnModel):
            val_scores = model.get_anomaly_scores(data["X_val"])
            test_scores = model.get_anomaly_scores(data["X_test"])
        else:
            model.eval()
            with torch.no_grad():
                val_scores = model.get_anomaly_scores(
                    torch.FloatTensor(data["X_val"]).to(DEVICE)
                )
                test_scores = model.get_anomaly_scores(
                    torch.FloatTensor(data["X_test"]).to(DEVICE)
                )
        X_meta_val.append(val_scores.reshape(-1, 1) if val_scores.ndim == 1 else val_scores)
        X_meta_test.append(test_scores.reshape(-1, 1) if test_scores.ndim == 1 else test_scores)

    X_meta_val = np.hstack(X_meta_val)
    X_meta_test = np.hstack(X_meta_test)

    ensemble = EnsembleStacking()
    ensemble.fit(X_meta_val, data["y_val"], base_model_names=base_names)

    # Evaluate
    y_pred = ensemble.predict(X_meta_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

    y_test = data["y_test"]
    avg = "binary" if data["num_classes"] == 2 else "macro"
    scores = ensemble.get_anomaly_scores(X_meta_test)

    result = {
        "model_name": "ensemble_stacking",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "num_parameters": ensemble.count_parameters(),
        "base_models": base_names,
    }

    try:
        result["auc_roc"] = float(roc_auc_score(y_test, scores))
        result["auc_pr"] = float(average_precision_score(y_test, scores))
    except Exception:
        result["auc_roc"] = 0.0
        result["auc_pr"] = 0.0

    print(f"    Ensemble F1: {result['f1_score']:.4f}, AUC-ROC: {result['auc_roc']:.4f}")

    all_results["ensemble_stacking"] = result
    all_histories["ensemble_stacking"] = {"train_loss": [], "val_loss": []}
    all_scores["ensemble_stacking"] = scores
    all_y_tests["ensemble_stacking"] = y_test


def main():
    args = parse_args()
    total_start = time.time()

    print("\n" + "=" * 60)
    print("  NETWORK ANOMALY DETECTION BENCHMARK v2.0")
    print("  Next-Generation Research-Grade System")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Runs: {args.runs}")

    # ── Load Data ────────────────────────────────────────────────
    set_seed(args.seed)
    if args.dataset == "synthetic":
        data = generate_synthetic_dataset(
            n_samples=args.max_samples or 10000,
            n_features=46
        )
    else:
        data = load_and_preprocess(
            args.dataset, binary=args.binary,
            max_samples=args.max_samples, seed=args.seed
        )

    # ── Select Models ────────────────────────────────────────────
    if args.models:
        model_names = args.models
    else:
        # Default: all tabular models (GNN needs PyG)
        model_names = [
            "vanilla_ae", "vae", "cnn1d", "bilstm_attention",
            "cnn_lstm", "ft_transformer", "isolation_forest",
            "contrastive_ssl", "ensemble_stacking",
        ]
        # Try GNN models
        try:
            import torch_geometric
            model_names.insert(-1, "e_graphsage")
            model_names.insert(-1, "gnn_transformer")
        except ImportError:
            print("  [INFO] PyTorch Geometric not installed — skipping GNN models")

    print(f"  Models: {model_names}")

    # ── Multi-Run Experiments ────────────────────────────────────
    multi_run_results = defaultdict(list)
    final_results = {}
    final_histories = {}
    final_models = {}
    final_scores = {}
    final_y_tests = {}

    seeds = SEEDS[:args.runs]
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"  RUN {run_idx + 1}/{args.runs} (seed={seed})")
        print(f"{'#'*60}")

        set_seed(seed)

        results, histories, models, scores, y_tests = run_single_experiment(
            data, model_names, args
        )

        for model_name, result in results.items():
            multi_run_results[model_name].append(result)

        # Keep last run's artifacts for plotting
        final_results = results
        final_histories = histories
        final_models = models
        final_scores = scores
        final_y_tests = y_tests

    # ── Statistical Tests ────────────────────────────────────────
    if args.runs >= 3:
        print(f"\n{'='*60}")
        print(f"  STATISTICAL ANALYSIS")
        print(f"{'='*60}")
        stats = compute_all_statistics(multi_run_results)
        stats_path = METRICS_DIR / "statistical_tests.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  Saved: {stats_path}")

        # Print Friedman test result
        for key, val in stats.items():
            if key.startswith("friedman_"):
                metric = key.replace("friedman_", "")
                if "ranking" in val:
                    print(f"\n  Friedman test ({metric}):")
                    print(f"    p-value: {val.get('p_value', 'N/A')}")
                    print(f"    Ranking: {val['ranking']}")

    # ── Aggregate Results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    summary_table = []
    for model_name in model_names:
        if model_name in final_results and "error" not in final_results[model_name]:
            r = final_results[model_name]
            summary_table.append({
                "Model": model_name,
                "Accuracy": f"{r.get('accuracy', 0):.4f}",
                "Precision": f"{r.get('precision', 0):.4f}",
                "Recall": f"{r.get('recall', 0):.4f}",
                "F1": f"{r.get('f1_score', 0):.4f}",
                "AUC-ROC": f"{r.get('auc_roc', 0):.4f}",
                "Latency(ms)": f"{r.get('latency_ms_per_sample', 0):.3f}",
            })

    # Print table
    if summary_table:
        headers = list(summary_table[0].keys())
        widths = [max(len(h), max(len(r[h]) for r in summary_table)) for h in headers]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = "-+-".join("-" * w for w in widths)
        print(f"  {header_line}")
        print(f"  {sep_line}")
        for row in summary_table:
            print(f"  {' | '.join(row[h].ljust(w) for h, w in zip(headers, widths))}")

    # ── Save all results ─────────────────────────────────────────
    all_results_path = config.METRICS_DIR / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n  Results saved: {all_results_path}")

    # ── Plots ────────────────────────────────────────────────────
    if not args.skip_plots:
        generate_all_plots(
            final_results, final_histories,
            multi_run_results if args.runs > 1 else None,
            final_y_tests, final_scores,
            dataset_name=args.dataset
        )

    # ── Adversarial Robustness ───────────────────────────────────
    if not args.skip_adversarial:
        print(f"\n{'='*60}")
        print(f"  ADVERSARIAL ROBUSTNESS EVALUATION")
        print(f"{'='*60}")
        from robustness.adversarial_attacks import (
            evaluate_adversarial_robustness, plot_robustness_curves
        )
        all_robustness = {}
        for model_name, model in final_models.items():
            rob = evaluate_adversarial_robustness(
                model, data["X_test"], data["y_test"], model_name,
                num_classes=data["num_classes"]
            )
            if rob:
                all_robustness[model_name] = rob

        if all_robustness:
            plot_robustness_curves(all_robustness)
            rob_path = config.METRICS_DIR / "adversarial_robustness.json"
            with open(rob_path, "w") as f:
                json.dump(all_robustness, f, indent=2, default=str)

    # ── Explainability ───────────────────────────────────────────
    if not args.skip_explainability:
        print(f"\n{'='*60}")
        print(f"  EXPLAINABILITY ANALYSIS")
        print(f"{'='*60}")
        from explainability.shap_analysis import (
            compute_permutation_importance
        )
        # Permutation importance for best model
        best_model_name = max(
            [m for m in final_results if "error" not in final_results[m]],
            key=lambda m: final_results[m].get("f1_score", 0),
            default=None
        )
        if best_model_name and best_model_name in final_models:
            compute_permutation_importance(
                final_models[best_model_name],
                data["X_test"][:1000],
                data["y_test"][:1000],
                data["feature_names"],
                best_model_name
            )

    # ── Streaming Simulation ─────────────────────────────────────
    if not args.skip_streaming:
        print(f"\n{'='*60}")
        print(f"  STREAMING SIMULATION")
        print(f"{'='*60}")
        from runtime.streaming import StreamingInferencePipeline

        best_model_name = max(
            [m for m in final_results if "error" not in final_results[m]],
            key=lambda m: final_results[m].get("f1_score", 0),
            default=None
        )
        if best_model_name and best_model_name in final_models:
            pipeline = StreamingInferencePipeline(
                final_models[best_model_name],
                scaler=data.get("scaler")
            )
            stream_results = pipeline.simulate_stream(data["X_test"])
            stream_path = config.METRICS_DIR / "streaming_results.json"
            with open(stream_path, "w") as f:
                json.dump(stream_results, f, indent=2, default=str)

    # ── Final Summary ────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Results: {config.RESULTS_DIR}")
    print(f"  Plots:   {config.PLOTS_DIR}")
    print(f"  Metrics: {config.METRICS_DIR}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    import sys
    args = parse_args()
    if args.run_name:
        update_results_dir(args.run_name)
    # The original main() read from parse_args() internally or sys.argv
    # Let me ensure main uses global args
    sys.argv.extend([]) # Keep the parsing clean
    main()
