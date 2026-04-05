#!/usr/bin/env python3
"""
Debug script to check model predictions on streaming data.
Tests why streaming simulation shows 0 anomalies detected.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import config
from config import DATASET_CONFIGS, set_seed
from data.preprocessing import _load_unsw_nb15, _to_binary
from data.dataloader import create_dataloaders
from models.ft_transformer import FTTransformer
from models.cnn1d import CNN1DClassifier
import torch

def debug_model_predictions():
    """Debug model predictions on streaming data."""
    print("=" * 80)
    print("DEBUGGING MODEL PREDICTIONS ON STREAMING DATA")
    print("=" * 80)

    # Set seed for reproducibility
    set_seed(42)

    # ── 1. Load and preprocess data (same as benchmark) ───────────────────
    print("1. LOADING AND PREPROCESSING DATA")
    print("-" * 40)

    cfg = DATASET_CONFIGS["unsw_nb15"]
    df = _load_unsw_nb15(cfg)

    # Preprocessing steps
    y_raw = df[cfg.label_col].copy()
    X = df.drop(columns=[cfg.label_col], errors="ignore")

    if cfg.attack_cat_col and cfg.attack_cat_col != cfg.label_col:
        if cfg.attack_cat_col in X.columns:
            X = X.drop(columns=[cfg.attack_cat_col])

    X = X.drop(columns=[c for c in cfg.drop_cols if c in X.columns], errors="ignore")

    for cat_col in cfg.categorical_cols:
        if cat_col in X.columns:
            le = LabelEncoder()
            X[cat_col] = le.fit_transform(X[cat_col].astype(str))

    y = _to_binary(y_raw, "unsw_nb15").values

    # Subsampling
    max_samples = 10000
    if max_samples and len(X) > max_samples:
        indices = np.random.choice(len(X), size=max_samples, replace=False)
        X_sub = X.iloc[indices].reset_index(drop=True)
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y

    # Train/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_sub, y_sub, test_size=cfg.test_ratio, random_state=42, stratify=y_sub
    )

    # Take streaming subset (first 1500 from test)
    streaming_size = 1500
    X_stream = X_test[:streaming_size].values
    y_stream = y_test[:streaming_size]

    print(f"Streaming data shape: {X_stream.shape}")
    print(f"True anomalies in streaming: {np.sum(y_stream)}/{len(y_stream)} ({np.sum(y_stream)/len(y_stream)*100:.2f}%)")
    print(f"First 20 true labels: {y_stream[:20]}")
    print()

    # ── 2. Load trained models ─────────────────────────────────────────────
    print("2. LOADING TRAINED MODELS")
    print("-" * 40)

    # Load results to find best model
    results_dir = Path("backend/results")
    metrics_file = results_dir / "metrics" / "all_results.json"

    if not metrics_file.exists():
        print(f"ERROR: Metrics file not found: {metrics_file}")
        return

    with open(metrics_file, "r") as f:
        all_results = json.load(f)

    # Find best model for unsw_nb15
    best_model = None
    best_f1 = 0
    for model_name, results in all_results.items():
        if results.get("dataset") == "UNSW-NB15":
            f1 = results.get("f1_score", 0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name

    if not best_model:
        print("Available models and datasets:")
        for name, results in all_results.items():
            print(f"  {name}: {results.get('dataset', 'unknown')}")
        print("ERROR: No trained model found for UNSW-NB15")
        return

    print(f"Best model: {best_model} (F1: {best_f1:.4f})")

    # Load model weights
    model_path = results_dir / "models" / f"{best_model}_best.pt"
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print(f"Available model files: {list((results_dir / 'models').glob('*.pt'))}")
        return

    # Create model instance
    input_dim = X_stream.shape[1]
    if "ft_transformer" in best_model:
        model = FTTransformer(
            input_dim=input_dim,
            num_classes=2,
            d_model=128,
            n_heads=8,
            n_layers=4,
            dropout=0.1
        )
    elif "cnn1d" in best_model:
        model = CNN1DClassifier(
            input_dim=input_dim,
            num_classes=2,
            channels=[64, 128, 256],
            kernel_size=3,
            dropout=0.1
        )
    else:
        print(f"ERROR: Unsupported model type: {best_model}")
        return

    # Load weights
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys

    # The checkpoint appears to be the state_dict directly
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Model loaded: {type(model).__name__}")
    print()

    # ── 3. Test model predictions ──────────────────────────────────────────
    print("3. TESTING MODEL PREDICTIONS")
    print("-" * 40)

    # Scale data if needed (check if scaler was used)
    scaler_path = results_dir / "models" / f"{best_model}_scaler.pkl"
    scaler = None
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        print("Loaded scaler, transforming data...")
        X_stream_scaled = scaler.transform(X_stream)
    else:
        print("No scaler found, using raw data")
        X_stream_scaled = X_stream

    # Make predictions
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_stream_scaled)
        output = model(x_tensor)

        if isinstance(output, tuple):
            output = output[0]

        # Get predictions based on model type
        if hasattr(model, 'model_type') and model.model_type == "unsupervised":
            # Reconstruction error based anomaly detection
            reconstruction = output
            errors = ((x_tensor - reconstruction) ** 2).mean(dim=-1).numpy()
            threshold = np.mean(errors) + 2 * np.std(errors)
            predictions = (errors > threshold).astype(int)
            scores = errors
            print("Unsupervised model: using reconstruction error")
        else:
            # Supervised classification
            probs = torch.softmax(output, dim=-1)
            predictions = output.argmax(dim=-1).numpy()
            scores = probs[:, 1].numpy() if probs.shape[1] > 1 else probs[:, 0].numpy()
            print("Supervised model: using classification")

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions unique: {np.unique(predictions, return_counts=True)}")
    print(f"Anomalies detected: {np.sum(predictions)}/{len(predictions)} ({np.sum(predictions)/len(predictions)*100:.2f}%)")

    print(f"\nFirst 20 predictions: {predictions[:20]}")
    print(f"First 20 true labels:  {y_stream[:20]}")

    # Compare with true labels
    correct = np.sum(predictions == y_stream)
    accuracy = correct / len(predictions)
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(predictions)})")

    # Confusion matrix
    tp = np.sum((predictions == 1) & (y_stream == 1))
    tn = np.sum((predictions == 0) & (y_stream == 0))
    fp = np.sum((predictions == 1) & (y_stream == 0))
    fn = np.sum((predictions == 0) & (y_stream == 1))

    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    if np.sum(predictions) == 0:
        print("\n🔴 CRITICAL: Model predicts NO anomalies!")
        print("Possible causes:")
        print("1. Model trained on different data distribution")
        print("2. Incorrect threshold for unsupervised models")
        print("3. Model not converged during training")
        print("4. Data preprocessing mismatch between training and inference")
    else:
        print("\n✅ Model does detect anomalies")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    debug_model_predictions()