"""
Comprehensive evaluation metrics for Network Anomaly Detection.
Tier 1: Standard classification metrics
Tier 2: Operational / security-specific metrics
"""
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEVICE, METRICS_DIR


def evaluate_model(
    model,
    data: Dict,
    model_name: str,
    loaders: Optional[Dict] = None,
    device: torch.device = DEVICE,
) -> Dict:
    """
    Comprehensive evaluation. Returns dict of all metrics.
    """
    from models.base import BaseSklearnModel

    X_test = data["X_test"]
    y_true = data["y_test"]
    num_classes = data["num_classes"]

    print(f"\n  Evaluating: {model_name}")

    # ── Get predictions ──────────────────────────────────────────
    start_time = time.time()

    if isinstance(model, BaseSklearnModel):
        y_pred = model.predict(X_test)
        scores = model.get_anomaly_scores(X_test)
        inference_time = time.time() - start_time
    else:
        model.eval()
        model = model.to(device)
        y_pred, scores = _predict_pytorch(model, X_test, device, num_classes)
        inference_time = time.time() - start_time

    # ── Tier 1: Standard metrics ──────────────────────────────────
    # Detect actual number of classes in the data to handle edge cases
    actual_num_classes = len(np.unique(y_true))
    avg = "binary" if actual_num_classes == 2 else "macro"

    metrics = {
        "model_name": model_name,
        "dataset": data.get("dataset_name", "unknown"),
        "num_test_samples": len(y_true),
        "num_classes": num_classes,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # AUC-ROC and AUC-PR
    try:
        if num_classes == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
            metrics["auc_pr"] = float(average_precision_score(y_true, scores))
        else:
            # One-vs-Rest for multiclass
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            if len(scores.shape) == 1:
                # Unsupervised models only produce single score
                metrics["auc_roc"] = float(roc_auc_score(y_true, scores, multi_class="ovr"))
            else:
                metrics["auc_roc"] = float(roc_auc_score(y_bin, scores, average="macro", multi_class="ovr"))
                metrics["auc_pr"] = float(average_precision_score(y_bin, scores, average="macro"))
    except Exception as e:
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0
        print(f"    Warning: AUC calculation failed: {e}")

    # ── Tier 2: Operational metrics ───────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["detection_rate"] = float(tp / (tp + fn + 1e-8))
        metrics["false_alarm_rate"] = float(fp / (fp + tn + 1e-8))
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

    metrics["inference_time_total"] = float(inference_time)
    metrics["latency_ms_per_sample"] = float(inference_time / len(y_true) * 1000)
    metrics["throughput_samples_per_sec"] = float(len(y_true) / max(inference_time, 1e-8))

    # Parameter count
    if hasattr(model, "count_parameters"):
        metrics["num_parameters"] = model.count_parameters()

    # Confusion matrix
    metrics["confusion_matrix"] = cm.tolist()

    # Print summary
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1_score']:.4f}")
    print(f"    AUC-ROC:   {metrics.get('auc_roc', 'N/A')}")
    print(f"    Latency:   {metrics['latency_ms_per_sample']:.3f} ms/sample")
    print(f"    Throughput: {metrics['throughput_samples_per_sec']:.0f} samples/sec")

    # Save
    save_path = METRICS_DIR / f"{model_name}_{data.get('dataset_name', 'unknown')}.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _predict_pytorch(model, X_test, device, num_classes):
    """Get predictions from a PyTorch model."""
    X_tensor = torch.FloatTensor(X_test).to(device)
    all_preds = []
    all_scores = []

    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]

            output = model(batch)

            if isinstance(output, tuple):
                output = output[0]  # For VAE: (recon, mu, logvar)

            if model.model_type == "unsupervised":
                # Reconstruction error as anomaly score
                if output.shape == batch.shape:
                    scores = ((batch - output) ** 2).mean(dim=-1)
                    # Threshold: scores > mean + 2*std
                    all_scores.append(scores.cpu().numpy())
                else:
                    probs = torch.softmax(output, dim=-1)
                    all_scores.append(probs.cpu().numpy())
            else:
                probs = torch.softmax(output, dim=-1)
                if num_classes == 2:
                    all_scores.append(probs[:, 1].cpu().numpy())
                else:
                    all_scores.append(probs.cpu().numpy())
                preds = output.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())

    if model.model_type == "unsupervised" and not all_preds:
        scores = np.concatenate(all_scores)
        threshold = np.mean(scores) + 2 * np.std(scores)
        y_pred = (scores > threshold).astype(int)
        return y_pred, scores
    else:
        y_pred = np.concatenate(all_preds)
        scores = np.concatenate(all_scores)
        return y_pred, scores


def aggregate_results(all_results: list) -> Dict:
    """Aggregate results across multiple runs for statistical analysis."""
    if not all_results:
        return {}

    keys = [k for k in all_results[0] if isinstance(all_results[0][k], (int, float))]

    agg = {}
    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
            agg[f"{key}_min"] = float(np.min(values))
            agg[f"{key}_max"] = float(np.max(values))

    return agg
