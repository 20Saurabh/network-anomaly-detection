"""
Explainability module: SHAP analysis and permutation feature importance.
Critical for cybersecurity — analysts need to understand WHY a flow is flagged.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEVICE, PLOTS_DIR


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    model_name: str,
    max_background: int = 100,
    max_explain: int = 200,
):
    """
    Compute SHAP values for a model.
    Uses KernelExplainer (model-agnostic) for broad compatibility.
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed. pip install shap")
        return None

    from models.base import BaseSklearnModel

    # Subsample for speed
    bg = X_background[:max_background]
    exp = X_explain[:max_explain]

    # Create prediction function
    if isinstance(model, BaseSklearnModel):
        if hasattr(model.model, "predict_proba"):
            predict_fn = model.model.predict_proba
        else:
            predict_fn = lambda x: np.column_stack([
                1 - model.get_anomaly_scores(x),
                model.get_anomaly_scores(x)
            ])
    else:
        model.eval()
        def predict_fn(x):
            with torch.no_grad():
                t = torch.FloatTensor(x).to(DEVICE)
                if hasattr(model, 'model_type') and model.model_type == "unsupervised":
                    # For unsupervised, return anomaly scores as a 2-column probability-like matrix
                    scores = model.get_anomaly_scores(t) # This returns a numpy array
                    return np.column_stack([1 - scores, scores])
                
                out = model(t)
                if isinstance(out, tuple):
                    out = out[0]
                if out.shape[-1] > 1:
                    return torch.softmax(out, dim=-1).cpu().numpy()
                return out.cpu().numpy()

    print(f"  [SHAP] Computing values for {model_name}...")
    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(exp, nsamples=50)

    # Plot global feature importance
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs_shap = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[-20:]  # Top 20

    fnames = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in sorted_idx]
    ax.barh(fnames, mean_abs_shap[sorted_idx], color="#2196F3", alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"shap_{model_name}.png", dpi=300)
    plt.savefig(PLOTS_DIR / f"shap_{model_name}.pdf")
    plt.close()

    print(f"  [SHAP] Saved: shap_{model_name}.png/pdf")
    return sv


def compute_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    model_name: str,
    num_classes: int = 2,
    n_repeats: int = 5,
):
    """
    Model-agnostic permutation feature importance.
    Measures F1 drop when each feature is randomly shuffled.
    """
    from sklearn.metrics import f1_score as f1
    from models.base import BaseSklearnModel

    print(f"  [PERM] Computing permutation importance for {model_name}...")

    # Determine average parameter for f1_score
    unique_y = np.unique(y_test)
    if num_classes == 2 and set(unique_y).issubset({0, 1}):
        avg = "binary"
    else:
        avg = "macro"

    # Get baseline predictions
    if isinstance(model, BaseSklearnModel):
        y_base = model.predict(X_test)
    else:
        model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X_test).to(DEVICE)
            if hasattr(model, 'model_type') and model.model_type == "unsupervised":
                scores = model.get_anomaly_scores(t)
                threshold = np.percentile(scores, 95)
                y_base = (scores > threshold).astype(int)
            else:
                out = model(t)
                if isinstance(out, tuple):
                    out = out[0]
                y_base = out.argmax(dim=-1).cpu().numpy()

    # Final safety check for average='binary'
    if avg == "binary":
        unique_preds = np.unique(y_base)
        if not set(unique_preds).issubset({0, 1}):
            avg = "macro"

    base_f1 = f1(y_test, y_base, average=avg, zero_division=0)

    importances = []
    for i in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])

            if isinstance(model, BaseSklearnModel):
                y_perm = model.predict(X_perm)
            else:
                with torch.no_grad():
                    t = torch.FloatTensor(X_perm).to(DEVICE)
                    if hasattr(model, 'model_type') and model.model_type == "unsupervised":
                        scores_perm = model.get_anomaly_scores(t)
                        y_perm = (scores_perm > threshold).astype(int)
                    else:
                        out = model(t)
                        if isinstance(out, tuple):
                            out = out[0]
                        y_perm = out.argmax(dim=-1).cpu().numpy()

            perm_f1 = f1(y_test, y_perm, average=avg, zero_division=0)
            drops.append(base_f1 - perm_f1)

        importances.append({
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })

    # Sort and plot
    importances.sort(key=lambda x: x["importance_mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(importances))
    names = [imp["feature"] for imp in importances[:top_n]]
    means = [imp["importance_mean"] for imp in importances[:top_n]]
    stds = [imp["importance_std"] for imp in importances[:top_n]]

    ax.barh(names[::-1], means[::-1], xerr=stds[::-1],
           color="#4CAF50", alpha=0.8, capsize=3)
    ax.set_xlabel("F1 Drop (importance)")
    ax.set_title(f"Permutation Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"perm_importance_{model_name}.png", dpi=300)
    plt.savefig(PLOTS_DIR / f"perm_importance_{model_name}.pdf")
    plt.close()

    print(f"  [PERM] Saved: perm_importance_{model_name}.png/pdf")
    return importances
