"""
Publication-quality visualization for all experiment results.
Generates: ROC curves, PR curves, confusion matrices, bar charts,
box plots, training curves, t-SNE embeddings, radar charts.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PLOTS_DIR

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    "#8BC34A",
]


def plot_training_curves(histories: Dict[str, Dict], save_name: str = "training_curves"):
    """Plot training + validation loss for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, hist) in enumerate(histories.items()):
        color = COLORS[i % len(COLORS)]
        if hist.get("train_loss"):
            ax1.plot(hist["train_loss"], label=name, color=color, alpha=0.8)
            ax2.plot(hist["val_loss"], label=name, color=color, alpha=0.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.legend(loc="upper right", ncol=2)
    ax1.set_yscale("log")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Convergence")
    ax2.legend(loc="upper right", ncol=2)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")


def plot_roc_curves(all_results: Dict[str, Dict], y_tests: Dict[str, np.ndarray],
                     scores: Dict[str, np.ndarray], save_name: str = "roc_curves"):
    """ROC curves for all models on single plot."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, result) in enumerate(all_results.items()):
        if name in scores and name in y_tests:
            try:
                fpr, tpr, _ = roc_curve(y_tests[name], scores[name])
                auc = result.get("auc_roc", 0)
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                       color=COLORS[i % len(COLORS)], linewidth=1.5)
            except Exception:
                pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")


def plot_precision_recall_curves(all_results, y_tests, scores,
                                  save_name="pr_curves"):
    """Precision-Recall curves — critical for imbalanced NIDS data."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, result) in enumerate(all_results.items()):
        if name in scores and name in y_tests:
            try:
                precision, recall, _ = precision_recall_curve(y_tests[name], scores[name])
                auc_pr = result.get("auc_pr", 0)
                ax.plot(recall, precision, label=f"{name} (AP={auc_pr:.3f})",
                       color=COLORS[i % len(COLORS)], linewidth=1.5)
            except Exception:
                pass

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")


def plot_confusion_matrices(all_results: Dict, save_name: str = "confusion_matrices"):
    """Confusion matrices as heatmaps for all models."""
    models_with_cm = {k: v for k, v in all_results.items() if "confusion_matrix" in v}
    n = len(models_with_cm)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, result) in enumerate(models_with_cm.items()):
        cm = np.array(result["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   ax=axes[i], cbar=False)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")


def plot_metric_comparison(all_results: Dict, save_name: str = "metric_comparison"):
    """Bar chart comparing key metrics across models."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    model_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 1.5), 6))

    x = np.arange(len(model_names))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = [all_results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(),
                     color=COLORS[i], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")


def plot_latency_f1_pareto(all_results: Dict, save_name: str = "latency_f1_pareto"):
    """Latency vs F1 Pareto frontier — efficiency-accuracy tradeoff."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, result) in enumerate(all_results.items()):
        f1 = result.get("f1_score", 0)
        latency = result.get("latency_ms_per_sample", 0)
        params = result.get("num_parameters", 0)

        size = max(20, min(300, params / 1000)) if params else 50
        ax.scatter(latency, f1, s=size, color=COLORS[i % len(COLORS)],
                  alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.annotate(name, (latency, f1), textcoords="offset points",
                   xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Latency (ms/sample)")
    ax.set_ylabel("F1-Score")
    ax.set_title("Efficiency-Accuracy Tradeoff")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")


def plot_box_plots(multi_run_results: Dict[str, List[Dict]],
                   metric: str = "f1_score", save_name: str = "box_plots"):
    """Box plots of metrics across multiple runs."""
    model_names = list(multi_run_results.keys())
    data = []
    labels = []
    for name in model_names:
        values = [r.get(metric, 0) for r in multi_run_results[name]]
        data.append(values)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.6)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} Distribution Across Runs")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}_{metric}.png")
    plt.savefig(PLOTS_DIR / f"{save_name}_{metric}.pdf")
    plt.close()
    print(f"  Saved: {save_name}_{metric}.png/pdf")


def generate_all_plots(all_results, histories=None, multi_run_results=None,
                       y_tests=None, scores_dict=None, dataset_name=""):
    """Generate all publication-quality plots."""
    print(f"\n{'='*60}")
    print(f"  Generating plots for: {dataset_name}")
    print(f"{'='*60}")

    suffix = f"_{dataset_name}" if dataset_name else ""

    if histories:
        plot_training_curves(histories, f"training_curves{suffix}")

    if all_results:
        plot_metric_comparison(all_results, f"metric_comparison{suffix}")
        plot_confusion_matrices(all_results, f"confusion_matrices{suffix}")
        plot_latency_f1_pareto(all_results, f"latency_f1{suffix}")

    if y_tests and scores_dict:
        plot_roc_curves(all_results, y_tests, scores_dict, f"roc_curves{suffix}")
        plot_precision_recall_curves(all_results, y_tests, scores_dict, f"pr_curves{suffix}")

    if multi_run_results:
        for metric in ["f1_score", "auc_roc", "precision", "recall"]:
            plot_box_plots(multi_run_results, metric, f"box_plots{suffix}")

    print(f"  All plots saved to: {PLOTS_DIR}")
