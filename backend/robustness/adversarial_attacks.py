"""
Adversarial robustness evaluation.
Attacks: FGSM (single-step), PGD (iterative) with feature constraints.
Evaluates model robustness under adversarial evasion attacks.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEVICE, ADVERSARIAL_EPSILONS, PGD_STEPS, PGD_ALPHA, PLOTS_DIR
)


def fgsm_attack(model, x, y, epsilon, device=DEVICE):
    """
    FGSM Attack: x_adv = x + ε * sign(∇_x L(θ, x, y))
    """
    x_adv = x.clone().detach().to(device).requires_grad_(True)
    y = y.to(device)

    model.eval()
    output = model(x_adv)
    if isinstance(output, tuple):
        output = output[0]

    loss = F.cross_entropy(output, y)
    loss.backward()

    perturbation = epsilon * x_adv.grad.sign()
    x_adv = (x_adv + perturbation).detach()

    return x_adv


def pgd_attack(model, x, y, epsilon, steps=PGD_STEPS,
               alpha=PGD_ALPHA, device=DEVICE):
    """
    PGD Attack: Iterative FGSM with projection to ε-ball.
    """
    x_orig = x.clone().detach().to(device)
    x_adv = x_orig.clone().detach()
    y = y.to(device)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        model.eval()
        output = model(x_adv)
        if isinstance(output, tuple):
            output = output[0]

        loss = F.cross_entropy(output, y)
        loss.backward()

        perturbation = alpha * x_adv.grad.sign()
        x_adv = (x_adv + perturbation).detach()

        # Project back to ε-ball
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = x_orig + delta

    return x_adv.detach()


def evaluate_adversarial_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    num_classes: int = 2,
    epsilons: List[float] = None,
    device: torch.device = DEVICE,
    max_samples: int = 2000,
) -> Dict:
    """
    Evaluate model under FGSM and PGD attacks at various ε levels.
    """
    from models.base import BaseSklearnModel
    if isinstance(model, BaseSklearnModel):
        print(f"  [ADV] Skipping {model_name} (sklearn model, not differentiable)")
        return {}

    if epsilons is None:
        epsilons = ADVERSARIAL_EPSILONS

    # Subsample
    n = min(max_samples, len(X_test))
    X_sub = X_test[:n]
    y_sub = y_test[:n]

    x_tensor = torch.FloatTensor(X_sub).to(device)
    y_tensor = torch.LongTensor(y_sub).to(device)

    model = model.to(device)
    model.eval()

    # Determine average parameter for f1_score
    # In a research benchmark, 'macro' is generally safer and more informative for multiclass,
    # while 'binary' is only for [0, 1] labels.
    unique_y = np.unique(y_sub)
    if num_classes == 2 and set(unique_y).issubset({0, 1}):
        avg = "binary"
    else:
        avg = "macro"
    
    # Clean accuracy
    with torch.no_grad():
        clean_out = model(x_tensor)
        if isinstance(clean_out, tuple):
            clean_out = clean_out[0]
        
        if hasattr(model, 'model_type') and model.model_type == "unsupervised":
            scores = model.get_anomaly_scores(x_tensor)
            threshold = np.percentile(scores, 95)  # Top 5% as anomalies
            clean_preds = (scores > threshold).astype(int)
        else:
            clean_preds = clean_out.argmax(dim=-1).cpu().numpy()
    
    # Final safety check for average='binary' - both true and pred must be binary
    if avg == "binary":
        unique_preds = np.unique(clean_preds)
        if not set(unique_preds).issubset({0, 1}):
            avg = "macro"
            
    clean_f1 = f1_score(y_sub, clean_preds, average=avg, zero_division=0)


    results = {
        "model_name": model_name,
        "clean_f1": float(clean_f1),
        "fgsm": {},
        "pgd": {},
    }

    print(f"  [ADV] {model_name} — Clean F1: {clean_f1:.4f}")

    for eps in epsilons:
        # FGSM
        x_fgsm = fgsm_attack(model, x_tensor, y_tensor, eps, device)
        with torch.no_grad():
            fgsm_out = model(x_fgsm)
            if isinstance(fgsm_out, tuple):
                fgsm_out = fgsm_out[0]
            if hasattr(model, 'model_type') and model.model_type == "unsupervised":
                fgsm_scores = model.get_anomaly_scores(x_fgsm)
                fgsm_preds = (fgsm_scores > threshold).astype(int)
            else:
                fgsm_preds = fgsm_out.argmax(dim=-1).cpu().numpy()
        fgsm_f1 = f1_score(y_sub, fgsm_preds, average=avg, zero_division=0)
        results["fgsm"][str(eps)] = {
            "f1": float(fgsm_f1),
            "f1_drop": float((clean_f1 - fgsm_f1) / max(clean_f1, 1e-8) * 100),
        }

        # PGD
        x_pgd = pgd_attack(model, x_tensor, y_tensor, eps, device=device)
        with torch.no_grad():
            pgd_out = model(x_pgd)
            if isinstance(pgd_out, tuple):
                pgd_out = pgd_out[0]
            if hasattr(model, 'model_type') and model.model_type == "unsupervised":
                pgd_scores = model.get_anomaly_scores(x_pgd)
                pgd_preds = (pgd_scores > threshold).astype(int)
            else:
                pgd_preds = pgd_out.argmax(dim=-1).cpu().numpy()
        pgd_f1 = f1_score(y_sub, pgd_preds, average=avg, zero_division=0)
        results["pgd"][str(eps)] = {
            "f1": float(pgd_f1),
            "f1_drop": float((clean_f1 - pgd_f1) / max(clean_f1, 1e-8) * 100),
        }

        print(f"    eps={eps:.2f} | FGSM F1: {fgsm_f1:.4f} ({results['fgsm'][str(eps)]['f1_drop']:.1f}% drop) | "
              f"PGD F1: {pgd_f1:.4f} ({results['pgd'][str(eps)]['f1_drop']:.1f}% drop)")

    return results


def plot_robustness_curves(all_robustness: Dict[str, Dict],
                           save_name: str = "robustness_curves"):
    """Plot F1 vs ε for all models under FGSM and PGD."""
    COLORS_LOCAL = [
        "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (name, results) in enumerate(all_robustness.items()):
        if not results or "fgsm" not in results:
            continue
        color = COLORS_LOCAL[i % len(COLORS_LOCAL)]

        # FGSM
        eps_vals = sorted([float(e) for e in results["fgsm"].keys()])
        fgsm_f1s = [results["fgsm"][str(e)]["f1"] for e in eps_vals]
        ax1.plot([0] + eps_vals, [results["clean_f1"]] + fgsm_f1s,
                marker="o", label=name, color=color, linewidth=1.5, markersize=5)

        # PGD
        pgd_f1s = [results["pgd"][str(e)]["f1"] for e in eps_vals]
        ax2.plot([0] + eps_vals, [results["clean_f1"]] + pgd_f1s,
                marker="s", label=name, color=color, linewidth=1.5, markersize=5)

    ax1.set_xlabel("Perturbation (eps)")
    ax1.set_ylabel("F1-Score")
    ax1.set_title("FGSM Attack Robustness")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_ylim(0, 1.05)

    ax2.set_xlabel("Perturbation (eps)")
    ax2.set_ylabel("F1-Score")
    ax2.set_title("PGD Attack Robustness")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{save_name}.png", dpi=300)
    plt.savefig(PLOTS_DIR / f"{save_name}.pdf")
    plt.close()
    print(f"  Saved: {save_name}.png/pdf")
