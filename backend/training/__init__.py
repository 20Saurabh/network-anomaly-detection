"""
Unified training loop for all models.
Features: early stopping, LR scheduling, gradient clipping,
class-weighted loss, model checkpointing, VAE/SSL special handling.
"""
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEVICE, TRAINING_CONFIG, MODELS_DIR, MODEL_CONFIGS


def train_model(
    model,
    loaders: Dict,
    model_name: str,
    data: Dict,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    device: torch.device = DEVICE,
    save_best: bool = True,
) -> Dict:
    """
    Train a PyTorch model with full training infrastructure.
    Returns training history dict.
    """
    from models.base import BaseSklearnModel

    # Handle sklearn models
    if isinstance(model, BaseSklearnModel):
        return _train_sklearn(model, data, model_name)

    epochs = epochs or TRAINING_CONFIG.epochs
    lr = lr or TRAINING_CONFIG.learning_rate

    model = model.to(device)
    is_vae = model_name == "vae"
    is_ssl = model_name == "contrastive_ssl"

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=TRAINING_CONFIG.weight_decay
    )

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=TRAINING_CONFIG.lr_patience,
        factor=TRAINING_CONFIG.lr_factor, min_lr=1e-7
    )

    # Loss function
    if is_vae:
        loss_fn = None  # VAE has custom loss
    elif is_ssl and not model._pretrained:
        loss_fn = None  # SSL has custom pretrain loss
    elif model.model_type == "unsupervised":
        loss_fn = nn.MSELoss()
    else:
        from data.dataloader import compute_class_weights
        weights = compute_class_weights(data["y_train"])
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    # History
    history = defaultdict(list)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'-'*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
    print(f"{'-'*60}")

    # ── SSL Pre-training Phase ────────────────────────────────────
    if is_ssl and not model._pretrained:
        print("  [Phase 1] Contrastive pre-training...")
        model = _pretrain_ssl(model, loaders["train"], optimizer, device, epochs=MODEL_CONFIGS.ssl_epochs)
        model.freeze_backbone()
        # Reset optimizer for linear probe
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * 10, weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        from data.dataloader import compute_class_weights
        weights = compute_class_weights(data["y_train"])
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        print("  [Phase 2] Training linear probe...")

    # ── Main Training Loop ────────────────────────────────────────
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train epoch
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            optimizer.zero_grad()

            if model.model_type == "unsupervised" and not is_ssl:
                if len(batch) == 3:
                    x, target, _ = batch
                else:
                    x, target = batch
                x, target = x.to(device), target.to(device)

                if is_vae:
                    model.set_beta(epoch, MODEL_CONFIGS.vae_beta_warmup_epochs)
                    recon, mu, logvar = model(x)
                    loss, _, _ = model.loss_function(recon, target, mu, logvar)
                else:
                    output = model(x)
                    loss = loss_fn(output, target)
            else:
                x, y = batch[0].to(device), batch[-1].to(device)
                output = model(x)
                loss = loss_fn(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING_CONFIG.grad_clip
            )
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        val_loss = _validate(model, loaders["val"], loss_fn, device, is_vae)

        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Progress
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_best:
                save_path = MODELS_DIR / f"{model_name}_best.pt"
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_CONFIG.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    history["training_time"] = elapsed
    history["best_val_loss"] = best_val_loss

    # Load best model
    if save_best:
        best_path = MODELS_DIR / f"{model_name}_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    print(f"  Done in {elapsed:.1f}s | Best val loss: {best_val_loss:.6f}")

    return dict(history)


def _validate(model, val_loader, loss_fn, device, is_vae=False):
    """Run validation and return average loss."""
    model.eval()
    val_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if model.model_type == "unsupervised" and not hasattr(model, '_pretrained'):
                if len(batch) == 3:
                    x, target, _ = batch
                else:
                    x, target = batch
                x, target = x.to(device), target.to(device)
                if is_vae:
                    recon, mu, logvar = model(x)
                    loss, _, _ = model.loss_function(recon, target, mu, logvar)
                else:
                    output = model(x)
                    loss = loss_fn(output, target)
            else:
                x, y = batch[0].to(device), batch[-1].to(device)
                output = model(x)
                if loss_fn is not None:
                    loss = loss_fn(output, y)
                else:
                    loss = torch.tensor(0.0)

            val_loss += loss.item()
            n_batches += 1

    return val_loss / max(n_batches, 1)


def _pretrain_ssl(model, train_loader, optimizer, device, epochs=100):
    """Contrastive pre-training loop for SSL model."""
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss = model.pretrain_step(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if epoch % max(1, epochs // 5) == 0:
            print(f"    SSL Epoch {epoch}/{epochs} | Loss: {total_loss/n:.4f}")
    return model


def _train_sklearn(model, data, model_name):
    """Train an sklearn model."""
    print(f"\n{'-'*60}")
    print(f"  Training: {model_name} (sklearn)")
    print(f"{'-'*60}")

    start = time.time()
    model.fit(data["X_train"], data.get("y_train"))
    elapsed = time.time() - start

    print(f"  Done in {elapsed:.1f}s")

    return {
        "training_time": elapsed,
        "train_loss": [],
        "val_loss": [],
    }
