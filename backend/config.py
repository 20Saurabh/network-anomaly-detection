"""
Central Configuration for Network Anomaly Detection System v2.0
Auto-detects CUDA GPU; falls back to CPU transparently.
"""
import os
import torch
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
# Default paths (will be overridden by update_results_dir during a tracked run)
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def update_results_dir(run_name: str):
    """Dynamically update output directories for tracking history."""
    global MODELS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR
    run_dir = RESULTS_DIR / "runs" / run_name
    MODELS_DIR = run_dir / "models"
    PLOTS_DIR = run_dir / "plots"
    METRICS_DIR = run_dir / "metrics"
    REPORTS_DIR = run_dir / "reports"
    
    for d in [run_dir, MODELS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─── Device ──────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Auto-detect GPU. Switch seamlessly between CPU and CUDA."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[CONFIG] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[CONFIG] Using CPU (set CUDA_VISIBLE_DEVICES or install CUDA for GPU)")
    return dev

DEVICE = get_device()

# ─── Reproducibility ────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]  # 5 runs for statistical rigor

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─── Dataset Configuration ──────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    name: str
    train_file: str
    test_file: Optional[str] = None
    label_col: str = "label"
    attack_cat_col: Optional[str] = None
    categorical_cols: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)
    max_samples: Optional[int] = None  # Subsample for CPU training
    val_ratio: float = 0.15
    test_ratio: float = 0.15

DATASET_CONFIGS = {
    "ciciot2023": DatasetConfig(
        name="CICIoT2023",
        train_file=str(DATA_DIR / "ciciot2023"),
        label_col="label",
        attack_cat_col="label",  # multi-class labels in same column
        categorical_cols=[],
        drop_cols=[],
        max_samples=500_000 if not torch.cuda.is_available() else None,
    ),
    "edge_iiot": DatasetConfig(
        name="Edge-IIoTset",
        train_file=str(DATA_DIR / "edge_iiot" / "DNN-EdgeIIoT-dataset.csv"),
        label_col="Attack_type",
        attack_cat_col="Attack_type",
        categorical_cols=[],
        drop_cols=[],
        max_samples=None,
    ),
    "unsw_nb15": DatasetConfig(
        name="UNSW-NB15",
        train_file=str(DATA_DIR / "unsw_nb15" / "UNSW_NB15_training-set.csv"),
        test_file=str(DATA_DIR / "unsw_nb15" / "UNSW_NB15_testing-set.csv"),
        label_col="label",
        attack_cat_col="attack_cat",
        categorical_cols=["proto", "service", "state"],
        drop_cols=["id"],
        max_samples=None,
    ),
    "cicids2017": DatasetConfig(
        name="CIC-IDS2017",
        train_file=str(DATA_DIR / "cicids2017"),
        label_col=" Label",
        attack_cat_col=" Label",
        categorical_cols=[],
        drop_cols=[],
        max_samples=500_000 if not torch.cuda.is_available() else None,
    ),
    "custom": DatasetConfig(
        name="Custom Dataset",
        train_file=str(DATA_DIR / "custom" / "data.csv"),
        label_col="label",
        attack_cat_col="label",
        categorical_cols=[],
        drop_cols=[],
        max_samples=None,
    ),
}

# ─── Model Hyperparameters ──────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10        # Early stopping patience
    lr_patience: int = 5      # ReduceLROnPlateau patience
    lr_factor: float = 0.5    # LR reduction factor
    grad_clip: float = 1.0    # Gradient clipping max norm
    num_workers: int = 0      # DataLoader workers (0 for Windows compat)
    pin_memory: bool = torch.cuda.is_available()

TRAINING_CONFIG = TrainingConfig()

# Adjust for CPU to keep training feasible
if not torch.cuda.is_available():
    TRAINING_CONFIG.batch_size = 128
    TRAINING_CONFIG.epochs = 30

@dataclass
class ModelConfigs:
    """Architecture-specific hyperparameters."""
    # Autoencoders
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    ae_dropout: float = 0.2
    vae_beta: float = 1.0
    vae_beta_warmup_epochs: int = 10

    # CNN
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 3

    # LSTM
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    attention_heads: int = 4

    # FT-Transformer
    ft_d_model: int = 64
    ft_n_heads: int = 4
    ft_n_layers: int = 3
    ft_ffn_factor: float = 4.0 / 3.0
    ft_dropout: float = 0.1

    # GNN (E-GraphSAGE)
    gnn_hidden: int = 128
    gnn_out: int = 64
    gnn_layers: int = 2
    gnn_dropout: float = 0.3

    # Contrastive SSL
    ssl_projection_dim: int = 64
    ssl_temperature: float = 0.5
    ssl_mask_ratio: float = 0.15
    ssl_noise_std: float = 0.1
    ssl_epochs: int = 100

    # Isolation Forest
    if_n_estimators: int = 200
    if_contamination: str = "auto"

MODEL_CONFIGS = ModelConfigs()

# ─── Adversarial Robustness ────────────────────────────────────────────────
ADVERSARIAL_EPSILONS = [0.01, 0.05, 0.1, 0.2]
PGD_STEPS = 20
PGD_ALPHA = 0.01

# ─── Streaming / Runtime ───────────────────────────────────────────────────
STREAM_BATCH_SIZE = 512
STREAM_WINDOW_SEC = 5.0
DRIFT_DELTA = 0.002        # ADWIN confidence parameter
DRIFT_THRESHOLD = 50       # Page-Hinkley threshold
INCREMENTAL_LR = 1e-5
INCREMENTAL_EPOCHS = 5
EWC_LAMBDA = 5000          # EWC regularization strength

# ─── All Registered Models ──────────────────────────────────────────────────
ALL_MODEL_NAMES = [
    "vanilla_ae",
    "vae",
    "cnn1d",
    "bilstm_attention",
    "cnn_lstm",
    "ft_transformer",
    "e_graphsage",
    "gnn_transformer",
    "contrastive_ssl",
    "isolation_forest",
    "ensemble_stacking",
]

# Tabular-only models (no graph construction needed)
TABULAR_MODELS = [
    "vanilla_ae", "vae", "cnn1d", "bilstm_attention",
    "cnn_lstm", "ft_transformer", "contrastive_ssl",
    "isolation_forest", "ensemble_stacking",
]

# Unsupervised models (use reconstruction error for anomaly scoring)
UNSUPERVISED_MODELS = ["vanilla_ae", "vae", "contrastive_ssl", "isolation_forest"]

print(f"[CONFIG] Project root: {PROJECT_ROOT}")
print(f"[CONFIG] Results dir:  {RESULTS_DIR}")
