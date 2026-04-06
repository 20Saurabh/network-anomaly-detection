"""
Unified data preprocessing for all supported datasets.
Handles: CICIoT2023, Edge-IIoTset, UNSW-NB15, CIC-IDS2017
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_CONFIGS, PROCESSED_DIR, set_seed


def load_and_preprocess(
    dataset_name: str,
    binary: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Load, clean, and split a dataset. Returns dict with:
      X_train, X_val, X_test, y_train, y_val, y_test,
      feature_names, label_encoder, scaler, num_classes
    """
    set_seed(seed)
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*60}")
    print(f"  Loading dataset: {cfg.name}")
    print(f"{'='*60}")

    # ── Load ───────────────────────────────────────────────────────
    if dataset_name == "ciciot2023":
        df = _load_ciciot2023(cfg)
    elif dataset_name == "edge_iiot":
        df = _load_edge_iiot(cfg)
    elif dataset_name == "unsw_nb15":
        df = _load_unsw_nb15(cfg)
    elif dataset_name == "cicids2017":
        df = _load_cicids2017(cfg)
    elif dataset_name == "custom":
        df = _load_custom(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Raw shape: {df.shape}")

    # ── Subsample if needed (CPU mode) ────────────────────────────
    effective_max = max_samples or cfg.max_samples
    if effective_max and len(df) > effective_max:
        df = df.sample(n=effective_max, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to: {len(df)} rows")

    # ── Separate features and labels ──────────────────────────────
    label_col = cfg.label_col
    if label_col not in df.columns:
        # Try stripped version
        label_col = label_col.strip()
        if label_col not in df.columns:
            raise KeyError(f"Label column '{cfg.label_col}' not found. Columns: {list(df.columns[:10])}...")

    y_raw = df[label_col].copy()
    X = df.drop(columns=[label_col], errors="ignore")

    # Drop attack_cat if separate from label
    if cfg.attack_cat_col and cfg.attack_cat_col != cfg.label_col:
        if cfg.attack_cat_col in X.columns:
            X = X.drop(columns=[cfg.attack_cat_col])

    # Drop specified columns
    X = X.drop(columns=[c for c in cfg.drop_cols if c in X.columns], errors="ignore")

    # ── Encode labels ────────────────────────────────────────────
    le = LabelEncoder()
    if binary:
        # Convert to binary: 0 = benign, 1 = attack
        y_binary = _to_binary(y_raw, dataset_name)
        y = y_binary.values
        num_classes = 2
    else:
        y = le.fit_transform(y_raw.astype(str))
        num_classes = len(le.classes_)

    print(f"  Classes: {num_classes} ({'binary' if binary else 'multiclass'})")

    # ── Handle non-numeric columns ────────────────────────────────
    # Encode categoricals
    cat_cols = [c for c in cfg.categorical_cols if c in X.columns]
    label_encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        label_encoders[col] = enc

    # Drop remaining object columns
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        print(f"  Dropping non-numeric columns: {obj_cols[:5]}...")
        X = X.drop(columns=obj_cols)

    # ── Clean numeric data ────────────────────────────────────────
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Drop constant columns
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print(f"  Dropped {len(constant_cols)} constant columns")

    feature_names = X.columns.tolist()
    X = X.values.astype(np.float32)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Train / Val / Test split ──────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(cfg.val_ratio + cfg.test_ratio),
        random_state=seed, stratify=y
    )
    relative_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=seed, stratify=y_temp
    )

    # ── Scale ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": le,
        "scaler": scaler,
        "num_classes": num_classes,
        "dataset_name": cfg.name,
    }


# ── Dataset-specific loaders ─────────────────────────────────────────────────

def _load_ciciot2023(cfg) -> pd.DataFrame:
    """Load CICIoT2023 — may be split across multiple CSVs."""
    data_path = Path(cfg.train_file)
    if data_path.is_dir():
        csvs = sorted(data_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files in {data_path}. Download CICIoT2023 from "
                f"https://www.unb.ca/cic/datasets/iotdataset-2023.html"
            )
        dfs = []
        for f in csvs:
            try:
                dfs.append(pd.read_csv(f, low_memory=False))
            except Exception as e:
                print(f"  Warning: skipping {f.name}: {e}")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(str(data_path), low_memory=False)
    return df


def _load_edge_iiot(cfg) -> pd.DataFrame:
    """Load Edge-IIoTset."""
    path = Path(cfg.train_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Edge-IIoTset not found at {path}. Download from "
            f"https://ieee-dataport.org (search 'Edge-IIoTset')"
        )
    return pd.read_csv(str(path), low_memory=False)


def _load_unsw_nb15(cfg) -> pd.DataFrame:
    """Load UNSW-NB15 with optional separate test file."""
    train_path = Path(cfg.train_file)
    if not train_path.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 not found at {train_path}. Download from Kaggle: "
            f"https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
        )
    df_train = pd.read_csv(str(train_path), low_memory=False)
    if cfg.test_file and Path(cfg.test_file).exists():
        df_test = pd.read_csv(cfg.test_file, low_memory=False)
        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df = df_train
    return df


def _load_cicids2017(cfg) -> pd.DataFrame:
    """Load CIC-IDS2017 — typically multiple CSVs per day."""
    data_path = Path(cfg.train_file)
    if data_path.is_dir():
        csvs = sorted(data_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files in {data_path}. Download CIC-IDS2017."
            )
        dfs = []
        for f in csvs:
            try:
                dfs.append(pd.read_csv(f, low_memory=False, encoding="utf-8"))
            except Exception:
                try:
                    dfs.append(pd.read_csv(f, low_memory=False, encoding="latin-1"))
                except Exception as e:
                    print(f"  Warning: skipping {f.name}: {e}")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(str(data_path), low_memory=False)
    # Strip column whitespace
    df.columns = df.columns.str.strip()
    if "Label" in df.columns:
        cfg.label_col = "Label"
    return df


def _load_custom(cfg) -> pd.DataFrame:
    """Load custom uploaded dataset."""
    path = Path(cfg.train_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Custom dataset not found at {path}. Please upload a CSV file via the Streamlit app."
        )
    return pd.read_csv(str(path), low_memory=False)


def _to_binary(y: pd.Series, dataset_name: str) -> pd.Series:
    """Convert multi-class labels to binary (0=benign, 1=attack)."""
    # Handle different label formats
    if y.dtype in ['int64', 'int32', 'float64', 'float32']:
        # For numeric labels, treat 0 as benign, everything else as attack
        result = (y != 0).astype(int)
    else:
        # For string labels, convert to string and check against benign set
        y_str = y.astype(str).str.strip().str.lower()

        benign_labels = {
            "benign", "normal", "0", "0.0",
            "nan", "", "false", "f",
        }

        result = y_str.apply(lambda x: 0 if x in benign_labels else 1)

    return result


def generate_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 46,
    anomaly_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Generate a synthetic dataset for testing the pipeline
    when real datasets are not yet downloaded.
    """
    set_seed(seed)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal: centered around 0 with unit variance
    X_normal = np.random.randn(n_normal, n_features).astype(np.float32)
    # Anomalies: shifted distribution
    X_anomaly = np.random.randn(n_anomalies, n_features).astype(np.float32) * 2 + 3

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomalies)

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    feature_names = [f"feature_{i}" for i in range(n_features)]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[SYNTHETIC] Generated {n_samples} samples, {n_features} features, "
          f"{anomaly_ratio*100:.0f}% anomaly rate")

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": LabelEncoder().fit(["normal", "attack"]),
        "scaler": scaler,
        "num_classes": 2,
        "dataset_name": "Synthetic",
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    data = generate_synthetic_dataset(n_samples=5000)
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
