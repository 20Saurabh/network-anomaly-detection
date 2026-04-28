<!-- <img width="1024" height="559" alt="eeea979d-b2f3-40fd-8eee-1b706dfad04e" src="https://github.com/user-attachments/assets/00a3b178-f9d6-4e9f-b1c4-c7e9297edc42" /> -->
# NetGuard: A Unified Benchmark for Next-Generation Network Anomaly Detection

> **A comprehensive research framework** for evaluating state-of-the-art deep learning, graph neural network, and self-supervised architectures for network intrusion detection — with adversarial robustness testing, explainability, and real-time adaptive inference.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [System Architecture](#system-architecture)
- [Datasets](#datasets)
- [Model Architectures](#model-architectures)
- [Evaluation Methodology](#evaluation-methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Experiment Reproduction](#full-experiment-reproduction)
- [Results](#results)
- [Adversarial Robustness](#adversarial-robustness)
- [Explainability](#explainability)
- [Real-Time Streaming Pipeline](#real-time-streaming-pipeline)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Authors](#authors)

---

## Abstract

Network Intrusion Detection Systems (NIDS) are critical for protecting modern networks against increasingly sophisticated cyber threats. While deep learning approaches have shown promise, existing research suffers from: **(1)** reliance on outdated datasets (KDD'99, NSL-KDD) that lack modern IoT attack vectors; **(2)** evaluation limited to a single architecture without cross-model comparison; **(3)** absence of adversarial robustness analysis; and **(4)** no consideration of real-time deployment constraints including concept drift.

**NetGuard** addresses these gaps with a unified benchmarking framework that:

- Evaluates **10 architectures** (including a novel GNN-Transformer hybrid) across modern IoT/IIoT datasets
- Introduces **self-supervised contrastive learning** for label-scarce network anomaly detection
- Provides **adversarial robustness analysis** (FGSM, PGD) with feature-constrained attacks
- Implements **streaming inference** with concept drift detection (ADWIN, Page-Hinkley)
- Includes **cross-dataset generalization** testing to validate model transferability
- Offers **explainability** via SHAP analysis for security analyst trust

---

## Key Contributions

| # | Contribution | Section |
|:--|:---|:---|
| **C1** | A unified benchmark comparing 10 architectures (traditional ML → GNN → Transformer → Self-Supervised) under standardized evaluation | [Models](#model-architectures) |
| **C2** | **Novel GNN-Transformer Hybrid** architecture combining graph topology awareness with temporal self-attention for superior detection of distributed attacks | [GNN-Transformer](#7-gnn-transformer-hybrid-novel) |
| **C3** | First systematic evaluation of **contrastive self-supervised learning** for network anomaly detection, demonstrating competitive performance at 10% label availability | [SSL](#9-contrastive-self-supervised-learning) |
| **C4** | Comprehensive **adversarial robustness** analysis with domain-specific feature constraints, revealing critical vulnerabilities in existing models | [Robustness](#adversarial-robustness) |
| **C5** | **Streaming inference pipeline** with concept drift detection and incremental retraining for real-world deployment | [Streaming](#real-time-streaming-pipeline) |
| **C6** | **Cross-dataset generalization** evaluation — beyond single-dataset accuracy | [Evaluation](#evaluation-methodology) |

---

## System Architecture

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/94301a6a-429e-44e2-a41c-b70949b15b85" />

## Datasets

### Dataset Comparison

| Property | CICIoT2023 | Edge-IIoTset | UNSW-NB15 | CIC-IDS2017 |
|:---|:---:|:---:|:---:|:---:|
| **Year** | 2023 | 2022 | 2015 | 2017 |
| **Samples** | 47M+ | ~157K | ~2.5M | ~2.8M |
| **Attack Types** | 33 | 14 | 9 | 14 |
| **IoT Coverage** | ✅ | ✅ | ❌ | ❌ |
| **Protocols** | MQTT, CoAP, DNS | MQTT, Modbus | TCP/UDP | HTTP/SSH/FTP |
| **Role** | Primary benchmark | Secondary benchmark | Legacy baseline | Legacy baseline |

### Multi-Dataset Strategy

- **Training**: CICIoT2023 or Edge-IIoTset (70% split)
- **Validation**: 15% stratified split
- **Testing**: 15% stratified split
- **Cross-Dataset**: Train on A → Test on B (generalization proof)

### Download Instructions

1. **CICIoT2023**: [UNB Official](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — Place CSVs in `backend/data/raw/ciciot2023/`
2. **Edge-IIoTset**: [IEEE Dataport](https://ieee-dataport.org/) or [Kaggle](https://www.kaggle.com/) — Place `DNN-EdgeIIoT-dataset.csv` in `backend/data/raw/edge_iiot/`
3. **UNSW-NB15**: [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) — Place CSVs in `backend/data/raw/unsw_nb15/`
4. **CIC-IDS2017**: [UNB Official](https://www.unb.ca/cic/datasets/ids-2017.html) — Place CSVs in `backend/data/raw/cicids2017/`

---

## Model Architectures

### Summary Table

| # | Model | Type | Approach | Key Innovation |
|:--|:---|:---|:---|:---|
| 1 | **Vanilla AE** | Unsupervised | Reconstruction | Baseline with BatchNorm + Dropout regularization |
| 2 | **β-VAE** | Unsupervised | Probabilistic | KL annealing prevents posterior collapse |
| 3 | **1D-CNN** | Supervised | Classification | Residual connections for deeper feature extraction |
| 4 | **BiLSTM+Attention** | Supervised | Classification | Multi-head attention pooling over temporal features |
| 5 | **CNN-LSTM Hybrid** | Supervised | Classification | CNN spatial features → LSTM temporal modeling |
| 6 | **FT-Transformer** | Supervised | Classification | Feature Tokenizer — SOTA for tabular data (ICLR 2021) |
| 7 | **E-GraphSAGE** | Supervised | Graph Classification | Edge-centric GNN for network topology modeling |
| 8 | **GNN-Transformer** | Supervised | Hybrid | **Novel**: Graph topology + temporal self-attention |
| 9 | **Contrastive SSL** | Semi-supervised | Representation | NT-Xent contrastive pre-training + linear probe |
| 10 | **Isolation Forest** | Unsupervised | Tree-based | Traditional ML baseline |
| 11 | **Ensemble (Stacking)** | Supervised | Meta-learning | Learned combination of all base models |

### Novel Architecture: GNN-Transformer Hybrid

```
Phase 1 — Graph Encoding:
  E-GraphSAGE → Per-node latent embeddings (topology-aware)

Phase 2 — Sequence Formation:
  Node embeddings → Transformer Encoder with [CLS] token

Phase 3 — Classification:
  [CLS] → LayerNorm → MLP → num_classes
```

This architecture captures both **topological dependencies** (which IPs communicate with which) and **temporal patterns** (how communication evolves), addressing a key limitation of standalone GNNs (no temporal context) and standalone Transformers (no graph structure awareness).

---

## Evaluation Methodology

### Metrics (15+)

| Tier | Metrics | Purpose |
|:---|:---|:---|
| **Standard** | Accuracy, Precision, Recall, F1 (macro/weighted), AUC-ROC, AUC-PR | Classification performance |
| **Operational** | Detection Rate, False Alarm Rate, Latency (ms), Throughput (samples/sec) | Deployment readiness |
| **Robustness** | F1 under FGSM, F1 under PGD, F1 degradation % | Adversarial resilience |

### Statistical Rigor

- **5 independent runs** per model (seeds: 42, 123, 456, 789, 1024)
- **Wilcoxon signed-rank test** for pairwise model comparison
- **Friedman test** for all-model simultaneous comparison
- **95% bootstrap confidence intervals** for each metric

### Cross-Dataset Generalization

| Train ↓ \ Test → | CICIoT2023 | Edge-IIoTset | UNSW-NB15 |
|:---|:---:|:---:|:---:|
| **CICIoT2023** | In-distribution | Cross-dataset | Cross-dataset |
| **Edge-IIoTset** | Cross-dataset | In-distribution | Cross-dataset |

---

## Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ (CPU or CUDA)

### Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional: PyTorch Geometric (for GNN models)
pip install torch-geometric
```

### GPU Support (Optional)

The system **auto-detects CUDA**. No code changes needed:
- CUDA available → trains on GPU
- No CUDA → trains on CPU (smaller batch sizes, stratified subsampling)

---

## Quick Start

### 1. Test with Synthetic Data (No downloads needed)

```bash
cd backend
python run_experiments.py --dataset synthetic --epochs 10 --runs 1
```

This generates synthetic data, trains all models, evaluates them, and produces comparison plots in `results/`.

### 2. Run with Real Dataset

```bash
# After downloading UNSW-NB15 to backend/data/raw/unsw_nb15/
python run_experiments.py --dataset unsw_nb15 --epochs 30 --runs 1

# Specific models only
python run_experiments.py --dataset unsw_nb15 --models cnn1d ft_transformer vanilla_ae

# Full statistical analysis (5 runs)
python run_experiments.py --dataset unsw_nb15 --runs 5
```

### 3. Quick Run (Skip Expensive Steps)

```bash
python run_experiments.py --dataset synthetic --epochs 5 \
    --skip-adversarial --skip-streaming --skip-explainability
```

---

## Full Experiment Reproduction

To reproduce the complete results reported in the paper:

```bash
# Step 1: Download datasets (see instructions above)

# Step 2: Primary benchmark — CICIoT2023
python run_experiments.py --dataset ciciot2023 --epochs 50 --runs 5

# Step 3: Secondary benchmark — Edge-IIoTset
python run_experiments.py --dataset edge_iiot --epochs 50 --runs 5

# Step 4: Cross-dataset evaluation
python run_experiments.py --dataset unsw_nb15 --epochs 50 --runs 5
python run_experiments.py --dataset cicids2017 --epochs 50 --runs 5

# Results will be in backend/results/
```

---

## Results

After running experiments, results are saved in:

```
backend/results/
├── models/          # Saved model checkpoints (.pt)
├── plots/           # Publication-quality figures (.png + .pdf)
│   ├── training_curves_*.pdf
│   ├── roc_curves_*.pdf
│   ├── pr_curves_*.pdf
│   ├── confusion_matrices_*.pdf
│   ├── metric_comparison_*.pdf
│   ├── latency_f1_*.pdf
│   ├── box_plots_*.pdf
│   └── robustness_curves.pdf
├── metrics/         # JSON metric files
│   ├── all_results.json
│   ├── statistical_tests.json
│   ├── adversarial_robustness.json
│   └── streaming_results.json
└── reports/         # Generated markdown reports
```

---

## Adversarial Robustness

We evaluate model robustness under two white-box attacks:

- **FGSM** (Fast Gradient Sign Method): Single-step perturbation
- **PGD** (Projected Gradient Descent): 20-step iterative attack

Perturbation budgets: ε ∈ {0.01, 0.05, 0.1, 0.2}

```bash
# Run adversarial evaluation
python run_experiments.py --dataset synthetic --epochs 10 --runs 1
```

Output: `results/plots/robustness_curves.pdf` — F1 vs ε for each model.

---

## Explainability

### SHAP Analysis
- Global feature importance (aggregated |SHAP values|)
- Local explanations for individual flagged anomalies

### Permutation Feature Importance
- Model-agnostic F1 degradation when features are shuffled
- Output: `results/plots/perm_importance_*.pdf`

---

## Real-Time Streaming Pipeline

The streaming module simulates real-time deployment:

1. **Micro-batch inference**: Processes flows in configurable windows
2. **Concept drift detection**: ADWIN + Page-Hinkley test
3. **Latency/throughput measurement**: Per-batch timing

```
Input Stream → Feature Extract → Model Inference → Alert System
                                       ↓
                              Drift Detector (ADWIN)
                                       ↓
                            Incremental Retrain Trigger
```

---

## Project Structure

```
backend/
├── config.py                           # Central configuration (CPU/GPU auto-switch)
├── requirements.txt                    # Python dependencies
├── run_experiments.py                  # Master experiment orchestrator
│
├── data/
│   ├── preprocessing.py                # Multi-dataset preprocessing pipeline
│   ├── dataloader.py                   # PyTorch DataLoader + weighted sampling
│   ├── graph_builder.py                # k-NN and IP-based graph construction
│   └── raw/                            # Place downloaded datasets here
│
├── models/
│   ├── __init__.py                     # Model registry
│   ├── base.py                         # Abstract base classes
│   ├── vanilla_autoencoder.py          # Vanilla AE (BatchNorm + Dropout)
│   ├── variational_autoencoder.py      # β-VAE with KL annealing
│   ├── cnn1d.py                        # 1D-CNN + residual connections
│   ├── bilstm_attention.py             # BiLSTM + multi-head attention
│   ├── cnn_lstm_hybrid.py              # CNN-LSTM hybrid
│   ├── ft_transformer.py              # Feature Tokenizer Transformer
│   ├── e_graphsage.py                  # E-GraphSAGE (PyG, optional)
│   ├── gnn_transformer_hybrid.py       # GNN-Transformer (novel, PyG optional)
│   ├── contrastive_ssl.py              # Contrastive pre-training + probe
│   ├── isolation_forest_wrapper.py     # Isolation Forest (sklearn)
│   └── ensemble_stacking.py            # Stacking meta-learner
│
├── training/
│   ├── __init__.py                     # Unified training loop
│   ├── evaluate.py                     # Metrics engine (15+ metrics)
│   └── statistical_tests.py           # Wilcoxon, Friedman, bootstrap CI
│
├── visualization/
│   └── plots.py                        # Publication-quality plot generation
│
├── explainability/
│   └── shap_analysis.py                # SHAP + permutation importance
│
├── robustness/
│   └── adversarial_attacks.py          # FGSM, PGD attacks + robustness curves
│
├── runtime/
│   └── streaming.py                    # Streaming inference + drift detection
│
└── results/                            # Auto-generated experiment outputs
```

---

## Hardware Requirements

| Configuration | Notes |
|:---|:---|
| **Minimum (CPU)** | Intel i5, 8GB RAM — synthetic data, reduced epochs |
| **Recommended (CPU)** | Intel i7/Ryzen 7, 16GB RAM — real datasets with subsampling |
| **Optimal (GPU)** | NVIDIA GPU (8GB+ VRAM), 32GB RAM — full-scale experiments |

The system **automatically detects and uses GPU** when available. No code changes required.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{netguard2026,
  title     = {NetGuard: A Unified Benchmark for Next-Generation Network
               Anomaly Detection with Graph-Temporal Hybrid Architectures},
  author    = {Singh, Anurag and Kumar, Saurabh and Jha, Suraj Kumar},
  booktitle = {Proceedings of the IEEE International Conference on
               Communications (ICC)},
  year      = {2026},
  note      = {Under review}
}
```

---

## Authors

- **Anurag Singh** — Deep learning architecture design, GNN implementation
- **Saurabh Kumar** — System architecture, evaluation framework, streaming pipeline
- **Suraj Kumar Jha** — Data preprocessing, adversarial robustness, explainability

**Guide:** Prof. J. B. Jawale
**Department of Electronics and Telecommunication Engineering**

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Canadian Institute for Cybersecurity (CIC) for CICIoT2023 and CIC-IDS2017 datasets
- University of New South Wales for UNSW-NB15 dataset
- PyTorch team for the deep learning framework
- PyTorch Geometric team for GNN infrastructure
