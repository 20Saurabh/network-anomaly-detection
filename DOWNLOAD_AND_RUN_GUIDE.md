# NetGuard Documentation & Quick Start Guide

This guide contains everything you need to know to acquire the datasets, run the benchmarks (training, evaluating, testing, explainability), and start the frontend viewer.

---

## 1. Automated Dataset Downloads

To make downloading the massive research datasets easy, this repo includes an automated Kaggle downloader script. You must have your Kaggle API key configured (`C:\Users\<user>\.kaggle\kaggle.json`).

### Fetch All Datasets Automatically
Run the python downloader script from the `backend/` directory:
```bash
cd backend
python download_all_datasets.py
```
*This will automatically pull UNSW-NB15, Edge-IIoTset, and CIC-IDS2017 and extract them to `backend/data/raw/`.*

### Manual Download (CICIoT2023)
Because the official **CICIoT2023 dataset** is massive (30GB+), it is recommended to download it manually from the [Official UNB Data Portal](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and place the `.csv` files into:
`backend/data/raw/ciciot2023/`

---

## 2. Running the Research Benchmark Pipeline

With the datasets acquired, you can now run the complete, state-of-the-art anomaly detection benchmarks.

All experiments are heavily parameterized. The orchestrator script natively supports CPU and GPU fallback dynamically.

### A. Quick Verifications (Dry Run)
Before running the true experiments, verify your setup using synthetic streams:
```bash
cd backend
python run_experiments.py --dataset synthetic --epochs 5 --runs 1
```

### B. Standard Pipeline (Single Dataset)
To benchmark the 10 included model architectures over 5 separate random seeds (for statistical validation):
```bash
python run_experiments.py --dataset unsw_nb15 --epochs 30 --runs 5
```
*(You can swap `--dataset` to `ciciot2023` or `edge_iiot`)*

### C. Specific Architecture Targeting
If you only want to compare the novel `GNN-Transformer` against the SOTA `FT-Transformer`:
```bash
python run_experiments.py --dataset edge_iiot --models gnn_transformer ft_transformer --runs 3
```

### D. Bypassing Intensive Modules (For Quick Prototyping)
You can choose to skip the heavily intensive background computations using flags:
```bash
python run_experiments.py --dataset ciciot2023 --skip-adversarial --skip-explainability --skip-streaming
```

> [!NOTE]
> **Check your results!** All final JSON performance metrics, graphical PDF box plots, Pareto efficiency graphs, latency measures, and SHAP explainability charts will be automatically flushed into `backend/results/metrics/` and `backend/results/plots/`.

---

## 3. Running the Legacy React Interface (UI)

If you wish to visualize the system via the local React user interface, perform the following. (Note: the primary execution interface for research is via the backend CLI commands above, but the UI is intact for demonstration operations.)

**Step 1: Start the Backend Server (Optional)**
```bash
cd backend
# If the UI connects to a FastAPI layer, launch it:
# uvicorn server:app --reload
```
*(Note: NetGuard v2.0 shifted core functionality into `run_experiments.py` for decoupled benchmarking. Using the mockup web server `backend_server.py` is generally deprecated).*

**Step 2: Start the Web Dashboard**
```bash
cd frontend

# Install Node modules if it's the first time
npm install

# Run the dev server
npm run dev
# OR: npm start (depending on your react-scripts version)
```

The system will start locally, generally accessible at `http://localhost:5173` or `http://localhost:3000`.

---
*Happy Researching! See README.md for the full academic overview and architectural outlines.*
