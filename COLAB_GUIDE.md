# 🐍 Google Colab Integration Guide

## Overview
NetGuard now supports running benchmarks on **Google Colab** with free GPU/TPU access. Run full experiments without local hardware limitations.

---

## Quick Start

### 1. Open the Dashboard
```bash
cd streamlit_app
streamlit run app.py
```

### 2. Go to "Run Benchmark" Tab
- Select dataset (e.g., `unsw_nb15`)
- Choose models to benchmark
- Select **"🐍 Google Colab (GPU/TPU)"** option
- Click **"🚀 START BENCHMARK"**

### 3. Copy the Generated Code
The app will show you Python code optimized for Colab. Copy it.

### 4. Open Python Code in Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click `+ New notebook`
3. Paste the code into the first cell
4. Press `Ctrl+Enter` to execute

### 5. Authenticate & Run
- Allow Google Drive access when prompted
- Benchmark runs automatically
- Results save to your Google Drive

### 6. View Results
- Option A: Auto-sync results back to dashboard (future feature)
- Option B: Download from Drive manually and refresh dashboard

---

## What is Google Colab?

### Benefits
✅ **Free GPU Computing**
- T4 GPU: ~10-50x faster than CPU for training
- A100 GPU: ~100x faster (limited availability)
- TPU: ~500x faster for certain tasks

✅ **No Installation Required**
- Python, PyTorch, TensorFlow pre-installed
- All dependencies available

✅ **Free Cloud Storage**
- 15GB free Google Drive
- Save results automatically

✅ **Anywhere Access**
- Run from any device
- Check progress from phone
- Pause/resume anytime

### Free Tier Limits
- **GPU Time**: ~15-30 hours/week (varies)
- **RAM**: 25GB available
- **Disk**: 100GB temporary storage
- **Timeout**: 12 hours idle disconnect

---

## Why Use Colab Instead of Local?

| Feature | Local | Colab |
|---------|-------|-------|
| GPU Cost | $$ | Free |
| Setup Time | Hours | Minutes |
| Training Speed | Slow (CPU) | Fast (GPU) |
| Full Dataset | Limited | Unlimited |
| 24/7 Access | Yes | 12h sessions |
| Local Hardware | Required | Not needed |

**Recommendation**: Use Colab for large-scale benchmarks, use Local for quick tests.

---

## Step-by-Step Colab Instructions

### Step 1: Configure in Streamlit UI
```
Run Benchmark Tab
├─ Select Dataset: unsw_nb15
├─ Models: [Select All] or pick favorites
├─ Epochs: 50
├─ Runs: 1
├─ Max Samples: Full (don't limit)
└─ Execution: Google Colab (GPU/TPU)
```

### Step 2: Copy Generated Code
Click "🚀 START BENCHMARK" → Code appears → Click "Copy"

### Step 3: Open Google Colab
1. Visit [colab.research.google.com](https://colab.research.google.com)
2. Login with your Google account
3. Click `+ New notebook`

### Step 4: Set GPU/TPU
**Important**: Enable GPU before running code!

1. Click `Runtime` menu (top)
2. Select `Change runtime type`
3. Choose:
   - **GPU**: T4 or higher (recommended for CNN/Transformer)
   - **TPU**: v2 or v3 (best for RNN/LSTM models)
4. Click `Save`

### Step 5: Paste & Run Code
1. Click first cell (includes code from UI)
2. Paste the copied code
3. Press `Ctrl+Enter` or click ▶ button
4. Authorize Google Drive access when prompted

### Step 6: Monitor Progress
- Watch cell output for training progress
- Check GPU usage: `!nvidia-smi`
- Estimated time: 1-3 hours depending on models/epochs

### Step 7: Results
After completion, results saved to:
```
Google Drive
└── NetGuard_Results/
    ├── metrics/
    │   ├── all_results.json
    │   └── [other metrics]
    └── plots/
        ├── roc_curves.png
        ├── confusion_matrix.png
        └── [other visualizations]
```

### Step 8: Download Results
**Option A (Recommended)**: Manual Download
1. Go to Google Drive
2. Navigate to `NetGuard_Results`
3. Download all files
4. Extract to `backend/results/`
5. Refresh Streamlit Dashboard

**Option B (Automatic)**: Coming soon!
- Dashboard will auto-sync from Drive

---

## Example: Full UNSW-NB15 Benchmark on Colab GPU

```python
# Step 1: Install dependencies
!pip install -q torch pytorch-lightning scikit-learn pandas numpy plotly shap xgboost

# Step 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Clone repository
!cd /content && git clone https://github.com/yourusername/Network-Anomaly-Detection.git
# (Replace with your actual repo URL)

# Step 4: Run benchmark
import os
os.chdir('/content/Network-Anomaly-Detection/backend')

import subprocess
result = subprocess.run([
    'python', 'run_experiments.py',
    '--dataset', 'unsw_nb15',
    '--epochs', '50',
    '--runs', '1',
    '--max-samples', '100000',  # Full dataset!
    '--models', 'vanilla_ae', 'vae', 'cnn1d', 'bilstm_attention',
              'cnn_lstm', 'ft_transformer', 'contrastive_ssl',
              'isolation_forest', 'ensemble_stacking'
])

# Step 5: Save results to Drive
import shutil
os.makedirs('/content/drive/MyDrive/NetGuard_Results', exist_ok=True)
shutil.copytree('results', '/content/drive/MyDrive/NetGuard_Results', dirs_exist_ok=True)

print("✅ Benchmark complete! Check Google Drive for results.")
```

Expected output:
```
============================================================
  NETWORK ANOMALY DETECTION BENCHMARK v2.0
  Device: cuda:0
  Dataset: unsw_nb15
...
  EXPERIMENT COMPLETE
  Total time: 180.5s (3.0min)
============================================================

✅ Benchmark complete! Check Google Drive for results.
```

---

## Performance Tips

### 1. Enable GPU Before Running
```python
# Check GPU status
!nvidia-smi

# Check PyTorch can see GPU
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show T4 or A100
```

### 2. Optimize for Speed
- Use `--epochs 50` not 500 (diminishing returns after 50)
- Use `--max-samples 100000` (full UNSW-NB15 is ~250K samples)
- Skip `--skip-adversarial` for basic benchmarks (adds 2+ hours)
- Skip `--skip-explainability` unless needed (adds 1+ hour)

### 3. Monitor GPU Memory
```python
# Check GPU memory usage during training
!watch -n 1 nvidia-smi
```

### 4. Multiple Runs for Statistics
```
--runs 3  # Run 3 times with different seeds for statistical significance
```

### 5. Full Dataset Advantage
Since Colab has 25GB RAM:
```
--max-samples 250000  # Full UNSW-NB15 dataset
--dataset unsw_nb15
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce `--max-samples` or use TPU instead

### Issue: Repository "not found"
**Solution**: Update GitHub URL to your actual repository:
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Issue: Google Drive quota exceeded
**Solution**: 
1. Check Drive storage (settings)
2. Clear old results: `rm -rf /content/drive/MyDrive/NetGuard_Results/*`
3. Try again

### Issue: Session timeout (disconnected)
**Solution**:
- Colab auto-disconnects after 12 hours idle
- Benchmark usually completes in 1-3 hours
- If needed, run shorter jobs: `--epochs 20` instead of 50

### Issue: GPU not detected
**Solution**:
1. Go to Runtime → Change runtime type
2. Select GPU (not None)
3. Click Save
4. Restart kernel: Runtime → Restart runtime

### Issue: Results not appearing in Drive
**Solution**:
1. Check manual creation: `/content/drive/MyDrive/NetGuard_Results`
2. Use full paths in copy command:
   ```python
   import shutil
   shutil.copytree(
       '/content/Network-Anomaly-Detection/backend/results',
       '/content/drive/MyDrive/NetGuard_Results',
       dirs_exist_ok=True
   )
   ```

---

## Advanced Usage

### Run Multiple Benchmarks in Parallel
Open multiple Colab tabs → run different model sets simultaneously

### Custom Model Testing
Modify the models list:
```python
'--models', 'vanilla_ae', 'cnn1d'  # Only test 2 models
```

### Experiment Tracking
Each run creates timestamp folder:
```
backend/results/metrics/
├── 20260406_120000/  # Run timestamp
│   ├── all_results.json
│   └── plots/
```

### Post-Processing in Colab
```python
# After benchmark, analyze in same notebook
import json
import pandas as pd

with open('/content/Network-Anomaly-Detection/backend/results/metrics/all_results.json') as f:
    results = json.load(f)

df = pd.DataFrame([
    {**v, 'model': k} 
    for k, v in results.items() 
    if isinstance(v, dict)
])

# Find best model
best = df.loc[df['f1_score'].idxmax()]
print(f"Best Model: {best['model']} (F1: {best['f1_score']:.4f})")
```

---

## Frequently Asked Questions

**Q: Is Colab really free?**
A: Yes! Free tier includes GPU/TPU for reasonable usage.

**Q: Can I run 500 epochs on Colab?**
A: Yes, but it's slow. 50-100 epochs gives best results/time trade-off.

**Q: How long does full benchmark take?**
A: ~1.5-3 hours with GPU for all 9 architectures on UNSW-NB15.

**Q: Can I interrupt and resume?**
A: No, session is lost on disconnect. Design jobs to complete <12 hours.

**Q: Do I need to pay for GPU?**
A: No, free tier includes GPU. Premium ($10/mo) gives A100 access.

**Q: How to use own dataset?**
A: Upload CSV to Drive, modify data loading in code.

---

## Contact & Support

For issues or questions:
1. Check Help tab in Streamlit
2. Review Colab docs: [colab.research.google.com/notebooks](https://colab.research.google.com)
3. Check project README.md

**Happy Benchmarking on the Cloud! 🚀**
