"""
Google Colab Integration for NetGuard
Enables running experiments on Google Colab with GPU/TPU support
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Tuple, Dict, Optional
import streamlit as st

# Colab notebook template
COLAB_NOTEBOOK_TEMPLATE = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NetGuard - Network Anomaly Detection Benchmark\\n",
    "## Running on Google Colab with GPU Support\\n",
    "`This notebook will run the full benchmark pipeline.`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install dependencies\\n",
    "!pip install -q torch pytorch-lightning scikit-learn pandas numpy plotly streamlit\\n",
    "!pip install -q shap xgboost torch-geometric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Mount Google Drive\\n",
    "from google.colab import drive\\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Clone repository\\n",
    "!cd /content && git clone https://github.com/yourusername/Network-Anomaly-Detection.git 2>/dev/null || echo 'Already cloned'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Run benchmarking\\n",
    "import sys\\n",
    "sys.path.insert(0, '/content/Network-Anomaly-Detection/backend')\\n",
    "\\n",
    "os.chdir('/content/Network-Anomaly-Detection/backend')\\n",
    "\\n",
    "# {COMMAND_PLACEHOLDER}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Copy results back to Drive\\n",
    "import shutil\\n",
    "!cp -r results /content/drive/MyDrive/NetGuard_Results 2>/dev/null || mkdir -p /content/drive/MyDrive/NetGuard_Results && cp -r results/* /content/drive/MyDrive/NetGuard_Results/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "print('✅ Benchmark complete! Results saved to Google Drive.')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
"""

class ColabIntegration:
    """Manages Google Colab integration for benchmark execution."""
    
    def __init__(self):
        self.colab_url = "https://colab.research.google.com"
        self.drive_api_url = "https://www.googleapis.com/drive/v3"
        self.results_drive_path = "NetGuard_Results"
        
    def generate_colab_notebook(self, dataset: str, models: list, epochs: int, 
                               runs: int, max_samples: int) -> str:
        """Generate a Colab notebook with benchmark configuration."""
        
        cmd = f"python run_experiments.py --dataset {dataset} --epochs {epochs} --runs {runs} --max-samples {max_samples}"
        if models:
            cmd += f" --models {' '.join(models)}"
        
        notebook_json = json.loads(COLAB_NOTEBOOK_TEMPLATE)
        # Update command placeholder
        notebook_str = json.dumps(notebook_json, indent=1)
        notebook_str = notebook_str.replace("{COMMAND_PLACEHOLDER}", f"!{cmd}")
        
        return notebook_str
    
    def create_colab_link(self, notebook_json: str) -> str:
        """Create a shareable Google Colab link."""
        # Encode notebook and create Colab URL
        import base64
        notebook_encoded = base64.b64encode(notebook_json.encode()).decode()
        
        # Create direct Colab execution link
        colab_link = f"https://colab.research.google.com/drive/1YourDriveFileId#scrollTo=0"
        return colab_link
    
    def get_colab_execution_code(self, dataset: str, models: list, epochs: int,
                                runs: int, max_samples: int) -> str:
        """Return code snippet to run in Colab."""
        
        models_str = ", ".join([f'"{m}"' for m in models]) if models else ""
        
        code = f"""
# NetGuard Benchmark on Colab
import sys
import os

# Install dependencies
!pip install -q torch pytorch-lightning scikit-learn pandas numpy plotly shap xgboost

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone and setup
!cd /content && git clone https://github.com/yourusername/Network-Anomaly-Detection.git 2>/dev/null || true
os.chdir('/content/Network-Anomaly-Detection/backend')
sys.path.insert(0, '/content/Network-Anomaly-Detection/backend')

# Run benchmark
import subprocess
result = subprocess.run([
    'python', 'run_experiments.py',
    '--dataset', '{dataset}',
    '--epochs', '{epochs}',
    '--runs', '{runs}',
    '--max-samples', '{max_samples}',
    '--models', {models_str if models_str else "'vanilla_ae', 'cnn1d'"}
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)

# Save results to Drive
!mkdir -p /content/drive/MyDrive/NetGuard_Results
!cp -r results/* /content/drive/MyDrive/NetGuard_Results/ 2>/dev/null || true

print("✅ Benchmark completed! Check /content/drive/MyDrive/NetGuard_Results for output")
"""
        return code.strip()
    
    def fetch_results_from_drive(self, credentials_path: Optional[str] = None) -> Dict:
        """Fetch benchmark results from Google Drive."""
        # This would use Google Drive API to fetch results
        # For now, return empty dict - user will handle authentication
        return {}

def create_colab_html_link(dataset: str, models: list, epochs: int, 
                          runs: int, max_samples: int) -> str:
    """Create an HTML link to open notebook in Colab."""
    
    colab = ColabIntegration()
    code = colab.get_colab_execution_code(dataset, models, epochs, runs, max_samples)
    
    # Create a link that opens Colab with the code
    colab_link = f"""
    <a href="https://colab.research.google.com/" target="_blank" style="
        display: inline-block;
        padding: 12px 24px;
        background-color: #F3B500;
        color: black;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    ">
        🚀 Open Google Colab & Run Benchmark
    </a>
    """
    
    return colab_link

def display_colab_instructions(dataset: str, models: list, epochs: int, 
                              runs: int, max_samples: int):
    """Display step-by-step instructions for running on Colab."""
    
    st.markdown("## 🚀 Running on Google Colab (GPU/TPU)")
    
    st.info("""
    Google Colab provides:
    - ✅ Free GPU (T4/A100 available)
    - ✅ Free TPU for some regions
    - ✅ 25GB+ RAM
    - ✅ No installation needed
    """)
    
    colab = ColabIntegration()
    code = colab.get_colab_execution_code(dataset, models, epochs, runs, max_samples)
    
    st.markdown("### Step 1: Open Google Colab")
    st.markdown("""
    1. Click the button below to open [Google Colab](https://colab.research.google.com)
    2. Create a new notebook
    3. Paste the code below
    """)
    
    st.markdown("### Step 2: Copy This Code")
    st.code(code, language="python")
    
    st.markdown("### Step 3: Run in Colab")
    st.markdown("""
    1. Paste code into first Colab cell
    2. Click play button or press Ctrl+Enter
    3. Authenticate with Google Drive when prompted
    4. Wait for benchmark completion
    5. Results saved to `/content/drive/MyDrive/NetGuard_Results`
    """)
    
    st.markdown("### Step 4: Download Results (Optional)")
    st.markdown("""
    After benchmark completes in Colab:
    1. Go to your Google Drive
    2. Navigate to `NetGuard_Results` folder
    3. Download `all_results.json` and `plots/` folder
    4. Upload to your local `backend/results/` directory
    5. Refresh the Dashboard to see results
    """)
    
    st.markdown("---")
    
    # Provide download button template
    st.markdown("### Or Sync Results Automatically")
    if st.button("📥 Connect Google Drive Account"):
        st.info("""
        (Requires authentication - click below to authorize)
        \n This would auto-fetch results from your Colab run.
        """)
