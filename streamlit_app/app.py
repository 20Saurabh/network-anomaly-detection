"""
NetGuard v2.0 - Network Anomaly Detection Dashboard
Comprehensive benchmarking, evaluation, and comparison framework
"""

import streamlit as st
import os
import sys
import subprocess
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import Colab integration
try:
    from colab_integration import ColabIntegration, display_colab_instructions
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

st.set_page_config(
    page_title="NetGuard v2.0 Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
RESULTS_DIR = BACKEND_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Add backend to path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Create directories if needed
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #f5f5f5;
        color: #333;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_metrics(filename="all_results.json"):
    """Load metrics from JSON file."""
    metrics_file = METRICS_DIR / filename
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    return None

@st.cache_data
def load_plot_image(plot_name):
    """Load PNG plot image."""
    plot_file = PLOTS_DIR / f"{plot_name}.png"
    if plot_file.exists():
        return plot_file
    return None

def convert_metrics_to_dataframe(metrics):
    """Convert metrics dict to DataFrame for comparison."""
    rows = []
    for model_name, data in metrics.items():
        if isinstance(data, dict) and "error" not in data:
            row = {"Model": model_name}
            row.update(data)
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def generate_comparison_report(df):
    """Generate a detailed comparison report."""
    report = "# NetGuard v2.0 - Model Comparison Report\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if df.empty:
        return report + "No results available for comparison."
    
    # Summary statistics
    report += "## Summary Statistics\n\n"
    report += f"**Total Models Evaluated:** {len(df)}\n\n"
    
    # Best models by metric
    metrics_to_compare = ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc']
    for metric in metrics_to_compare:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_value = df.loc[best_idx, metric]
            report += f"- **Best {metric}:** {best_model} ({best_value:.4f})\n"
    
    report += "\n## Detailed Results Table\n\n"
    
    # Create detailed table
    display_cols = ['Model'] + [c for c in ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc', 'latency_ms_per_sample'] if c in df.columns]
    display_df = df[display_cols].copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if col != 'Model' and display_df[col].dtype in [float, np.float64]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    report += display_df.to_markdown(index=False) + "\n\n"
    
    # Performance insights
    report += "## Key Insights\n\n"
    
    if 'latency_ms_per_sample' in df.columns and 'f1_score' in df.columns:
        fastest = df.loc[df['latency_ms_per_sample'].idxmin()]
        most_accurate = df.loc[df['f1_score'].idxmax()]
        
        report += f"**Fastest Model:** {fastest['Model']} ({fastest['latency_ms_per_sample']:.3f}ms per sample)\n\n"
        report += f"**Most Accurate Model:** {most_accurate['Model']} (F1: {most_accurate['f1_score']:.4f})\n\n"
    
    # Model rankings
    report += "## Model Rankings (by F1 Score)\n\n"
    if 'f1_score' in df.columns:
        ranked = df[['Model', 'f1_score']].sort_values('f1_score', ascending=False).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            report += f"{idx+1}. **{row['Model']}** - {row['f1_score']:.4f}\n"
    
    return report

def run_backend_experiment(dataset, models, epochs, runs, max_samples, skip_flags):
    """Execute backend experiment."""
    cmd = [
        sys.executable,
        str(BACKEND_DIR / "run_experiments.py"),
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--runs", str(runs),
        "--max-samples", str(max_samples)
    ]
    
    if models:
        cmd.extend(["--models"] + models)
    
    for flag in skip_flags:
        cmd.append(flag)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=86400  # 24 hour timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Experiment timed out (24 hours)"
    except Exception as e:
        return -1, "", str(e)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🛡️ NetGuard v2.0")
st.sidebar.markdown("---")
st.sidebar.markdown("**Network Anomaly Detection Framework**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Run Benchmark", "Comparison Report", "Help"]
)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    st.title("🛡️ NetGuard Dashboard")
    st.markdown("### Real-time Model Performance Monitoring")
    
    # Load latest metrics
    metrics = load_metrics()
    
    if metrics:
        df = convert_metrics_to_dataframe(metrics)
        
        if not df.empty:
            # Key Metrics Row
            st.markdown("### 📊 Top Performing Models")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_f1_idx = df['f1_score'].idxmax()
                best_f1 = df.loc[best_f1_idx, 'f1_score']
                best_f1_model = df.loc[best_f1_idx, 'Model']
                st.metric("Best F1 Score", f"{best_f1:.4f}", best_f1_model, label_visibility="visible")
            
            with col2:
                best_acc_idx = df['accuracy'].idxmax()
                best_acc = df.loc[best_acc_idx, 'accuracy']
                best_acc_model = df.loc[best_acc_idx, 'Model']
                st.metric("Best Accuracy", f"{best_acc:.4f}", best_acc_model, label_visibility="visible")
            
            with col3:
                if 'auc_roc' in df.columns:
                    best_auc_idx = df['auc_roc'].idxmax()
                    best_auc = df.loc[best_auc_idx, 'auc_roc']
                    best_auc_model = df.loc[best_auc_idx, 'Model']
                    st.metric("Best AUC-ROC", f"{best_auc:.4f}", best_auc_model, label_visibility="visible")
            
            with col4:
                if 'latency_ms_per_sample' in df.columns:
                    fastest_idx = df['latency_ms_per_sample'].idxmin()
                    fastest_latency = df.loc[fastest_idx, 'latency_ms_per_sample']
                    fastest_model = df.loc[fastest_idx, 'Model']
                    st.metric("Fastest Inference", f"{fastest_latency:.3f}ms", fastest_model, label_visibility="visible")
            
            # Performance Comparison Charts
            st.markdown("---")
            st.markdown("### 📈 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # F1 Score Comparison
                fig = px.bar(
                    df.sort_values('f1_score', ascending=False),
                    x='Model',
                    y='f1_score',
                    title='F1 Score by Model',
                    color='f1_score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Accuracy Comparison
                fig = px.bar(
                    df.sort_values('accuracy', ascending=False),
                    x='Model',
                    y='accuracy',
                    title='Accuracy by Model',
                    color='accuracy',
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Latency vs F1 Trade-off
            st.markdown("---")
            st.markdown("### ⚡ Latency vs Accuracy Trade-off")
            
            if 'latency_ms_per_sample' in df.columns and 'f1_score' in df.columns:
                fig = px.scatter(
                    df,
                    x='latency_ms_per_sample',
                    y='f1_score',
                    size='accuracy',
                    color='Model',
                    hover_data=['precision', 'recall'],
                    title='Model Trade-off Analysis: Latency vs F1 Score (size = accuracy)',
                    labels={'latency_ms_per_sample': 'Latency (ms)', 'f1_score': 'F1 Score'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualizations & Analysis
            st.markdown("---")
            st.markdown("### 📊 Detailed Analysis & Visualizations")
            
            # Get available plots
            available_plots = []
            if PLOTS_DIR.exists():
                available_plots = list(PLOTS_DIR.glob("*.png"))
            
            if available_plots:
                # Categorize plots
                perf_plots = [p for p in available_plots if any(x in p.name for x in ["roc", "precision_recall", "confusion", "metrics"])]
                shap_plots = [p for p in available_plots if any(x in p.name for x in ["shap", "importance", "feature"])]
                robust_plots = [p for p in available_plots if any(x in p.name for x in ["robustness", "adversarial", "fgsm", "pgd"])]
                loss_plots = [p for p in available_plots if any(x in p.name for x in ["loss", "history", "training"])]
                
                # Create tabs for different analysis types
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Metrics", "🔍 Explainability (SHAP)", "🛡️ Adversarial Robustness", "📉 Training Loss"])
                
                with tab1:
                    st.markdown("### Performance Evaluation Metrics")
                    if perf_plots:
                        st.markdown(f"**Found {len(perf_plots)} performance plots**")
                        cols = st.columns(min(len(perf_plots), 2))
                        for i, plot_file in enumerate(perf_plots):
                            with cols[i % 2]:
                                try:
                                    st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(), use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error loading plot: {e}")
                    else:
                        st.info("📌 No performance metric plots found. Run benchmark with default settings to generate ROC, Precision-Recall, and confusion matrix plots.")
                
                with tab2:
                    st.markdown("### Explainability Analysis (SHAP & Feature Importance)")
                    if shap_plots:
                        st.markdown(f"**Found {len(shap_plots)} SHAP/importance plots**")
                        for plot_file in shap_plots:
                            try:
                                st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(), use_container_width=True)
                                st.markdown("---")
                            except Exception as e:
                                st.error(f"Error loading plot: {e}")
                    else:
                        st.info("📌 SHAP plots not available. Run benchmark **without** skipping explainability (uncheck 'Skip Explainability') to generate feature importance analysis.")
                
                with tab3:
                    st.markdown("### Adversarial Robustness Evaluation")
                    st.markdown("Robustness to adversarial perturbations (FGSM, PGD attacks)")
                    if robust_plots:
                        st.markdown(f"**Found {len(robust_plots)} robustness plots**")
                        for plot_file in robust_plots:
                            try:
                                st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(), use_container_width=True)
                                st.markdown("---")
                            except Exception as e:
                                st.error(f"Error loading plot: {e}")
                    else:
                        st.info("📌 Robustness plots not available. Run benchmark **without** skipping adversarial robustness (uncheck 'Skip Adversarial Robustness') to generate attack analysis plots.")
                
                with tab4:
                    st.markdown("### Training Loss & Convergence History")
                    if loss_plots:
                        st.markdown(f"**Found {len(loss_plots)} loss history plots**")
                        for plot_file in loss_plots:
                            try:
                                st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(), use_container_width=True)
                                st.markdown("---")
                            except Exception as e:
                                st.error(f"Error loading plot: {e}")
                    else:
                        st.info("📌 Loss history plots not found. Check backend/results/plots/ directory after running benchmark.")
            else:
                st.info("📌 **No visualizations available yet.** Run a benchmark to generate plots (ROC, SHAP, adversarial robustness, loss curves).")
            
            # Full Results Table
            st.markdown("---")
            st.markdown("### 📋 Complete Results")
            
            display_cols = ['Model'] + [c for c in ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc', 'latency_ms_per_sample'] if c in df.columns]
            st.dataframe(
                df[display_cols].style.highlight_max(subset=['f1_score', 'accuracy', 'auc_roc'], color='lightgreen')
                                    .highlight_min(subset=['latency_ms_per_sample'], color='lightgreen'),
                use_container_width=True
            )
        else:
            st.warning("No valid metric data found.")
    else:
        st.info("📌 **No results yet.** Run a benchmark to see results here.")

# ============================================================================
# PAGE: RUN BENCHMARK
# ============================================================================

elif page == "Run Benchmark":
    st.title("🚀 Run Benchmark")
    st.markdown("Configure and execute a full benchmarking experiment across all architectures.")
    
    st.markdown("---")
    
    # Execution Mode Selection
    execution_mode = st.radio(
        "Choose Execution Environment",
        ["Local Machine", "🐍 Google Colab (GPU/TPU)"],
        help="Local = your computer | Colab = Free GPU/TPU from Google"
    )
    
    st.markdown("---")
    st.markdown("### Experiment Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox(
            "Select Dataset",
            ["synthetic", "unsw_nb15", "ciciot2023", "edge_iiot", "cicids2017", "custom"],
            help="Choose dataset for benchmarking"
        )
        
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=500,
            value=50,
            help="Number of training epochs"
        )
        
        skip_adversarial = st.checkbox(
            "Skip Adversarial Robustness",
            value=False,
            help="Skip FGSM/PGD attacks"
        )
    
    with col2:
        runs = st.number_input(
            "Number of Runs",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of runs for statistical analysis"
        )
        
        max_samples = st.number_input(
            "Max Samples",
            min_value=100,
            max_value=1000000,
            value=50000,
            step=5000,
            help="Limit samples for faster execution"
        )
        
        skip_explainability = st.checkbox(
            "Skip Explainability (SHAP)",
            value=True,
            help="Skip SHAP analysis (slow)"
        )
    
    st.markdown("---")
    st.markdown("### Select Models to Benchmark")
    
    all_models = [
        "vanilla_ae", "vae", "cnn1d", "bilstm_attention",
        "cnn_lstm", "ft_transformer", "contrastive_ssl",
        "isolation_forest", "ensemble_stacking"
    ]
    
    # Quick select all or none
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✓ Select All", use_container_width=True):
            st.session_state.selected_models = all_models
    with col2:
        if st.button("✗ Clear All", use_container_width=True):
            st.session_state.selected_models = []
    with col3:
        if st.button("⚡ Quick Test", use_container_width=True):
            st.session_state.selected_models = ["cnn1d", "ft_transformer"]
    
    # Model selection
    selected_models = st.multiselect(
        "Choose Models",
        all_models,
        default=["cnn1d", "ft_transformer", "isolation_forest"],
        key="selected_models"
    )
    
    st.markdown("---")
    
    # Run button
    if st.button("🚀 START BENCHMARK", type="primary", use_container_width=True):
        if not selected_models:
            st.error("❌ Please select at least one model.")
        else:
            if execution_mode == "🐍 Google Colab (GPU/TPU)":
                # Display Colab instructions
                st.markdown("---")
                display_colab_instructions(dataset, selected_models, epochs, runs, max_samples)
                
            else:
                # Local execution
                with st.spinner("⏳ Executing benchmark... This may take a while."):
                    st.markdown("### Execution Progress")
                    
                    skip_flags = []
                    if skip_adversarial:
                        skip_flags.append("--skip-adversarial")
                    if skip_explainability:
                        skip_flags.append("--skip-explainability")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    output_area = st.empty()
                    
                    status_text.info(f"🔄 Starting benchmark with {len(selected_models)} models...")
                    
                    returncode, stdout, stderr = run_backend_experiment(
                        dataset, selected_models, epochs, runs, max_samples, skip_flags
                    )
                    
                    output_area.code(stdout[:2000] if stdout else (stderr[:2000] if stderr else "No output"))
                    
                    if returncode == 0:
                        progress_bar.progress(100)
                        st.success("✅ Benchmark completed successfully!")
                        st.balloons()
                        
                        # Offer to view results
                        if st.button("📊 View Results", type="primary"):
                            st.switch_page("pages/Dashboard")
                    else:
                        st.error(f"❌ Benchmark failed with exit code {returncode}")
                        if stderr:
                            st.error(f"Error:\n{stderr}")

# ============================================================================
# PAGE: COMPARISON REPORT
# ============================================================================

elif page == "Comparison Report":
    st.title("📊 Model Comparison Report")
    st.markdown("Comprehensive analysis and rankings of all evaluated architectures.")
    
    metrics = load_metrics()
    
    if metrics:
        df = convert_metrics_to_dataframe(metrics)
        
        if not df.empty:
            # Generate report
            report_text = generate_comparison_report(df)
            
            # Display report
            st.markdown(report_text)
            
            # Interactive ranking table
            st.markdown("---")
            st.markdown("### 🏆 Detailed Rankings")
            
            ranking_metrics = ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc']
            selected_metric = st.selectbox("Rank by Metric", ranking_metrics)
            
            if selected_metric in df.columns:
                ranked_df = df[['Model', selected_metric]].sort_values(selected_metric, ascending=False).reset_index(drop=True)
                ranked_df.index = ranked_df.index + 1
                ranked_df.index.name = 'Rank'
                
                st.dataframe(ranked_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    ranked_df.reset_index(),
                    x='Model',
                    y=selected_metric,
                    title=f'Model Rankings by {selected_metric}',
                    color=selected_metric,
                    color_continuous_scale='RdYlGn',
                    labels={'Rank': 'Rank'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export report
            st.markdown("---")
            st.markdown("### 📥 Export Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"netguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps(metrics, indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"netguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.warning("No valid metrics to display.")
    else:
        st.info("No results available yet. Run a benchmark first.")

# ============================================================================
# PAGE: HELP
# ============================================================================

elif page == "Help":
    st.title("❓ Help & Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Getting Started", "Colab Guide", "Troubleshooting"])
    
    with tab1:
        st.markdown("""
        ## Getting Started
        
        ### 1. Run a Benchmark
        - Go to **Run Benchmark** tab
        - Select dataset (UNSW-NB15 recommended for full evaluation)
        - Choose execution mode: Local or Google Colab
        - Select models to evaluate
        - Click **START BENCHMARK**
        - Wait for completion (time varies by model count and dataset size)
        
        ### 2. View Results
        - Results appear automatically on **Dashboard** after completion
        - View key metrics and performance charts
        - Analyze latency vs accuracy trade-offs
        
        ### 3. Generate Report
        - Visit **Comparison Report** tab
        - View detailed model rankings
        - Export results as CSV or JSON
        
        ---
        
        ## Recommended Configurations
        
        ### Quick Test (5-10 min - Local)
        - Dataset: Synthetic
        - Models: cnn1d, ft_transformer
        - Epochs: 5-10
        - Samples: 5000
        - Disable: Adversarial, Explainability
        
        ### Standard Benchmark (30-60 min - Local)
        - Dataset: UNSW-NB15
        - Models: All Core (vanilla_ae, vae, cnn1d, bilstm_attention, cnn_lstm, ft_transformer, isolation_forest, contrastive_ssl)
        - Epochs: 20-30
        - Samples: 50000
        
        ### Full Evaluation (1-2 hours - Colab with GPU)
        - Dataset: UNSW-NB15
        - Models: All including ensemble_stacking
        - Epochs: 50+
        - Samples: Full dataset
        - Enable: All options
        
        ---
        
        ## Key Metrics Explained
        
        - **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
        - **Accuracy**: Percentage of correct predictions (0-1, higher is better)
        - **Precision**: True positives / (True positives + False positives)
        - **Recall**: True positives / (True positives + False negatives)
        - **AUC-ROC**: Area under ROC curve (0-1, higher is better)
        - **Latency**: Inference time per sample in milliseconds (lower is better)
        """)
    
    with tab2:
        st.markdown("""
        ## 🐍 Google Colab GPU/TPU Guide
        
        ### What is Google Colab?
        Google Colab provides **free cloud computing** with:
        - ✅ **GPU**: NVIDIA T4, A100 (free tier available)
        - ✅ **TPU**: v2/v3 (in select regions)
        - ✅ **RAM**: 25GB+ available
        - ✅ **No installation**: Everything pre-configured
        - ✅ **No cost**: Completely free for reasonable usage
        
        ### Why Use Colab?
        1. **Speed**: GPU training is 10-50x faster than CPU
        2. **Free**: No GPU purchase needed
        3. **No Setup**: Nothing to install locally
        4. **Always Available**: Work from any device
        5. **Cloud Storage**: Save results to Google Drive
        
        ### Step-by-Step Guide
        
        #### Step 1: Configure Benchmark
        1. Set parameters in the UI (dataset, models, epochs, etc.)
        2. Select "🐍 Google Colab (GPU/TPU)" option
        3. Click "🚀 START BENCHMARK"
        
        #### Step 2: Copy Generated Code
        - UI will show you the Python code to run
        - Click "Copy" to copy to clipboard
        
        #### Step 3: Open Google Colab
        1. Go to [colab.research.google.com](https://colab.research.google.com)
        2. Click "+ New notebook"
        3. Paste the code into the first cell
        4. Press Ctrl+Enter to run
        
        #### Step 4: Authenticate with Google Drive
        1. First cell will ask for Google Drive permission
        2. Click the authorization link
        3. Select your Google account
        4. Grant access to Colab
        
        #### Step 5: Wait for Results
        1. Benchmark will run automatically
        2. Watch the progress in cell output
        3. Results save to `/content/drive/MyDrive/NetGuard_Results`
        
        #### Step 6: Download Results (Optional)
        1. Go to your Google Drive
        2. Navigate to `NetGuard_Results` folder
        3. Download the files
        4. Upload to local `backend/results/` (optional)
        
        ### Quick Copy-Paste Template
        ```python
        # NetGuard Benchmark on Colab
        !pip install -q torch pytorch-lightning scikit-learn pandas numpy plotly shap xgboost
        
        from google.colab import drive
        drive.mount('/content/drive')
        
        !cd /content && git clone https://github.com/yourusername/Network-Anomaly-Detection.git 2>/dev/null || true
        
        import os, subprocess
        os.chdir('/content/Network-Anomaly-Detection/backend')
        
        result = subprocess.run([
            'python', 'run_experiments.py',
            '--dataset', 'unsw_nb15',
            '--epochs', '50',
            '--runs', '1',
            '--max-samples', '50000',
            '--models', 'cnn1d', 'ft_transformer', 'vanilla_ae', 'vae'
        ])
        
        !mkdir -p /content/drive/MyDrive/NetGuard_Results
        !cp -r results/* /content/drive/MyDrive/NetGuard_Results/ 2>/dev/null || true
        ```
        
        ### Pro Tips
        - **Select GPU**: Runtime → Change runtime type → GPU (T4 or higher)
        - **Enable TPU**: Runtime → Change runtime type → TPU (faster for some models!)
        - **Monitor**: Check GPU usage: `!nvidia-smi`
        - **Full Dataset**: Use `--max-samples` without limit for better results
        - **Multiple Runs**: Set `--runs 5` for statistical significance testing
        """)
    
    with tab3:
        st.markdown("""
        ## Troubleshooting
        
        ### Local Execution Issues
        
        **Q: Out of memory error**
        - A: Reduce `max_samples`, disable `--skip-adversarial`, or reduce `epochs`
        
        **Q: Benchmark taking too long**
        - A: Use GPU compatible model (no GNN), reduce models count, or use Colab
        
        **Q: CUDA out of memory**
        - A: Reduce `max_samples` or batch size in config.py
        
        ---
        
        ### Google Colab Issues
        
        **Q: "Repository not found" error**
        - A: Update the GitHub URL in the code snippet to your repo
        
        **Q: Google Drive authentication fails**
        - A: Re-run the drive.mount() cell and re-authorize
        
        **Q: Results not saving to Drive**
        - A: Check `/content/drive/MyDrive/` folder exists and you have write permissions
        
        **Q: GPU session timeout**
        - A: Colab auto-disconnects after 12+ hours idle. Use shorter `--epochs` or run multiple smaller jobs
        
        **Q: How to use my own GitHub repo?**
        - A: Edit this line in the code:
          ```python
          !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
          ```
        
        ---
        
        ### General Questions
        
        **Q: No results appearing on Dashboard**
        - A: Manually upload results from Colab to `backend/results/metrics/`
        
        **Q: How to compare Colab vs Local results?**
        - A: Save both under different `--run-name` timestamps
        
        **Q: Can I use PyTorch Geometric (torch_geometric)?**
        - A: Colab has it pre-installed! Uncomment GNN models in UI to use
        
        **Q: How much free GPU time do I get?**
        - A: Typically 15-30 hours per week of GPU time (varies by usage)
        
        ---
        
        ### System Requirements
        
        **Local:**
        - Python 3.9+
        - PyTorch 2.0+
        - 4GB+ RAM
        - GPU optional (CPU will be slow)
        
        **Colab:**
        - Google account (free)
        - Internet connection
        - That's it!
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("🛡️ NetGuard v2.0 | Network Anomaly Detection Research Framework | Local + Google Colab Support")