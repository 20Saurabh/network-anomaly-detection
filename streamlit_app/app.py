import streamlit as st
import os
import sys
import subprocess
import json
import time
import pandas as pd
from pathlib import Path
import glob
import re
import threading
import queue
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="NetGuard v2.0 - Research Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
RESULTS_BASE = BACKEND_DIR / "results"
RUNS_DIR = RESULTS_BASE / "runs"
DATA_DIR = BACKEND_DIR / "data" / "raw"

# Add backend to path for direct config access if needed
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stAlert {
        border-radius: 10px;
    }
    .plot-container {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .reportview-container .main .block-container{
        padding-top: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #1e293b;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_available_datasets():
    datasets = ["synthetic"]
    if DATA_DIR.exists():
        for d in DATA_DIR.iterdir():
            if d.is_dir() and any(d.iterdir()):
                datasets.append(d.name)
    return datasets

def get_historical_runs():
    if not RUNS_DIR.exists():
        return []
    runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir()]
    return sorted(runs, reverse=True)

def load_run_results(run_name):
    run_path = RUNS_DIR / run_name
    metrics_file = run_path / "metrics" / "all_results.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return None
    return None

def get_run_plots(run_name):
    run_path = RUNS_DIR / run_name
    plots_dir = run_path / "plots"
    if plots_dir.exists():
        return list(plots_dir.glob("*.png"))
    return []

def run_experiment_async(cmd, output_queue):
    """Run experiment in background thread."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(PROJECT_ROOT)
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output_lines.append(line.strip())
                output_queue.put(line.strip())
        
        process.wait()
        output_queue.put(f"PROCESS_EXIT:{process.returncode}")
        
    except Exception as e:
        output_queue.put(f"ERROR:{str(e)}")

# --- Sidebar Configuration ---

st.sidebar.title("🛡️ NetGuard v2.0")
st.sidebar.markdown("Research-Grade NIDS Benchmarking")

# Mode Selection
app_mode = st.sidebar.radio("Navigation", ["Run Benchmark", "View History", "Research Insights", "Upload Data"])

if app_mode == "Run Benchmark":
    st.sidebar.subheader("Configuration")
    
    selected_dataset = st.sidebar.selectbox("Select Dataset", get_available_datasets())
    
    available_models = [
        "vanilla_ae", "vae", "cnn1d", "bilstm_attention",
        "cnn_lstm", "ft_transformer", "e_graphsage", 
        "gnn_transformer", "contrastive_ssl", 
        "isolation_forest", "ensemble_stacking"
    ]
    selected_models = st.sidebar.multiselect("Models to Benchmark", available_models, default=["ft_transformer", "cnn1d"])
    
    epochs = st.sidebar.number_input("Max Epochs", min_value=1, max_value=500, value=10)
    runs = st.sidebar.number_input("Runs (for Statistics)", min_value=1, max_value=10, value=1)
    max_samples = st.sidebar.number_input("Max Samples (Subsampling)", value=10000, step=5000)
    
    skip_adversarial = st.sidebar.checkbox("Skip Adversarial Robustness", value=False)
    skip_explainability = st.sidebar.checkbox("Skip Explainability (SHAP)", value=True)
    
    if st.sidebar.button("🚀 Start Benchmark Pipeline", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to benchmark.")
        else:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            
            st.header(f"🌀 Benchmark Run: {run_id}")
            
            # Prepare command
            cmd = [
                sys.executable, "backend/run_experiments.py",
                "--dataset", selected_dataset,
                "--epochs", str(epochs),
                "--runs", str(runs),
                "--max-samples", str(max_samples),
                "--run-name", run_id
            ]
            if selected_models:
                cmd.append("--models")
                cmd.extend(selected_models)
            if skip_adversarial:
                cmd.append("--skip-adversarial")
            if skip_explainability:
                cmd.append("--skip-explainability")
                
            # Run in background thread
            output_queue = queue.Queue()
            thread = threading.Thread(target=run_experiment_async, args=(cmd, output_queue))
            thread.start()
            
            # Display progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            console_output = st.empty()
            
            output_buffer = []
            current_epoch = 0
            total_epochs = epochs
            
            while thread.is_alive() or not output_queue.empty():
                try:
                    line = output_queue.get(timeout=0.1)
                    if line.startswith("PROCESS_EXIT:"):
                        exit_code = int(line.split(":")[1])
                        if exit_code == 0:
                            st.session_state["latest_run"] = run_id
                            st.success("Benchmark Execution Complete!")
                            st.balloons()
                        else:
                            st.error(f"Execution failed with return code {exit_code}")
                        break
                    elif line.startswith("ERROR:"):
                        st.error(f"Execution error: {line[6:]}")
                        break
                    else:
                        output_buffer.append(line)
                        console_output.code("\n".join(output_buffer[-20:]))
                        
                        # Progress detection
                        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                        if epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            progress = min(current_epoch / total_epochs, 0.99)
                            progress_bar.progress(progress)
                            status_text.info(f"Training in progress: Epoch {current_epoch}/{total_epochs}")
                            
                        if "EXPERIMENT COMPLETE" in line:
                            progress_bar.progress(1.0)
                            status_text.success("Benchmark Execution Complete!")
                            
                except queue.Empty:
                    pass
            
            thread.join()

elif app_mode == "View History":
    st.sidebar.subheader("History Browser")
    if st.sidebar.button("🔄 Refresh History"):
        st.rerun()
    
    history = get_historical_runs()
    if not history:
        st.warning("No benchmark history found. Run a benchmark first!")
    else:
        selected_run = st.sidebar.selectbox("Select Historical Run", history)
        
        st.header(f"📊 Experiment Analysis: {selected_run}")
        
        results = load_run_results(selected_run)
        if results:
            # Metrics Table
            st.subheader("Model Performance Comparison")
            
            # Convert to DataFrame for visualization
            rows = []
            for model, metrics in results.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    metrics["Model"] = model
                    rows.append(metrics)
            
            if rows:
                df = pd.DataFrame(rows)
                # Reorder columns
                cols = ["Model", "f1_score", "accuracy", "precision", "recall", "auc_roc", "latency_ms_per_sample"]
                available_cols = [c for c in cols if c in df.columns]
                df = df[available_cols]
                
                st.dataframe(
                    df.style.highlight_max(axis=0, subset=["f1_score", "accuracy", "auc_roc"], color="#10b981")
                            .highlight_min(axis=0, subset=["latency_ms_per_sample"], color="#10b981"),
                    use_container_width=True
                )
                
                # Highlight best model
                best_model = df.loc[df['f1_score'].idxmax()]
                st.success(f"🏆 **Best Performing Architecture:** {best_model['Model']} (F1: {best_model['f1_score']:.4f})")
                
                # Interactive Plotly chart
                st.subheader("Interactive Performance Scatter Plot")
                fig = px.scatter(df, x="latency_ms_per_sample", y="f1_score", 
                               size="accuracy", color="Model", hover_name="Model",
                               title="F1 Score vs Latency Trade-off")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display Plots
                st.divider()
                st.subheader("Visual Analytics")
                plots = get_run_plots(selected_run)
                
                if plots:
                    tabs = st.tabs(["Performance Metrics", "Explainability", "Robustness", "Loss Curves"])
                    
                    with tabs[0]:
                        # Performance plots
                        perf_plots = [p for p in plots if any(x in p.name for x in ["metrics", "roc", "precision_recall", "confusion"])]
                        if perf_plots:
                            cols = st.columns(min(len(perf_plots), 2))
                            for i, p in enumerate(perf_plots):
                                cols[i % 2].image(str(p), caption=p.stem.replace("_", " ").title())
                        else:
                            st.info("No performance plots found.")
                            
                    with tabs[1]:
                        shap_plots = [p for p in plots if "shap" in p.name or "importance" in p.name]
                        if shap_plots:
                            cols = st.columns(min(len(shap_plots), 2))
                            for i, p in enumerate(shap_plots):
                                cols[i % 2].image(str(p), caption=p.stem.replace("_", " ").title())
                        else:
                            st.info("Explainability data not generated for this run.")
                            
                    with tabs[2]:
                        robust_plots = [p for p in plots if "robustness" in p.name or "adversarial" in p.name]
                        if robust_plots:
                            for p in robust_plots:
                                st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)
                        else:
                            st.info("Adversarial robustness data not generated for this run.")
                            
                    with tabs[3]:
                        loss_plots = [p for p in plots if "loss" in p.name or "history" in p.name]
                        if loss_plots:
                            for p in loss_plots:
                                st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)
                        else:
                            st.info("Loss history plots not found.")
                else:
                    st.info("No visualization artifacts found for this run.")
            else:
                st.error("Invalid metrics format in result file.")
        else:
            st.error("Could not load results for this run.")

elif app_mode == "Research Insights":
    st.header("🔬 Deep Insights & Pareto Analysis")
    st.markdown("""
    This section helps you pinpoint the absolute best architectures for production deployment 
    based on the **Accuracy vs. Latency** trade-off.
    """)
    
    history = get_historical_runs()
    if history:
        selected_run = st.selectbox("Select Run for Pareto Analysis", history)
        results = load_run_results(selected_run)
        if results:
            rows = []
            for model, metrics in results.items():
                if isinstance(metrics, dict) and "f1_score" in metrics:
                    metrics["Model"] = model
                    rows.append(metrics)
            
            if rows:
                df = pd.DataFrame(rows)
                
                # Altair chart
                chart = alt.Chart(df).mark_circle(size=100).encode(
                    x=alt.X('latency_ms_per_sample', title='Inference Latency (ms)'),
                    y=alt.Y('f1_score', title='F1 Score'),
                    color='Model',
                    tooltip=['Model', 'f1_score', 'latency_ms_per_sample']
                ).interactive().properties(height=500)
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("Lower-down (Low Latency) and To-the-right (High F1) is optimal.")
                
                # Pareto front analysis
                st.subheader("Pareto Front Analysis")
                # Sort by latency ascending, then f1 descending
                pareto_df = df.sort_values(['latency_ms_per_sample', 'f1_score'], ascending=[True, False])
                pareto_front = []
                max_f1 = 0
                for _, row in pareto_df.iterrows():
                    if row['f1_score'] > max_f1:
                        pareto_front.append(row)
                        max_f1 = row['f1_score']
                
                pareto_df = pd.DataFrame(pareto_front)
                st.dataframe(pareto_df[['Model', 'f1_score', 'latency_ms_per_sample']], use_container_width=True)
                st.info("These models represent the Pareto-optimal solutions (best trade-offs).")
            else:
                st.info("No valid results found for analysis.")
    else:
        st.info("Run a benchmark to see research insights.")

elif app_mode == "Upload Data":
    st.header("📤 Upload Custom Dataset")
    st.markdown("""
    Upload your own network traffic CSV file for anomaly detection analysis.
    The file should contain numerical features and a label column (0 for normal, 1 for anomaly).
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", list(df.columns))
            
            # Basic analysis
            if 'label' in df.columns or 'Label' in df.columns:
                label_col = 'label' if 'label' in df.columns else 'Label'
                anomaly_rate = df[label_col].mean()
                st.metric("Anomaly Rate", f"{anomaly_rate:.2%}")
                
                # Save to data/raw/custom
                custom_dir = DATA_DIR / "custom"
                custom_dir.mkdir(exist_ok=True)
                custom_file = custom_dir / "data.csv"
                df.to_csv(custom_file, index=False)
                st.success(f"Data saved to {custom_file}")
                st.info("You can now select 'custom' as dataset in the Run Benchmark section.")
            else:
                st.warning("No 'label' column found. Please ensure your CSV has a label column.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- Footer ---
st.divider()
st.caption("NetGuard v2.0 | Built for IoT and Edge Network Anomaly Detection Research")