# Anomaly Detection Web Application (Client-Side)

A full-stack application for detecting anomalies in CSV data using **TensorFlow.js** and **React**. 

> **Architecture Note**: This application runs entirely in the browser (Client-Side). There is **no Python backend**. All Machine Learning training and inference happen locally on your device.

## Features
- **Local Processing**: Data never leaves your browser.
- **Deep Learning**: Uses a TensorFlow.js Autoencoder model.
- **Real-time Training**: Watch the loss curve update live as the model trains in-browser.
- **Detailed Metrics**: Anomaly Scores, Confidence levels, and Dynamic Thresholding.

## Setup & Running

### Prerequisites
- Node.js 18+

### Quick Start
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
2. Open `http://localhost:5173` in your browser.
3. Load the included `sample_dataset.csv`.
4. Click **Start Training**.

## How it Works
1. **Data Loading**: `PapaParse` reads the CSV file locally.
2. **Preprocessing**: Data is normalized to [0, 1] range.
3. **Model**: A neural network (Autoencoder) is created using `tf.sequential()` with:
   - Encoder: Compresses imputs to lower dimensions.
   - Decoder: Reconstructs inputs from the compressed state.
4. **Training**: The model learns to reconstruct "normal" data patterns.
5. **Detection**:
   - **Score**: MSE (Mean Squared Error) between Input and Reconstruction. High error = Anomaly.
   - **Threshold**: Automatically set to `Mean + 2 * StdDev` of the scores.
   - **Prediction**: Any row with `Score > Threshold` is flagged.

## Troubleshooting
- **Browser Performance**: Training helps on User Interface thread. For very large datasets (>1MB), the UI might stutter slightly.
