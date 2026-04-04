import asyncio
import logging
import random
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging to look professional and complex
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NIDS_CORE_ENGINE")

app = FastAPI(title="NIDS Anomaly Engine", version="2.4.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    filename: str
    row_count: int

async def run_simulation_logic(filename: str, row_count: int):
    """Generates realistic looking logs for a ML pipeline."""
    logger.info(f"Received processing request for dataset: {filename} ({row_count} records)")
    await asyncio.sleep(1)
    
    logger.info("Initializing Data Preprocessing Pipeline...")
    await asyncio.sleep(0.5)
    logger.info(" > Cleaning null values...")
    logger.info(" > Normalizing numerical features (MinMax Scaling)...")
    logger.info(" > Encoding categorical variables...")
    await asyncio.sleep(1)
    logger.info("Data Preprocessing Complete. Feature Matrix Shape: ({}, 14)".format(row_count))

    logger.info("Loading Model Architecture: AutoEncoder_v3 (PyTorch Backend)")
    await asyncio.sleep(0.5)
    logger.info("Model compiled. Optimizer: Adam, Loss: MSE")

    logger.info("Starting Training Loop...")
    epochs = 10
    for i in range(epochs):
        loss = random.uniform(0.1, 0.5) / (i + 1)
        val_loss = loss * 1.1
        logger.info(f"Epoch {i+1}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f} - accuracy: {random.uniform(0.85, 0.99):.4f}")
        await asyncio.sleep(0.3)

    logger.info("Training Converged.")
    logger.info("Running Evaluation on Test Set...")
    await asyncio.sleep(0.5)
    
    logger.warning("Anomaly Detection Threshold Calibration...")
    logger.info("Calculating Reconstruction Error Distribution...")
    logger.info(f"Threshold set at {random.uniform(0.01, 0.05):.6f} (Mean + 2*StdDev)")
    
    logger.info(f"Analysis Finalized. Flagged {random.randint(2, 15)} potential intrusions.")
    logger.info("Result Metadata saved to local storage.")

@app.post("/api/simulate")
async def simulate_processing(req: SimulationRequest, background_tasks: BackgroundTasks):
    # Run in background so frontend doesn't wait
    background_tasks.add_task(run_simulation_logic, req.filename, req.row_count)
    return {"status": "Pipeline Triggered", "job_id": "job_" + str(random.randint(1000, 9999))}

if __name__ == "__main__":
    import uvicorn
    # Clean console startup
    print("NIDS Server Starting...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
