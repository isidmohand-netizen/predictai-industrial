from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
# Without this, Python cannot find src/llm_client.py from inside api/
# __file__ = path of this file (api/main.py)
# .parent = api/ folder
# .parent.parent = project root
sys.path.append(str(Path(__file__).parent.parent))

# Now we can import from src/ as if we were at the project root
from src.llm_client import generate_maintenance_report

# Initialize the FastAPI application
# title, description, version appear automatically in the Swagger docs
app = FastAPI(
    title="PredictAI Industrial",
    description="Predictive maintenance API for industrial equipment",
    version="1.0.0"
)
from pathlib import Path
import os

# Project root — same logic as in the notebook
ROOT = Path(__file__).parent.parent

import pickle

# Load the model from pickle file — works both locally and on Render
model_path = ROOT / "models" / "random_forest_rul.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

print(f"Model loaded successfully from {model_path}")
# Input schema — defines exactly what data the API expects
# Pydantic validates automatically : wrong type = clear error message
# A single cycle reading for one sensor snapshot
class CycleReading(BaseModel):
    cycle: int
    op_1: float
    op_2: float
    op_3: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_14: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


# Full input : unit ID + list of cycle readings (history)
# The more cycles we provide, the better the rolling features
class PredictInput(BaseModel):
    unit_id: int
    # Minimum 10 cycles recommended for reliable rolling features
    history: list[CycleReading]
    # Health check endpoint — confirms the API is running
# Convention : always have a /health endpoint in production APIs
@app.get("/health")
def health_check():
    """Returns API status and model info."""
    return {
        "status": "healthy",
        "model": "random-forest-rul",
        "run_id": RUN_ID
    }


# Main prediction endpoint
# POST because we send data in the request body (not in the URL)
# Main prediction endpoint
# POST because we send data in the request body (not in the URL)
@app.post("/predict")
def predict(data: PredictInput):
    """
    Receives unit history, builds features, returns RUL prediction.
    
    Args:
        data: PredictInput — unit_id + list of cycle readings
    Returns:
        dict with predicted RUL, alert level
    """
    from src.features import build_features, FEATURE_COLS

    # Convert history to dataframe
    records = [cycle.model_dump() for cycle in data.history]
    df = pd.DataFrame(records)
    df['unit_id'] = data.unit_id

    # Build the 48 features — same logic as in the notebook
    df_features = build_features(df)

    # Predict on the last cycle only — most recent state of the machine
    last_cycle = df_features.iloc[[-1]]
    predicted_rul = float(model.predict(last_cycle)[0])

    # Determine alert level
    if predicted_rul <= 20:
        alert_level = "CRITICAL"
    elif predicted_rul <= 50:
        alert_level = "WARNING"
    else:
        alert_level = "NORMAL"

    return {
        "unit_id": data.unit_id,
        "cycle": data.history[-1].cycle,
        "predicted_rul": round(predicted_rul, 1),
        "alert_level": alert_level
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)