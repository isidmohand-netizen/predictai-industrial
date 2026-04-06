from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
# Without this, Python cannot find src/llm_client.py from inside api/
# __file__ = path of this file (api/main.py)
# .parent = api/ folder
# .parent.parent = project root
sys.path.append(str(Path(__file__).parent.parent))

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
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model and ChromaDB at startup
# Same logic as the ML model — load once, use on every request
print("Loading RAG components...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(
    path=str(ROOT / "data" / "chroma_db")
)
collection = chroma_client.get_or_create_collection(name="maintenance_docs")

print(f"RAG ready — {collection.count()} documents indexed")
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
        "model_version": "light-20-trees"
    }


# Main prediction endpoint
# POST because we send data in the request body (not in the URL)
# Main prediction endpoint
# POST because we send data in the request body (not in the URL)
@app.post("/predict")
def predict(data: PredictInput):
    """
    Receives unit history, builds features, returns RUL prediction + RAG diagnosis.
    
    Args:
        data: PredictInput — unit_id + list of cycle readings
    Returns:
        dict with predicted RUL, alert level, and RAG-grounded diagnosis
    """
    from src.features import build_features, FEATURE_COLS

    # Step 1 — Build features from sensor history
    records = [cycle.model_dump() for cycle in data.history]
    df = pd.DataFrame(records)
    df['unit_id'] = data.unit_id
    df_features = build_features(df)

    # Step 2 — Predict RUL on the last cycle
    last_cycle = df_features.iloc[[-1]]
    predicted_rul = float(model.predict(last_cycle)[0])

    # Step 3 — Determine alert level
    if predicted_rul <= 20:
        alert_level = "CRITICAL"
    elif predicted_rul <= 50:
        alert_level = "WARNING"
    else:
        alert_level = "NORMAL"

   # Step 4 — RAG diagnosis only for WARNING and CRITICAL
    # NORMAL status doesn't need LLM call — saves API costs and latency
    diagnosis = None
    relevant_docs = []

    if alert_level in ["WARNING", "CRITICAL"]:

        # Build query from sensor context
        query = f"Engine unit {data.unit_id} — RUL {predicted_rul:.0f} cycles — alert {alert_level}. Sensor anomalies detected."
        
        # Retrieve relevant documents
        query_vector = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )

        # Build relevant_docs list with corrected relevance score
        # max(0, ...) prevents negative scores when distance > 1
        relevant_docs = [
            {
                "source": metadata['source'],
                "relevance_pct": max(0, round((1 - distance) * 100, 1))
            }
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0])
        ]

        # Build context for the LLM
        context = ""
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"\n[Doc {i+1} — {metadata['source']}]\n{doc}\n"

        # Generate diagnosis with Claude
        from src.llm_client import generate_maintenance_report
        diagnosis = generate_maintenance_report(
            sensor_readings={col: float(last_cycle[col].values[0])
                            for col in ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                                       'sensor_9', 'sensor_11', 'sensor_12', 'sensor_14',
                                       'sensor_17', 'sensor_20', 'sensor_21']},
            predicted_rul=predicted_rul,
            unit_id=data.unit_id
        )

    return {
        "unit_id": data.unit_id,
        "cycle": data.history[-1].cycle,
        "predicted_rul": round(predicted_rul, 1),
        "alert_level": alert_level,
        "relevant_docs": relevant_docs,
        "diagnosis": diagnosis
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)