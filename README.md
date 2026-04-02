# PredictAI Industrial

Predictive maintenance system for industrial equipment — ML + LLM + MLOps deployed in production.

## Live Demo

- **API** : https://predictai-industrial.onrender.com/docs
- **Health check** : https://predictai-industrial.onrender.com/health

## What it does

Predicts the Remaining Useful Life (RUL) of industrial turbofan engines from sensor data.
Send sensor readings → receive RUL prediction + alert level (NORMAL / WARNING / CRITICAL).

## Results

| Model | RMSE | Trees |
|-------|------|-------|
| Random Forest (full) | 20.05 cycles | 100 |
| Random Forest (production) | 20.96 cycles | 20 |

State of the art on CMAPSS FD001 : ~12 cycles (LSTM bidirectional).

## Dataset

NASA CMAPSS — Turbofan Engine Degradation Simulation
- 100 engines, 20,631 data points, 21 sensors
- 11 useful sensors identified (10 constant sensors removed)
- 48 features engineered : raw sensors + rolling mean/std/diff (window=10)

## Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn Random Forest |
| Experiment tracking | MLflow |
| API | FastAPI + Pydantic |
| LLM Diagnosis | Anthropic Claude API |
| Deployment | Render.com |
| Language | Python 3.13 |

## Quick Start
```bash
git clone https://github.com/isidmohand-netizen/predictai-industrial.git
cd predictai-industrial
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd api && uvicorn main:app --reload --port 8000
```

## Project Structure
```
predictai-industrial/
├── data/raw/          # NASA CMAPSS datasets
├── notebooks/         # Exploration and training
├── src/
│   ├── features.py    # Feature engineering pipeline
│   └── llm_client.py  # Anthropic API client
├── api/
│   └── main.py        # FastAPI endpoints
├── models/            # Trained model (pickle)
└── requirements.txt
```


## Next steps

The current production model is a lightweight Random Forest (20 trees, RMSE 20.96 cycles) optimized for deployment size and low latency. Planned improvements:

- **LSTM bidirectional** — expected RMSE ~12 cycles (state of the art on CMAPSS FD001)
- **XGBoost** — gradient boosting for better accuracy on tabular data
- **Unit tests** — pytest coverage on features.py and API endpoints
- **GitHub Actions CI** — automated testing and Docker build on every push
- **Generalization** — train and evaluate on all 4 CMAPSS datasets (FD001-FD004)
- **Monitoring** — Prometheus + Grafana dashboard for production metrics
## Author
Idir Sid Mohand — Data Scientist