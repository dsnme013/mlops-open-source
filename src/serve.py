import os
import mlflow
import pandas as pd
import numpy as np
import traceback
import time

from fastapi import FastAPI, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
from mlflow.tracking import MlflowClient

# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Metrics
# ---------------------------
REQUEST_COUNT = Counter("request_count", "Total requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

# ---------------------------
# MLflow config
# ---------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "iris_model")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    try:
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    except Exception:
        client = MlflowClient()
        exp = client.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT", "iris_experiment"))
        run = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)[0]
        run_id = run.info.run_id
        for p in ["model", "random_forest_model"]:
            try:
                return mlflow.pyfunc.load_model(f"runs:/{run_id}/{p}")
            except Exception:
                pass
        raise RuntimeError("No loadable model found")

# Lazy load
model = None
def get_model():
    global model
    if model is None:
        model = load_model()
    return model

# ---------------------------
# Feature + Species mapping
# ---------------------------
NAME_MAP = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}
REQUIRED = list(NAME_MAP.values())

SPECIES_MAP = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# ---------------------------
# Root + Health Endpoints
# ---------------------------

@app.get("/")
def root():
    """Cloud Run health check root"""
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy"}

# ---------------------------
# Frontend serving
# ---------------------------
@app.get("/ui")
def serve_index():
    path = "frontend_code/index.html"
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "UI not bundled in container"}

# Serve static assets
if os.path.exists("frontend_code"):
    app.mount("/static", StaticFiles(directory="frontend_code"), name="static")

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
async def predict(payload: Dict[str, Any] = Body(...)):
    t0 = time.time()
    status = "200"
    try:
        row = {}
        for k, v in payload.items():
            if k in REQUIRED:
                row[k] = float(v)
        for k, v in payload.items():
            if k in NAME_MAP:
                row[NAME_MAP[k]] = float(v)

        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"missing features: {missing}")

        df = pd.DataFrame([{k: row[k] for k in REQUIRED}])
        pred = get_model().predict(df)
        if isinstance(pred, (list, tuple, np.ndarray)):
            pred = pred[0]

        species = SPECIES_MAP.get(int(pred), "Unknown")
        return {"prediction": int(pred), "species": species}

    except Exception as e:
        traceback.print_exc()
        status = "500"
        return {"error": f"Prediction failed: {e}"}
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - t0)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status=status).inc()

# ---------------------------
# Metrics endpoint
# ---------------------------
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
