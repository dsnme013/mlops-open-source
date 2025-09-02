# src/serve.py  (only the relevant parts shown)

import os
import mlflow
import pandas as pd
from fastapi import FastAPI, Response
from typing import Dict, Any
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

REQUEST_COUNT = Counter("request_count", "Total requests", ["endpoint","method","status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "iris_model")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---- load model exactly as you prefer (registry Production or fallback) ----
# Example (registry only):
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

# Example (registry, then fallback to best run's 'model' or 'random_forest_model'):
from mlflow.tracking import MlflowClient
def load_model():
    try:
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    except Exception:
        client = MlflowClient()
        exp = client.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT","iris_experiment"))
        run = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)[0]
        run_id = run.info.run_id
        for p in ["model","random_forest_model"]:
            try:
                return mlflow.pyfunc.load_model(f"runs:/{run_id}/{p}")
            except Exception:
                pass
        raise RuntimeError("No loadable model found")
model = load_model()

# Map snake_case -> training names
NAME_MAP = {
    "sepal_length": "sepal length (cm)",
    "sepal_width":  "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width":  "petal width (cm)",
}
REQUIRED = list(NAME_MAP.values())

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    import time, numpy as np, traceback
    t0 = time.time()
    try:
        row: Dict[str, float] = {}

        # accept original names
        for k, v in payload.items():
            if k in REQUIRED:
                row[k] = float(v)

        # accept snake_case and remap
        for k, v in payload.items():
            if k in NAME_MAP:
                row[NAME_MAP[k]] = float(v)

        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"missing features: {missing}")

        df = pd.DataFrame([{k: row[k] for k in REQUIRED}])
        pred = model.predict(df)
        if isinstance(pred, (list, tuple, np.ndarray)):
            pred = pred[0]
        return {"prediction": str(pred)}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"prediction failed: {e}"}
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - t0)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
