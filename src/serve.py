# src/serve.py
import os
import time
import traceback
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("iris-api")

# ---- DagsHub auth ----
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "iris_model")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "iris_experiment")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "model/model.joblib")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
log.info(f"Tracking URI: {MLFLOW_TRACKING_URI}")

# ---- Feature names ----
SHORT_TO_FULL = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}
REQUIRED = list(SHORT_TO_FULL.values())

# For numeric labels
SPECIES_MAP = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# ---- FastAPI ----
app = FastAPI(title="Iris Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_COUNT = Counter("request_count", "Total requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

# ---- Model loading helpers ----
def try_load_registry(name: str):
    try:
        log.info(f"Loading from registry: models:/{name}/Production")
        return mlflow.pyfunc.load_model(f"models:/{name}/Production")
    except Exception as e:
        log.warning(f"Registry load failed: {e}")
        return None

def try_load_best_run(experiment: str):
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            log.warning(f"Experiment '{experiment}' not found")
            return None
        runs = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=3)
        for r in runs:
            rid = r.info.run_id
            for p in ["model", "random_forest_model"]:
                try:
                    path = f"runs:/{rid}/{p}"
                    log.info(f"Trying {path}")
                    return mlflow.pyfunc.load_model(path)
                except Exception:
                    pass
        return None
    except Exception:
        log.exception("Error while loading from best run")
        return None

def try_load_local(path: str):
    try:
        import joblib
        if not os.path.exists(path):
            return None
        m = joblib.load(path)
        class Wrapper:
            def __init__(self, model): self.m = model
            def predict(self, df): return self.m.predict(df)
        log.info("Loaded local joblib model")
        return Wrapper(m)
    except Exception:
        log.exception("Local model load failed")
        return None

_model = None
def get_model():
    global _model
    if _model is not None:
        return _model
    for loader in (lambda: try_load_registry(MODEL_NAME),
                   lambda: try_load_best_run(EXPERIMENT_NAME),
                   lambda: try_load_local(LOCAL_MODEL_PATH)):
        m = loader()
        if m is not None:
            _model = m
            log.info("Model loaded")
            return _model
    raise RuntimeError("No loadable model found (registry, runs, local all failed)")

# ---- Health ----
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    try:
        _ = get_model()
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

# ---- Optional GET for quick tests ----
@app.get("/predict")
def predict_get(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    return predict_post(payload)

# ---- Main POST endpoint ----
@app.post("/predict")
def predict_post(payload: Dict[str, Any] = Body(...)):
    t0 = time.time()
    status = "200"
    try:
        # Normalize input keys (accept short or full names)
        row = {}
        for k, v in payload.items():
            if v is None:
                continue
            if k in SHORT_TO_FULL:
                row[SHORT_TO_FULL[k]] = float(v)
            elif k in SHORT_TO_FULL.values():
                row[k] = float(v)

        missing = [k for k in REQUIRED if k not in row]
        if missing:
            status = "400"
            return JSONResponse(status_code=400, content={"error": f"missing features: {missing}"})

        df = pd.DataFrame([{k: row[k] for k in REQUIRED}])

        pred = get_model().predict(df)

        # unwrap common container types
        if isinstance(pred, (list, tuple)) and len(pred) > 0:
            pred = pred[0]
        elif isinstance(pred, np.ndarray):
            try:
                pred = pred.item()
            except Exception:
                if pred.size > 0:
                    pred = pred.ravel()[0]

        log.info(f"Prediction raw value: {pred!r} (type={type(pred)})")

        # ---- handle string labels from the model ----
        if isinstance(pred, (str, np.str_)):
            label = str(pred).strip()
            aliases = {
                "setosa": "Iris-setosa",
                "versicolor": "Iris-versicolor",
                "virginica": "Iris-virginica",
            }
            pretty = aliases.get(label.lower(), label)
            return {"prediction": label, "species": pretty}

        # Fallback: numeric labels 0/1/2
        try:
            idx = int(pred)
            species = SPECIES_MAP.get(idx, "Unknown")
            return {"prediction": idx, "species": species}
        except Exception:
            # last resort: do not crash; return as string
            return {"prediction": str(pred), "species": str(pred)}

    except Exception as e:
        traceback.print_exc()
        status = "500"
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {e}"})
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - t0)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status=status).inc()

# ---- Frontend (optional) ----
@app.get("/ui")
def serve_index():
    path = "frontend_code/index.html"
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "UI not bundled in container"}

if os.path.exists("frontend_code"):
    app.mount("/static", StaticFiles(directory="frontend_code"), name="static")

# ---- Metrics ----
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
