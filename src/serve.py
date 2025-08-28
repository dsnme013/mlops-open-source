# src/serve.py
import os
import mlflow
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

REQUEST_COUNT = Counter("request_count", "Total number of requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "iris_model")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        print("Loaded model from registry: Production")
        return model
    except Exception as e:
        print("Could not load from registry, trying latest run model:", e)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT", "iris_experiment"))
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            return mlflow.pyfunc.load_model(f"runs:/{run_id}/random_forest_model")
        raise RuntimeError("No model available")

model = load_model()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(payload: IrisData):
    import time
    t0 = time.time()
    df = pd.DataFrame([payload.dict().values()], columns=payload.dict().keys())
    try:
        pred = model.predict(df)[0]
        status = "200"
        return {"prediction": pred}
    finally:
        latency = time.time() - t0
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status=status).inc()

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
