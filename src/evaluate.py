# src/evaluate.py
import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import classification_report

def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    exp = client.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT", "iris_experiment"))
    if exp is None:
        print("No experiment found.")
        return

    runs = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)
    if not runs:
        print("No runs found.")
        return

    best = runs[0]
    print(f"Best run_id={best.info.run_id} accuracy={best.data.metrics.get('accuracy')}")
    model_uri = f"runs:/{best.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv("data/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == "__main__":
    main()
