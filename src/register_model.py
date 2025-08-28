# src/register_model.py
import os
from mlflow.tracking import MlflowClient

def main(metric_threshold=0.9):
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    exp = client.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT", "iris_experiment"))
    if exp is None:
        print("No experiment found.")
        return

    runs = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=10)
    if not runs:
        print("No runs found.")
        return

    best = runs[0]
    acc = best.data.metrics.get("accuracy", 0)
    run_id = best.info.run_id
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "iris_model")

    print(f"Best run {run_id} acc={acc:.4f} (threshold={metric_threshold})")
    if acc >= metric_threshold:
        try:
            client.create_registered_model(model_name)
            print("Registered model created.")
        except Exception:
            pass

        mv = client.create_model_version(name=model_name, source=f"runs:/{run_id}/random_forest_model", run_id=run_id)
        client.transition_model_version_stage(name=model_name, version=mv.version, stage="Production", archive_existing_versions=True)
        print(f"Model version {mv.version} registered and promoted to Production.")
    else:
        print("Metric below threshold — skipping registration.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    main(args.threshold)
