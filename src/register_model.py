import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    # DagsHub auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "iris_experiment")
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "iris_model")

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found to register")

    run = runs[0]
    acc = run.data.metrics.get("accuracy", 0.0)
    run_id = run.info.run_id
    print(f"Best run: {run_id}  accuracy={acc:.4f}")

    if acc < args.threshold:
        print(f"Accuracy {acc:.4f} < threshold {args.threshold:.4f}. Skipping registration.")
        return

    model_uri = f"runs:/{run_id}/model"
    rm = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=rm.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Registered {model_name} v{rm.version} → Production")

if __name__ == "__main__":
    main()
