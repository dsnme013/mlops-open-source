.PHONY: setup make_dataset dvc_add start_mlflow train evaluate register serve monitor docker_build docker_run

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

make_dataset:
	python src/make_dataset.py

dvc_add:
	dvc init || true
	dvc add data/iris.csv
	git add data/iris.csv.dvc .gitignore
	git commit -m "Add dataset with DVC"

start_mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

train:
	python src/train.py --n-estimators 100

evaluate:
	python src/evaluate.py

register:
	python src/register_model.py --threshold 0.90

serve:
	uvicorn src.serve:app --reload

monitor:
	python src/monitor_evidently.py

docker_build:
	docker build -t iris-api .

docker_run:
	docker run -p 8000:8000 --env MLFLOW_TRACKING_URI=http://host.docker.internal:5000 iris-api
