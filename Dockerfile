FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY frontend_code/ ./frontend_code

ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""
ENV MLFLOW_EXPERIMENT="iris_experiment"
ENV MLFLOW_MODEL_NAME="iris_model"
ENV PORT=8080

EXPOSE 8080
CMD exec uvicorn src.serve:app --host 0.0.0.0 --port ${PORT}

