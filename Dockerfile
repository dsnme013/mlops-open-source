# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src

# Cloud Run listens on $PORT. Default to 8080 if not set.
ENV PORT=8080

# (Optional, but fine to keep)
EXPOSE 8080

# Don't hardcode MLFLOW_TRACKING_URI here. We'll pass it at deploy time.
CMD ["sh", "-c", "uvicorn src.serve:app --host 0.0.0.0 --port ${PORT}"]