FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src
EXPOSE 8000
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
