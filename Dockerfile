FROM python:3.9-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src and web folder into the working directory
COPY src/ ./src
COPY web/ ./web

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:8080
ENV PORT=8080

# Run the app using uvicorn
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "${PORT}"]
