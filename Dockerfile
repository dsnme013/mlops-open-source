FROM python:3.9-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frontend
COPY src/ ./src
COPY frontend_code/ ./frontend_code

# Expose Cloud Run port
EXPOSE 8080

# Set environment variables (Cloud Run will also inject $PORT automatically)
ENV PORT=8080

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
