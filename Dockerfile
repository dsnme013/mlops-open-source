FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frontend
COPY src/ ./src
COPY frontend_code/ ./frontend_code

# Expose Cloud Run port
EXPOSE 8080
ENV PORT=8080

# Run the app with uvicorn (Cloud Run provides $PORT automatically)
CMD exec uvicorn src.serve:app --host 0.0.0.0 --port ${PORT}

