# Dockerfile for RAG API
# ----------------------
# Build:  docker build -t rag-api -f deployment/Dockerfile .
# Run:    docker run -p 8000:8000 rag-api

FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY config.yaml .
COPY main.py .
COPY api.py .

# Create data directories
RUN mkdir -p data/vectordb data/sample_docs

# Copy sample documents
COPY data/sample_docs/ ./data/sample_docs/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
