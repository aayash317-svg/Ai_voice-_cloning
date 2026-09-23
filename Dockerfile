# Production Multi-Stage Container for Voice Integrity Verification Framework
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies for audio wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pre-install CPU-only PyTorch (reduces build image by 2.5 GB & prevents cloud OOM/timeouts)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining framework dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime audio codecs & processing utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Pre-create runtime directories with full permissions
RUN mkdir -p /app/logs /app/results /app/features /app/experiments

# Copy application files
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY models/ /app/models/
COPY test_samples/ /app/test_samples/
COPY logs/ /app/logs/
COPY main.py /app/main.py
COPY app.py /app/app.py

# Expose HTTP ports (Standard 8000, Render 10000, Hugging Face 7860)
EXPOSE 8000 10000 7860

# Dynamic health check probing active cloud port
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD sh -c "curl -f http://127.0.0.1:${PORT:-8000}/health || exit 1"

# Production startup dynamically binding to cloud-assigned $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

