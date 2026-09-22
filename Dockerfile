# Production Multi-Stage Container for Voice Integrity Verification Framework
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies for audio wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime audio codecs & processing utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt python wheels/packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy application files
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY models/ /app/models/
COPY test_samples/ /app/test_samples/
COPY main.py /app/main.py
COPY app.py /app/app.py

# Expose HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
