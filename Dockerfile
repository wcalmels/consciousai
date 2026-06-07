# ConsciousAI v3.0 - Dockerfile
FROM python:3.10-slim AS base

LABEL maintainer="Walter Calmels <walter@tuch.systems>"
LABEL version="3.0.0"
LABEL description="ConsciousAI - Integrated Information Theory for Autonomous Systems"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/

# Create data directories
RUN mkdir -p /data/blockchain /data/manifests /data/logs

# Non-root user
RUN useradd -m -u 1000 consciousai && chown -R consciousai:consciousai /app /data
USER consciousai

ENV PYTHONPATH=/app
ENV CONSCIOUSAI_BLOCKCHAIN_PATH=/data/blockchain/chain.json
ENV CONSCIOUSAI_MANIFEST_PATH=/data/manifests/manifest.json

EXPOSE 8000

CMD ["python", "-m", "src.core.engine"]

# ── Dev stage ──────────────────────────────────────────────
FROM base AS dev

USER root
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY tests/ ./tests/
USER consciousai

CMD ["python", "-m", "pytest", "tests/", "-v"]
