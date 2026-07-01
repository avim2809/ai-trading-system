# syntax=docker/dockerfile:1
#
# Single image that serves BOTH the FastAPI backend and the built React frontend
# (FastAPI mounts frontend/dist as an SPA). No GPU required — LLM calls go out
# over the network. State (Chroma vectordb, DuckDB/SQLite caches, Parquet cache,
# run artifacts) lives on the /app/data and /app/runs volumes.

############################################################
# Stage 1 — build the React/Vite frontend -> frontend/dist
############################################################
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

############################################################
# Stage 2 — build a Python venv with all runtime deps
############################################################
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
# Install the project's third-party deps via its extras (api + llm + live).
# RAG embeddings/reranking use the hosted Voyage API (no torch), so the heavy
# `local` extra is intentionally omitted — keeps the image small + RAM low.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --upgrade pip \
 && pip install ".[api,llm,live]"

############################################################
# Stage 3 — slim runtime
############################################################
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src
# libgomp1: OpenMP runtime that scikit-learn / hmmlearn load at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Run from source (PYTHONPATH=/app/src) so app.py resolves frontend/dist
# relative to the repo layout it expects.
COPY src/ ./src/
COPY config/ ./config/
COPY --from=frontend /app/frontend/dist ./frontend/dist
# Persisted state — mount volumes here so data survives container restarts.
RUN mkdir -p /app/data /app/runs
VOLUME ["/app/data", "/app/runs"]
EXPOSE 8000
# ${PORT:-8000}: the same image works on a fixed-port VPS and on PaaS hosts
# (Railway/Render) that inject $PORT.
CMD ["sh", "-c", "uvicorn firm.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
