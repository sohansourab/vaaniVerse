# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgomp1 is required by torch/faiss CPU builds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    EMBEDDING_DEVICE=cpu \
    HOST=0.0.0.0

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Non-root user; /app/data is the mount point for the persisted FAISS index.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/documents /app/data/vectors \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The platform may inject PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
