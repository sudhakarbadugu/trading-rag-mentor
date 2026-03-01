# ═══════════════════════════════════════════════════════════════════════════════
# Trading RAG Mentor — Dockerfile
# Multi-stage build: lean, secure, production-ready
#
# Build:  docker build -t trading-rag-mentor .
# Run:    docker run -p 8501:8501 --env-file .env trading-rag-mentor
# Compose: docker compose up
#
# Image size note: torch + sentence-transformers + spaCy make this image large
# (~5-7 GB). This is expected for local ML inference. Consider GPU-enabled
# hosting (e.g. Render Pro, Railway GPU) for faster inference at scale.
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Install all Python dependencies with full build toolchain available.
# The compiled artefacts are then copied to the lean runtime stage.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
LABEL stage="builder"

# System build deps — removed in the runtime stage for a smaller attack surface
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── Layer cache: only re-install if requirements.txt changes ─────────────────
COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly to the runtime image.
# --no-cache-dir keeps the layer tight.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# Lean final image: only runtime libs, no compilers, no build artefacts.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# ── Runtime system dependencies ───────────────────────────────────────────────
# jq        : required by JSONLoader in build_index.py
# libgomp1  : OpenMP runtime (sentence-transformers / scikit-learn / torch)
# libstdc++6: C++ runtime for compiled ML extensions (onnxruntime, chromadb)
# libmagic1 : file-type detection (unstructured library)
# poppler-utils: PDF page rendering (pdf2image / unstructured)
# tesseract-ocr: OCR fallback (unstructured_inference)
# curl      : used by the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq \
        libgomp1 \
        libstdc++6 \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# ── Security: non-root user ───────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 --gid appgroup \
                --shell /bin/bash \
                --create-home appuser

WORKDIR /app

# ── Application source ────────────────────────────────────────────────────────
# Copy only the source — data/ is provided via Docker volumes at runtime
# so secrets and local DB files never bake into the image.
COPY --chown=appuser:appgroup src/         ./src/
COPY --chown=appuser:appgroup pyproject.toml ./

# Create runtime data directories.
# chroma_db/ and chat_history.db live here and are persisted via named volumes.
# transcripts/ is bind-mounted read-only from the host (see docker-compose.yml).
RUN mkdir -p data/transcripts data/chroma_db \
    && chown -R appuser:appgroup /app

# Switch to the non-root user for all subsequent steps
USER appuser

# ── Download spaCy model (en_core_web_sm) ────────────────────────────────────
# spaCy models are not on PyPI; they must be fetched via `spacy download`.
# The model is stored inside the image so runtime stays fully offline.
RUN python -m spacy download en_core_web_sm

# ── Pre-cache HuggingFace models inside the image ────────────────────────────
# Baking the models in (~90 MB embedding + ~85 MB cross-encoder) eliminates
# the cold-start download on every new container deploy.
#
# REQUIRES internet access during `docker build`.
# Remove both RUN blocks below if you prefer a smaller image and can tolerate
# a ~1-minute download on first container start (also remove TRANSFORMERS_OFFLINE
# and HF_DATASETS_OFFLINE env vars further down).
ENV HF_HOME=/home/appuser/.cache/huggingface

RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
print('✅  Embedding model cached.')"

RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('✅  Cross-encoder model cached.')"

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Streamlit: disable telemetry, browser pop-ups, and hot-reload watcher
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    # Run HuggingFace fully offline using the models baked into the image
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# Default port — overridden at runtime by Render/Railway via $PORT
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# start-period is generous to allow the index-build step on first boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Shell form is required so ${PORT:-8501} is expanded at container start time.
# Render and Railway inject $PORT automatically; locally it falls back to 8501.
CMD streamlit run src/app.py \
        --server.port            "${PORT:-8501}" \
        --server.address         0.0.0.0 \
        --server.headless        true \
        --server.fileWatcherType none \
        --server.enableCORS      false \
        --server.enableXsrfProtection false
