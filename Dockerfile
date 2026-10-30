# Slim production image (Render free tier: 512 MB RAM). No PyTorch: the ticket classifiers run as int8
# ONNX on onnxruntime, like the embedding models. Also used by docker-compose for local runs.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed MALLOC_ARENA_MAX=2
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install . && pip cache purge || true

# Bake the embedding models (bge-small int8 ~67 MB + BM25) into the image: a cold start on the free tier
# shouldn't download them again.
RUN python -c "from supportpilot.retrieval import embeddings as e; e.embed_query('x'); e.sparse_query('x')"

# Ticket classifiers: int8 ONNX export (scripts/export_onnx.py), published as a GitHub release asset.
ARG CLASSIFIER_URL=https://github.com/abhijeetGithu/SupportPilot-Agentic-RAG-Customer-Support-Copilot/releases/download/models-v1/classifiers-onnx-int8.zip
COPY scripts ./scripts
RUN python scripts/fetch_classifiers.py "$CLASSIFIER_URL"

EXPOSE 8000
# Render sets $PORT; one worker keeps memory predictable
CMD ["sh", "-c", "uvicorn supportpilot.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
