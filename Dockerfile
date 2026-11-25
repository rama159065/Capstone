# One image for the FastAPI backend and the Streamlit UI (docker-compose.yml runs it twice).
# CPU only: the hosted LLM does the generation, and retrieval models (MiniLM, the reranker) run on CPU.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_PROFILE=server \
    HF_HOME=/app/indexes/hf_cache \
    HOME=/tmp/app-home

# libgl1 / libglib2.0-0: OpenCV inside RapidOCR. curl: health checks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml deploy/requirements_from_pyproject.py /tmp/build/
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.9.0" "torchvision>=0.21.0" \
 && python /tmp/build/requirements_from_pyproject.py /tmp/build/pyproject.toml > /tmp/build/requirements.txt \
 && pip install -r /tmp/build/requirements.txt \
 && rm -rf /tmp/build

# Code and the small committed data (manifests, eval set, corpus plan). PDFs, indexes and outputs are
# excluded by .dockerignore and mounted at run time.
COPY scripts ./scripts
COPY configs ./configs
COPY deploy ./deploy
COPY data ./data
COPY .streamlit ./.streamlit
COPY pyproject.toml ./

# Runs as any uid (compose passes the host user so bind-mounted indexes stay writable).
RUN mkdir -p /app/indexes /tmp/app-home && chmod -R a+rwX /app/indexes /tmp/app-home

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["uvicorn", "scripts.api.api_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
