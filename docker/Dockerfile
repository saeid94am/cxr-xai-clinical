# ─────────────────────────────────────────────────────────────────────────────
#  CXR-XAI-Clinical — Dockerfile
#
#  Two build targets:
#    train  — CUDA 12.1 + PyTorch for Kaggle-equivalent GPU training
#    demo   — CPU-only, lighter image for the Streamlit demo
#
#  Build examples:
#    docker build --target train -t cxr-xai:train .
#    docker build --target demo  -t cxr-xai:demo  .
# ─────────────────────────────────────────────────────────────────────────────

# ── Base: CUDA runtime (shared by train target) ───────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS cuda-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git curl \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt


# ── Train target (GPU) ────────────────────────────────────────────────────────
FROM cuda-base AS train

COPY . .

# Bind-mount data/ and results/ at runtime; they are too large to bake in.
VOLUME ["/workspace/data", "/workspace/results"]

ENV WANDB_DIR=/workspace/wandb

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/train.yaml", "--model", "densenet121"]


# ── CPU base (demo and CI) ────────────────────────────────────────────────────
FROM python:3.11-slim AS cpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt


# ── Demo target (CPU + Streamlit) ────────────────────────────────────────────
FROM cpu-base AS demo

COPY . .

VOLUME ["/workspace/data", "/workspace/results"]

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "demo/app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
