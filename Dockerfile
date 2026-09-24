# NVIDIA CUDA runtime plus the CUDA 12.8 PyTorch wheels used by Laya.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.11.0
ARG LAYA_VERSION=0.3.20

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    LAYA_DEVICE=cuda \
    LAYA_PRELOAD=1 \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    HF_HOME=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# Install PyTorch from the CUDA wheel index first so pip cannot silently select
# a CPU-only build while resolving Laya's torch dependency.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install "laya[serve]==${LAYA_VERSION}" \
    && python -m pip check \
    && python -c 'import torch; assert torch.version.cuda == "12.8", torch.version.cuda'

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home --shell /usr/sbin/nologin laya \
    && mkdir -p /models \
    && chown -R laya:laya /models

# Mount a named volume at /models to keep downloaded Hugging Face checkpoints.
VOLUME ["/models"]

USER laya
WORKDIR /home/laya

EXPOSE 8000

# Set LAYA_API_KEY at run time before publishing the port beyond localhost.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("LAYA_PORT", "8000") + "/health", timeout=4)'

CMD ["laya-serve"]
