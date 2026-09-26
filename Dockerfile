# Slim Python base: the CUDA 12.8 PyTorch wheels carry their own CUDA
# libraries, and the NVIDIA Container Toolkit injects the driver at run time.
FROM python:3.12-slim-bookworm

# uv installs what pyproject.toml asks for. The versions it resolves are frozen in
# uv.lock, including torch on the CUDA 12.8 index, so nothing here can move them out
# from under the check at the end of the sync below. Keep this tag equal to the one
# CI pins in astral-sh/setup-uv.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/

# LAYA_IDLE_TIMEOUT and LAYA_MAX_LOADED are deliberately absent: their defaults
# live in one place, the code, so the image and a local run cannot drift apart.
# The venv goes on PATH so `docker run ... python` reaches the same interpreter
# the server does.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    LAYA_DEVICE=cuda \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    HF_HOME=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/home/laya/.venv/bin:$PATH"

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home --shell /usr/sbin/nologin laya \
    && mkdir -p /models \
    && chown -R laya:laya /models

# Mount a named volume at /models to keep downloaded Hugging Face checkpoints.
VOLUME ["/models"]

USER laya
WORKDIR /home/laya

# The lockfile on its own layer, so a source change does not re-resolve the wheels.
COPY --chown=laya:laya pyproject.toml uv.lock README.md ./
COPY --chown=laya:laya src ./src

# Sync as laya rather than root, because uv needs a writable environment and cache.
# The assert is what stops a CPU or wrong-CUDA torch from shipping in a server that
# would then answer every request in slow motion instead of failing the build.
RUN uv sync --frozen --no-dev --no-editable \
    && uv run --no-sync python -c 'import torch; assert torch.version.cuda == "12.8", torch.version.cuda'

EXPOSE 8000

# Set LAYA_API_KEY at run time before publishing the port beyond localhost. No
# checkpoint is resident until the first request, so a healthy container means the
# port is answering, not that a model is loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("LAYA_PORT", "8000") + "/health", timeout=4)'

# laya-idle-serve is laya-serve without the preload: checkpoints are downloaded and
# built by the first request, then freed again after LAYA_IDLE_TIMEOUT quiet seconds.
# laya-idle-mcp is the same server as an MCP endpoint on the same port; the Compose
# example runs it as a second container and publishes it on 8001, because the two
# cannot share one process or one GPU allocation.
CMD ["uv", "run", "--no-sync", "--frozen", "laya-idle-serve"]
