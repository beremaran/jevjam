# causal-conv1d ships no wheel for torch 2.11, so build one here, where nvcc exists.
# Keep the torch pin equal to pyproject.toml's, and the Python equal to the final stage's.
# Ubuntu 22.04's gcc 11 builds code the final stage's older libstdc++ (bookworm) can load.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS conv1d
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
RUN uv venv --python 3.12 /build \
    && uv pip install --python /build/bin/python --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
        torch==2.11.0+cu128 setuptools wheel packaging ninja pip \
    && /build/bin/python -m pip wheel --no-build-isolation --no-deps -w /wheels causal-conv1d==1.7.0

# Slim Python base: the CUDA 12.8 PyTorch wheels carry their own CUDA
# libraries, and the NVIDIA Container Toolkit injects the driver at run time.
FROM python:3.12-slim-bookworm

# uv installs what pyproject.toml asks for. The versions it resolves are frozen in
# uv.lock, including torch on the CUDA 12.8 index, so nothing here can move them out
# from under the check at the end of the sync below. Keep this tag equal to the one
# CI pins in astral-sh/setup-uv.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/

# No JEVJAM_* setting is set here: their defaults live in one place, the code, so
# the image and a local run cannot drift apart, and an old LAYA_* value passed at
# run time is not shadowed by an image default.
# The venv goes on PATH so `docker run ... python` reaches the same interpreter
# the server does.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/home/jevjam/.venv/bin:$PATH"

# Triton (used by flash-linear-attention) compiles a C launcher at run time, so it needs gcc.
# uv needs git to fetch Julia's code from its Hugging Face repo. Without git-lfs it
# fetches only pointers for the weights, which the server downloads on first use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 jevjam \
    && useradd --uid 10001 --gid jevjam --create-home --shell /usr/sbin/nologin jevjam \
    && mkdir -p /models \
    && chown -R jevjam:jevjam /models

# Mount a named volume at /models to keep downloaded Hugging Face checkpoints.
VOLUME ["/models"]

USER jevjam
WORKDIR /home/jevjam

# The lockfile on its own layer, so a source change does not re-resolve the wheels.
COPY --chown=jevjam:jevjam pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=jevjam:jevjam src ./src

# Sync as jevjam rather than root, because uv needs a writable environment and cache.
# The assert is what stops a CPU or wrong-CUDA torch from shipping in a server that
# would then answer every request in slow motion instead of failing the build.
RUN --mount=type=bind,from=conv1d,source=/wheels,target=/tmp/wheels \
    uv sync --frozen --no-dev --no-editable \
    && uv pip install --no-deps /tmp/wheels/*.whl \
    && uv run --no-sync python -c 'import torch; from causal_conv1d import causal_conv1d_fn; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; assert torch.version.cuda == "12.8", torch.version.cuda'

EXPOSE 8000

# Set JEVJAM_API_KEY at run time before publishing the port beyond localhost. No
# checkpoint is resident until the first request, so health means both endpoints
# answer, not that a model is loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c 'import json, os, urllib.request; response = urllib.request.urlopen("http://127.0.0.1:" + (os.environ.get("JEVJAM_PORT") or os.environ.get("LAYA_PORT") or "8000") + "/health", timeout=4); assert json.load(response)["mcp_ready"] is True'

# jevjam is laya-serve without the preload, plus Julia: checkpoints are downloaded
# and built by the first request, then freed after JEVJAM_IDLE_TIMEOUT seconds without
# an inference request on either the Jev-compatible HTTP API or the MCP endpoint.
CMD ["uv", "run", "--no-sync", "--frozen", "jevjam"]
