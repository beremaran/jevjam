# Configuration

How to run, configure and tune the jevjam container. Back to the [README](../README.md).

## Running

Compose (`docker compose up -d`) and `docker run` both work; the
[README](../README.md#quick-start) has the commands. To build the image yourself, run
`docker compose up --build -d` or `docker build -t jevjam:latest .`.

The port opens in about a second, and no checkpoint is resident yet, so `/health`
answers straight away:

```json
{"status": "ok", "loaded": [], "device": "auto", "mcp_ready": true}
```

An empty `loaded` is the cold state, not a fault. The first inference request builds
the checkpoint it routes to, which on a volume that has never been populated means
downloading it first and takes minutes. Follow it with `docker compose logs -f jevjam`,
and give the client a timeout that allows for it.

The MCP endpoint is available in the same container at
`http://127.0.0.1:8000/mcp`. The API and MCP tools share one router, so a request to
either endpoint can keep the same checkpoint warm.

Stop and start without redownloading:

```bash
docker compose down       # stop; the model volume survives
docker compose up -d      # start again
docker compose down -v    # stop and delete the downloaded checkpoints
```

Check GPU access without downloading any checkpoint:

```bash
docker run --rm --gpus all ghcr.io/beremaran/jevjam:latest python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

## Inside the image

- Installs `laya[serve,mcp]` (0.3.20), Julia-1's code pinned to one commit of its model repo, clef-flash's code copied from one commit of its model repo, and CUDA 12.8 PyTorch wheels on `python:3.12-slim-bookworm`, through uv and a committed `uv.lock`.
- Runs the HTTP API and MCP endpoint in one process, as non-root UID 10001, listening on `0.0.0.0:8000`.
- Serves MCP over streamable HTTP at `/mcp`; both endpoints share one router and one copy of each loaded model in VRAM.
- Downloads no checkpoint at boot. The first inference builds the one it routes to, and every checkpoint is freed after `JEVJAM_IDLE_TIMEOUT` seconds without inference on either endpoint.
- Caches downloaded checkpoints in `/models` (`HF_HOME`), designed to sit on a Docker volume.
- Ships a `/health` healthcheck that reports whether both endpoints are ready and which checkpoints are resident.

Verified on an RTX 4070 Ti SUPER (driver 615.71.09) with `torch 2.11.0+cu128`. There,
a warm request answers in 31 ms, sleeping hands back about 2.5 GB of the 16 GB card, and
the request that wakes it costs 0.6 s to rebuild the checkpoint.

## Model cache

Checkpoints go to `/models` inside the container. Keep it on a named volume (Compose and the README do)
and the download happens once; delete the volume to force a fresh download. Pass
`-e HF_TOKEN=...` if you have a Hugging Face token, or `-e HF_HUB_OFFLINE=1` to refuse
network access and use only what is cached.

## Settings

All configuration is by environment variable. Defaults shown are what the image runs with.

Every `JEVJAM_*` setting below used to be named `LAYA_*`, for example `LAYA_API_KEY`.
The old name still works and logs a deprecation warning at startup; when both are
set, the `JEVJAM_*` one wins.

| Variable | Default | Meaning |
| --- | --- | --- |
| `JEVJAM_HOST` | `0.0.0.0` | Bind address inside the container. |
| `JEVJAM_PORT` | `8000` | Bind port. Compose publishes this same value on the host. |
| `JEVJAM_DEVICE` | auto | Torch device for every model. Unset uses CUDA when there is a GPU, else CPU. `cuda` falls back to CPU with a warning if CUDA is unavailable. |
| `JEVJAM_IDLE_TIMEOUT` | `300` | Seconds without an inference request before every checkpoint is freed. `0` keeps them resident. Must be a whole number; anything else stops the server at startup. |
| `JEVJAM_CLEF_QUANT` | `auto` | How clef-flash loads: `auto` picks the first that fits of `bf16`, `8bit`, `4bit` and `offload` (split across GPU and CPU). Naming one forces it. |
| `LAYA_AUTO_TASK` | `0` | `1` lets Laya's router reach `typed-decisions` automatically. Laya-only, so it keeps its name. |
| `JEVJAM_THREADS` | torch default | Caps torch intra-op threads for CPU inference, for every model; keep at or below physical cores. |
| `JEVJAM_API_KEY` | unset | When set, requests to `/v1/models`, `/v1/systemone` and `/mcp` must send `Authorization: Bearer <key>`. `GET /health` remains public. |
| `JEVJAM_LOG_LEVEL` | `info` | Uvicorn log level, and the level of this server's own log lines. |
| `HF_HOME` | `/models` | Checkpoint cache location. Mount a volume here. |
| `HF_TOKEN` | unset | Optional Hugging Face credential. |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints. |

`LAYA_PRELOAD` and `LAYA_MODELS` used to choose what to load at boot, and
`JEVJAM_MAX_LOADED` (once `LAYA_MAX_LOADED`) how many checkpoints could stay resident.
They no longer do anything, because there is no boot load and one checkpoint is
resident at a time; setting any of them logs a warning and is otherwise ignored.

For example, to keep the checkpoints resident for a server under steady load, and to
require a key:

```bash
JEVJAM_IDLE_TIMEOUT=0 JEVJAM_API_KEY=secret docker compose up -d
```

## Sleeping on idle

The server never preloads. The first inference request builds the checkpoint it routes
to, and `JEVJAM_IDLE_TIMEOUT` seconds after the last inference on either endpoint,
every checkpoint is freed: the models go, the garbage collector runs, and the CUDA
caching allocator hands its blocks back to the driver. A server under load stays warm;
a server nobody calls stops holding GPU memory. `GET /health` reports resident models
in `loaded` and whether the MCP session manager is ready in `mcp_ready`.

One checkpoint is resident at a time. A request for any other checkpoint, in any
family, frees the resident one first and then builds its own. Every request, from
the HTTP API or MCP, waits its turn in one queue, so a load never overlaps another
request's inference.

Waking costs one checkpoint build, not three, because a request loads only what it
routes to. Measured on an RTX 4070 Ti SUPER from a warm volume, that build is 0.6 s, and
a warm request that does not need one answers in 31 ms. Measure it on your own GPU before
setting a client timeout; laya's published numbers are 7.4 s on CPU and 10.3 s on a T4.

```bash
docker compose exec jevjam uv run --no-sync python -c \
  'import time; from laya import Router; r = Router(device="cuda"); r.load("english"); \
   t = time.perf_counter(); r.unload(); r.load("english"); print("%.1fs" % (time.perf_counter() - t))'
```

Two other things worth knowing. Switching checkpoints pays a rebuild each time, so a
client that alternates between models, or between Laya's languages, pays it on every
request; clef-flash's rebuild takes far longer than the others'. And freeing the checkpoints does not release the CUDA context or the memory torch itself
holds: only letting the container stop does that, and nothing inside the image can
wake it up again on the next request.
