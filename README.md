# jevjam: Jev-compatible decision models over HTTP and MCP

jevjam builds and runs small typed-decision models as a GPU-backed HTTP server that
answers on the same `POST /v1/systemone` protocol as TypeSafe Jev. A client written
against Jev only needs its base URL changed; the request and response shapes already
match. The same models are also served as [MCP](https://modelcontextprotocol.io) tools
for agents that would rather call tools than HTTP.

It serves two model families:

- [Laya](https://huggingface.co/convaiinnovations/laya), the default. Its router picks
  one of three checkpoints (`english`, `multilingual`, `typed-decisions`) per request.
- [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) by Supersonic Labs, a 144M
  multilingual decision model. It answers only requests that name it; see
  [Julia-1](#julia-1).

This repository used to be `laya-docker`. The old image,
`ghcr.io/beremaran/laya-docker`, no longer gets updates. The old `LAYA_*` settings
still work; see [Configuration](#configuration).

What the image does:

- Installs `laya[serve,mcp]` (0.3.20), Julia-1's code pinned to one commit of its model repo, and CUDA 12.8 PyTorch wheels on `python:3.12-slim-bookworm`, through uv and a committed `uv.lock`.
- Runs the HTTP API and MCP endpoint in one process, as non-root UID 10001, listening on `0.0.0.0:8000`.
- Serves MCP over streamable HTTP at `/mcp`; both endpoints share one router and one copy of each loaded model in VRAM.
- Downloads no checkpoint at boot. The first inference builds the one it routes to, and every checkpoint is freed after `JEVJAM_IDLE_TIMEOUT` seconds without inference on either endpoint.
- Caches downloaded checkpoints in `/models` (`HF_HOME`), designed to sit on a Docker volume.
- Ships a `/health` healthcheck that reports whether both endpoints are ready and which checkpoints are resident.

## Requirements

- Docker Engine with Compose v2 (or plain `docker run`).
- An NVIDIA GPU with a working driver, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so containers can access it.
- Disk and VRAM for the checkpoints you use. Laya's three are ~1.2B parameters total,
  Julia-1 is 144M (550 MB), and the server holds at most `JEVJAM_MAX_LOADED` of them at a time.

Verified on an RTX 4070 Ti SUPER (driver 615.71.09) with `torch 2.11.0+cu128`. There,
a warm request answers in 31 ms, sleeping hands back about 2.5 GB of the 16 GB card, and
the request that wakes it costs 0.6 s to rebuild the checkpoint.

## Quick start (Compose)

```bash
docker compose up --build -d
docker compose logs -f jevjam
curl -s http://127.0.0.1:8000/health
```

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

## Quick start (Docker CLI)

```bash
docker build -t jevjam:latest .

docker run -d --name jevjam --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v jevjam-models:/models \
  jevjam:latest
```

If you would rather not build locally, CI publishes the image to GitHub Container
Registry on every push to `main`:

```bash
docker pull ghcr.io/beremaran/jevjam:latest
docker run -d --name jevjam --gpus all -p 127.0.0.1:8000:8000 \
  -v jevjam-models:/models ghcr.io/beremaran/jevjam:latest
```

The HTTP API and MCP endpoint both work from that container; use the same host port
and the MCP path `/mcp`.

Check GPU access without downloading any checkpoint:

```bash
docker run --rm --gpus all jevjam:latest python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

## Model cache

Checkpoints go to `/models` inside the container. Keep it on a named volume (as above)
and the download happens once; delete the volume to force a fresh download. Pass
`-e HF_TOKEN=...` if you have a Hugging Face token, or `-e HF_HUB_OFFLINE=1` to refuse
network access and use only what is cached.

## Configuration

All configuration is by environment variable. Defaults shown are what the image runs with.

Every `JEVJAM_*` setting below used to be named `LAYA_*`, for example `LAYA_API_KEY`.
The old name still works and logs a deprecation warning at startup; when both are
set, the `JEVJAM_*` one wins.

| Variable | Default | Meaning |
| --- | --- | --- |
| `JEVJAM_HOST` | `0.0.0.0` | Bind address inside the container. |
| `JEVJAM_PORT` | `8000` | Bind port. Compose publishes this same value on the host. |
| `JEVJAM_DEVICE` | auto | Torch device for Laya and Julia. Unset uses CUDA when there is a GPU, else CPU. `cuda` falls back to CPU with a warning if CUDA is unavailable. |
| `JEVJAM_IDLE_TIMEOUT` | `300` | Seconds without an inference request before every checkpoint is freed. `0` keeps them resident. Must be a whole number; anything else stops the server at startup. |
| `JEVJAM_MAX_LOADED` | `2` | Checkpoints, Laya's and Julia's together, that may stay resident while the server is awake; past this the least recently used one is dropped. Must be a whole number of 1 or more. |
| `LAYA_AUTO_TASK` | `0` | `1` lets Laya's router reach `typed-decisions` automatically. Laya-only, so it keeps its name. |
| `JEVJAM_THREADS` | torch default | Caps torch intra-op threads for CPU inference, for Laya and Julia; keep at or below physical cores. |
| `JEVJAM_API_KEY` | unset | When set, requests to `/v1/systemone` and `/mcp` must send `Authorization: Bearer <key>`. `GET /health` remains public. |
| `JEVJAM_LOG_LEVEL` | `info` | Uvicorn log level, and the level of this server's own log lines. |
| `HF_HOME` | `/models` | Checkpoint cache location. Mount a volume here. |
| `HF_TOKEN` | unset | Optional Hugging Face credential. |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints. |

`LAYA_PRELOAD` and `LAYA_MODELS` used to choose what to load at boot. They no longer
do anything, because there is no boot load; setting either logs a warning and is
otherwise ignored.

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

Waking costs one checkpoint build, not three, because a request loads only what it
routes to. Measured on an RTX 4070 Ti SUPER from a warm volume, that build is 0.6 s, and
a warm request that does not need one answers in 31 ms. Measure it on your own GPU before
setting a client timeout; laya's published numbers are 7.4 s on CPU and 10.3 s on a T4.

```bash
docker compose exec jevjam uv run --no-sync python -c \
  'import time; from laya import Router; r = Router(device="cuda"); r.load("english"); \
   t = time.perf_counter(); r.unload(); r.load("english"); print("%.1fs" % (time.perf_counter() - t))'
```

Two other things worth knowing. `JEVJAM_MAX_LOADED=1` holds only the checkpoint the last
request routed to, which frees more but pays a rebuild on every language or model switch. And
freeing the checkpoints does not release the CUDA context or the memory torch itself
holds: only letting the container stop does that, and nothing inside the image can
wake it up again on the next request.

## MCP server

The same models serve [MCP](https://modelcontextprotocol.io) tools at `/mcp` over
streamable HTTP. Laya's own `laya-mcp-server` speaks stdio; this app reuses its tool
handlers but serves them over HTTP so the server can keep running independently of a
client. The server calls itself `jevjam`, and four tools are available. They were
named `laya_*` before the rename; the old names are gone.

| Tool | What it does |
| --- | --- |
| `jevjam_predict` | Answer typed questions (`choice` / `score` / `noul`) over a state in one forward pass. Like `POST /v1/systemone`, but for a calling agent. `model: "julia-1"` asks Julia-1. |
| `jevjam_preset` | Run a built-in workflow on Laya: `guard`, `moderation`, `triage`, or `model_router`. |
| `jevjam_route` | Which Laya checkpoint would answer, without loading one. |
| `jevjam_status` | Device in use, package versions, and what is resident. |

Those are the calls that run a forward pass (`jevjam_predict` directly, `jevjam_preset`
through it), so they are the ones that keep a checkpoint resident and reset the idle
clock.

### Authentication and remote access

When `JEVJAM_API_KEY` is set, every request to `/mcp` must include
`Authorization: Bearer <key>`. The same key protects the HTTP API. Leave it unset to
allow unauthenticated requests; `GET /health` stays public either way.

For local clients, use `http://127.0.0.1:8000/mcp`. For remote clients, point them at
the proxy URL. The proxy handles TLS and routes to jevjam; jevjam checks the Bearer token.
The four combinations are:

| Proxy URL | `JEVJAM_API_KEY` on server | Client sends Bearer token | Result |
| --- | --- | --- | --- |
| `http://jevjam.example.com/mcp` | set | yes | Authenticated, but HTTP does not encrypt the token or traffic. Use only on a trusted private network. |
| `https://jevjam.example.com/mcp` | set | yes | Authenticated and encrypted in transit. |
| `http://jevjam.example.com/mcp` | unset | no | Open to anyone who can reach it; traffic is unencrypted. |
| `https://jevjam.example.com/mcp` | unset | no | Open to anyone who can reach it; traffic is encrypted. |

If the server key is set but the client omits it or sends a wrong one, MCP returns
`401`. Keep the key out of shared project config. Set it in the server environment
and in the environment used to start the client. For example, in a local shell:

```bash
export JEVJAM_API_KEY=secret
docker compose up -d
```

### Install in a client

The examples below use user-local settings so credentials do not go into a shared
project file. In each client, use the local URL and omit the `Authorization` header
when the server key is unset. For a remote connection, replace the URL; add the
Bearer setting when the server key is set. A client using a remote URL must be able
to reach that proxy from its own host or network.

#### [OpenCode](https://opencode.ai/v2/docs/mcp-servers)

Add this to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "servers": {
      "jevjam": {
        "type": "remote",
        "url": "http://127.0.0.1:8000/mcp"
      }
    }
  }
}
```

For a remote server with a key, use this entry instead. OpenCode reads the variable
from its process environment:

```json
{
  "mcp": {
    "servers": {
      "jevjam": {
        "type": "remote",
        "url": "https://jevjam.example.com/mcp",
        "oauth": false,
        "headers": {
          "Authorization": "Bearer {env:JEVJAM_API_KEY}"
        }
      }
    }
  }
}
```

Check the connection with `opencode mcp list`. You can also add a server with
`opencode mcp add jevjam --global --url http://127.0.0.1:8000/mcp`. OpenCode remote
servers use OAuth by default, so the authenticated config sets `"oauth": false` and
sends the Bearer header instead. For a project-only entry, use the same shape in the
project's `opencode.json` and omit `--global` from the CLI command.

#### [Pi](https://pi.dev/)

Pi core does not include MCP support. Install the third-party
[`pi-mcp-adapter`](https://pi.dev/packages/pi-mcp-adapter) extension, then restart
Pi:

```bash
pi install npm:pi-mcp-adapter
```

Add this to the user config at `~/.config/mcp/mcp.json`:

```json
{
  "mcpServers": {
    "jevjam": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

For a project-shared server, the adapter also reads `.mcp.json` in the project root.

For a remote server with a key, use this config instead. The adapter expands the
environment variable:

```json
{
  "mcpServers": {
    "jevjam": {
      "url": "https://jevjam.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${JEVJAM_API_KEY}"
      }
    }
  }
}
```

#### [Codex](https://developers.openai.com/codex/mcp)

Add the local server to `~/.codex/config.toml`:

```bash
codex mcp add jevjam --url http://127.0.0.1:8000/mcp
```

For a remote server with a key, use the proxy URL and name the environment variable
that holds the token:

```bash
codex mcp add jevjam --url https://jevjam.example.com/mcp \
  --bearer-token-env-var JEVJAM_API_KEY
```

Check with `codex mcp list` or `/mcp` in the Codex TUI. Codex also reads project
settings from `.codex/config.toml` in trusted projects.

#### [Claude Code](https://code.claude.com/docs/en/mcp)

Add the local server for all your projects:

```bash
claude mcp add --transport http --scope user jevjam http://127.0.0.1:8000/mcp
```

For a remote server with a key, use the proxy URL and pass the token from the
environment:

```bash
claude mcp add --transport http --scope user jevjam \
  https://jevjam.example.com/mcp \
  --header "Authorization: Bearer $JEVJAM_API_KEY"
```

Check with `claude mcp list` or `/mcp`. Omit `--scope user` to keep the server
private to the current project; `--scope project` writes a shared `.mcp.json`.

### Reverse proxy examples

These snippets route `/mcp` to jevjam over HTTP and pass the `Authorization` header
through. Configure TLS on the proxy as you normally would; these examples do not
set up certificates or add a second auth layer. If the proxy runs in Docker, use an
upstream address it can reach, such as `http://jevjam:8000` when both containers share
a network, instead of `127.0.0.1`.

Nginx, inside the proxy's existing `server` block:

```nginx
location /mcp {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_read_timeout 10m;
    proxy_set_header Host $host;
    proxy_set_header Authorization $http_authorization;
}
```

Traefik dynamic configuration:

```yaml
http:
  routers:
    jevjam-mcp:
      rule: "Host(`jevjam.example.com`) && PathPrefix(`/mcp`)"
      service: jevjam-mcp
  services:
    jevjam-mcp:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8000"
```

Traefik forwards `Authorization` by default. Do not add middleware that removes it.

## API

The HTTP API exposes `/health` and `/v1/systemone`; MCP is available at `/mcp` on the
same port. `JEVJAM_API_KEY` protects the HTTP API and MCP when set. Keep the published
port on loopback for local use, or put a TLS reverse proxy in front for remote clients.

### `GET /health`

Always unauthenticated. Reports whether MCP is ready and which checkpoints are
resident. An empty `loaded` means cold or asleep; see [Sleeping on idle](#sleeping-on-idle).

```json
{"status": "ok", "loaded": ["english", "multilingual"], "device": "cuda", "mcp_ready": true}
```

That is a server that has answered in both languages and not gone idle yet; the
default `JEVJAM_MAX_LOADED` of 2 is what stops a third one staying resident.

### `POST /v1/systemone`

The decision endpoint, wire-compatible with TypeSafe Jev.

| Field | Required | Description |
| --- | --- | --- |
| `state` | yes | The text or JSON object to decide on. Strings and objects both work. |
| `questions` | yes | Object of `name -> question`. 1 to 64 questions per request. |
| `model` | no | Force a checkpoint: `english`, `multilingual`, `typed-decisions`, or the Hub ids `convaiinnovations/laya-multilingual` / `convaiinnovations/laya-typed-decisions`. `julia-1` (or `julia`, `SupersonicLabs/Julia-1`, any case) asks Julia-1. Anything else (e.g. a Jev model id) is ignored and Laya's router auto-selects. |

Unknown fields are ignored, so existing Jev clients keep working. All questions in one
request are answered in a single forward pass.

#### Question types

| Type | Fields | Answer shape |
| --- | --- | --- |
| `choice` | `instructions`, `criteria` (object of `label -> description`, or a list) | `choice` (top label), `probabilities` per option, `confidence`, `answer_confidence` |
| `score` | `instructions`, `criteria` (ordered list of levels, low to high) | `score` (expected level as a number), `legend`, `probabilities`, `confidence`, `answer_confidence` |
| `noul` | `instructions`; optional `criteria` keyed exactly `true`/`false` | `noul` (probability that the answer is true), `confidence`, `answer_confidence` |

`confidence` is the model's confidence in its own answer; `answer_confidence` is the
probability of the chosen option. `action.act_probability` is present but carries no
usable signal today.

#### JSON Schema

The server does not require a schema, but request bodies can be validated against this
[JSON Schema](https://json-schema.org/) (draft 2020-12). Save it next to your client
code and run it through any validator, or use it for code generation:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "jevjam POST /v1/systemone request",
  "type": "object",
  "required": ["state", "questions"],
  "properties": {
    "state": {
      "description": "The text or JSON object to decide on.",
      "oneOf": [{"type": "string"}, {"type": "object"}]
    },
    "model": {
      "description": "Optional checkpoint override; unknown values are ignored and the router auto-selects.",
      "enum": [
        "english",
        "multilingual",
        "typed-decisions",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
        "julia-1",
        "julia",
        "SupersonicLabs/Julia-1"
      ]
    },
    "questions": {
      "type": "object",
      "minProperties": 1,
      "maxProperties": 64,
      "additionalProperties": {"$ref": "#/$defs/question"}
    }
  },
  "additionalProperties": true,
  "$defs": {
    "question": {
      "oneOf": [
        {"$ref": "#/$defs/choice"},
        {"$ref": "#/$defs/score"},
        {"$ref": "#/$defs/noul"}
      ]
    },
    "choice": {
      "type": "object",
      "required": ["type", "instructions", "criteria"],
      "properties": {
        "type": {"const": "choice"},
        "instructions": {"type": "string"},
        "criteria": {
          "oneOf": [
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"type": "array", "items": {"type": "string"}}
          ]
        }
      }
    },
    "score": {
      "type": "object",
      "required": ["type", "instructions", "criteria"],
      "properties": {
        "type": {"const": "score"},
        "instructions": {"type": "string"},
        "criteria": {
          "description": "Levels ordered low to high.",
          "type": "array",
          "items": {"type": "string"}
        }
      }
    },
    "noul": {
      "type": "object",
      "required": ["type", "instructions"],
      "properties": {
        "type": {"const": "noul"},
        "instructions": {"type": "string"},
        "criteria": {
          "description": "Optional model-facing option text, keyed exactly true/false.",
          "type": "object",
          "required": ["true", "false"],
          "properties": {"true": {"type": "string"}, "false": {"type": "string"}},
          "additionalProperties": false
        },
        "labels": {
          "description": "Optional model-facing labels, keyed exactly true/false.",
          "type": "object",
          "required": ["true", "false"],
          "properties": {"true": {"type": "string"}, "false": {"type": "string"}},
          "additionalProperties": false
        }
      }
    }
  }
}
```

For example, with [`check-jsonschema`](https://github.com/python-jsonschema/check-jsonschema):

```bash
check-jsonschema --schemafile systemone-request.schema.json request.json
```

#### Example

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {
      "from": "user@acme.com",
      "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs, outages, system errors",
          "other": "everything else"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "blocking"]
      },
      "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
      }
    }
  }'
```

Real response from this image:

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.972, "technical": 0.0157, "other": 0.0123},
      "confidence": 0.8663,
      "answer_confidence": 0.972,
      "action": {"act_probability": 1.0}
    },
    "urgency": {
      "type": "score",
      "score": 1.64,
      "legend": {"0": "not urgent", "1": "soon", "2": "blocking"},
      "probabilities": {"0": 0.0743, "1": 0.2114, "2": 0.7143},
      "confidence": 0.3064,
      "answer_confidence": 0.7143,
      "action": {"act_probability": 1.0}
    },
    "churn_risk": {
      "type": "noul",
      "noul": 0.8229,
      "confidence": 0.8229,
      "answer_confidence": 0.8229,
      "action": {"act_probability": 1.0}
    }
  },
  "usage": {"input_tokens": 253, "output_tokens": 0},
  "routing": {
    "model": "english",
    "repo": "convaiinnovations/laya",
    "reason": "English Latin text",
    "detection": {
      "script": "latin",
      "script_profile": {"latin": 1.0},
      "language": "en",
      "is_english": true,
      "language_undecided": false,
      "diacritic_rate": 0.0,
      "non_latin_fraction": 0.0
    },
    "workflow": null
  }
}
```

`routing` explains why a checkpoint was chosen; `usage.input_tokens` counts the
tokenized state and questions for the chosen checkpoint.

#### With authentication

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Authorization: Bearer secret' \
  -H 'Content-Type: application/json' \
  -d '{"state": "I was charged twice. Please refund this.", "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}'
```

#### Limits and errors

| Condition | Status |
| --- | --- |
| Body larger than 2 MiB, more than 64 questions, or state longer than 50,000 characters | `413` |
| Body is not JSON, or is not an object with a `questions` field | `400` |
| Missing/invalid bearer token when `JEVJAM_API_KEY` is set | `401` |
| Question validation error; the message names the question and the fix | `422` |
| Inference failure (OOM and similar) | `500` |

The HTTP API runs one inference at a time through its own worker. MCP calls do not use
that queue and may overlap with API or other MCP inferences. Combine related questions
in one call and avoid sending many MCP predictions at once if GPU memory is tight.

## Julia-1

[Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) answers a request only when
its `model` field (or `jevjam_predict`'s `model` argument) names it. Auto-routing,
presets and `jevjam_route` stay Laya-only. Its first request downloads 550 MB into
`/models`. It shares `JEVJAM_DEVICE`, `JEVJAM_THREADS`, `JEVJAM_MAX_LOADED` and the
idle timer with Laya, and on CUDA it runs in BF16.

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model": "julia-1", "state": "I was charged twice. Please refund this.",
       "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}'
```

The answer has the same shape as Laya's, with a few differences:

- `model` and `routing.model` are `julia-1`, and `routing.repo` is `SupersonicLabs/Julia-1`.
- `confidence` and `answer_confidence` are both the probability of the chosen option.
  Julia has no separate self-confidence, and no `action` field.
- Probabilities are not rounded.
- Each question takes 2 to 20 options, and an option may be at most 48 tokens. The
  state, question and options must fit in 8,192 tokens, with no truncation. A request
  past these limits gets `422`, naming the question when Julia can tell which one.

## Notes on the models

Laya returns calibrated-intent probabilities, but the shipped checkpoints are a base to
evaluate, not a drop-in oracle: validate thresholds on your own data, prefer semantic
labels over `true`/`false` in `choice` questions, and expect degraded accuracy on choice
questions with more than ~20 options. The
[model card](https://huggingface.co/convaiinnovations/laya) documents the benchmarks and
the known limits. Julia-1's
[model card](https://huggingface.co/SupersonicLabs/Julia-1) does the same; it scores
73% on typed decisions, and weaker on unfamiliar domains and long label lists.

## License

The image contains the Apache-2.0 [Laya](https://github.com/NandhaKishorM/laya) package
and the Apache-2.0 checkpoints by Convai Innovations, and the Apache-2.0 Julia-1 code
and checkpoint by Supersonic Labs.
