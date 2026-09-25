# Laya in Docker: Jev-compatible HTTP server

This repository builds and runs [Laya](https://huggingface.co/convaiinnovations/laya)
as a GPU-backed HTTP server that answers typed decision questions on the same
`POST /v1/systemone` protocol as TypeSafe Jev. A client written against Jev only needs
its base URL changed; the request and response shapes already match.

What the image does:

- Installs `laya[serve]` (0.3.20) with CUDA 12.8 PyTorch wheels on `python:3.12-slim-bookworm`.
- Runs `laya-serve` as non-root UID 10001, listening on `0.0.0.0:8000`.
- Caches downloaded checkpoints in `/models` (`HF_HOME`), designed to sit on a Docker volume.
- Ships a `/health` healthcheck that turns healthy only after checkpoints are preloaded.

## Requirements

- Docker Engine with Compose v2 (or plain `docker run`).
- An NVIDIA GPU with a working driver, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so containers can access it.
- Disk and VRAM for the checkpoints you preload. All three are ~1.2B parameters total.

Verified on an RTX 4070 Ti SUPER (driver 615.71.09) with `torch 2.11.0+cu128`.

## Quick start (Compose)

```bash
docker compose up --build -d
docker compose logs -f laya
curl -s http://127.0.0.1:8000/health
```

The first boot downloads the preloaded checkpoints before the server starts listening,
so `/health` can take minutes to answer. Once it does:

```json
{"status": "ok", "loaded": ["english", "multilingual", "typed-decisions"], "device": "cuda"}
```

Stop and start without redownloading:

```bash
docker compose down       # stop; the model volume survives
docker compose up -d      # start again
docker compose down -v    # stop and delete the downloaded checkpoints
```

## Quick start (Docker CLI)

```bash
docker build -t berkelaya:latest .

docker run -d --name laya --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v laya-models:/models \
  berkelaya:latest
```

If you would rather not build locally, CI publishes the image to GitHub Container
Registry on every push to `main`:

```bash
docker pull ghcr.io/beremaran/laya-docker:latest
docker run -d --name laya --gpus all -p 127.0.0.1:8000:8000 \
  -v laya-models:/models ghcr.io/beremaran/laya-docker:latest
```

Check GPU access without downloading any checkpoint:

```bash
docker run --rm --gpus all berkelaya:latest python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

## Model cache

Checkpoints go to `/models` inside the container. Keep it on a named volume (as above)
and the download happens once; delete the volume to force a fresh download. Pass
`-e HF_TOKEN=...` if you have a Hugging Face token, or `-e HF_HUB_OFFLINE=1` to refuse
network access and use only what is cached.

## Configuration

All configuration is by environment variable. Defaults shown are what the image runs with.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LAYA_HOST` | `0.0.0.0` | Bind address inside the container. |
| `LAYA_PORT` | `8000` | Bind port. Compose publishes this same value on the host. |
| `LAYA_DEVICE` | `cuda` | Torch device. Falls back to CPU with a warning if CUDA is unavailable. |
| `LAYA_PRELOAD` | `1` | Build preloaded checkpoints at startup instead of on first request. |
| `LAYA_MODELS` | all | Comma list to preload: `english,multilingual,typed-decisions`. Narrow it to save VRAM and download time, e.g. `english`. |
| `LAYA_AUTO_TASK` | `0` | `1` lets the router reach `typed-decisions` automatically. |
| `LAYA_THREADS` | torch default | Caps torch intra-op threads for CPU inference; keep at or below physical cores. |
| `LAYA_API_KEY` | unset | When set, requests must send `Authorization: Bearer <key>`. |
| `LAYA_LOG_LEVEL` | `info` | Uvicorn log level. |
| `HF_HOME` | `/models` | Checkpoint cache location. Mount a volume here. |
| `HF_TOKEN` | unset | Optional Hugging Face credential. |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints. |

For example, to serve English only and require a key:

```bash
LAYA_MODELS=english LAYA_API_KEY=secret docker compose up -d
```

## API

The server exposes two routes. There is no authentication until `LAYA_API_KEY` is set,
so keep the published port on loopback and put a TLS reverse proxy in front for remote
clients.

### `GET /health`

Always unauthenticated. Reports the server state and which checkpoints are resident.

```json
{"status": "ok", "loaded": ["english", "multilingual", "typed-decisions"], "device": "cuda"}
```

### `POST /v1/systemone`

The decision endpoint, wire-compatible with TypeSafe Jev.

| Field | Required | Description |
| --- | --- | --- |
| `state` | yes | The text or JSON object to decide on. Strings and objects both work. |
| `questions` | yes | Object of `name -> question`. 1 to 64 questions per request. |
| `model` | no | Force a checkpoint: `english`, `multilingual`, `typed-decisions`, or the Hub ids `convaiinnovations/laya-multilingual` / `convaiinnovations/laya-typed-decisions`. Anything else (e.g. a Jev model id) is ignored and the router auto-selects. |

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
  "title": "Laya POST /v1/systemone request",
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
        "convaiinnovations/laya-typed-decisions"
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
| Missing/invalid bearer token when `LAYA_API_KEY` is set | `401` |
| Question validation error; the message names the question and the fix | `422` |
| Inference failure (OOM and similar) | `500` |

Inference runs one request at a time (one worker), so put several questions in one
request rather than calling in parallel.

## Notes on the model

Laya returns calibrated-intent probabilities, but the shipped checkpoints are a base to
evaluate, not a drop-in oracle: validate thresholds on your own data, prefer semantic
labels over `true`/`false` in `choice` questions, and expect degraded accuracy on choice
questions with more than ~20 options. The
[model card](https://huggingface.co/convaiinnovations/laya) documents the benchmarks and
the known limits.

## License

The image contains the Apache-2.0 [Laya](https://github.com/NandhaKishorM/laya) package
and the Apache-2.0 checkpoints by Convai Innovations.