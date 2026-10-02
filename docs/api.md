# API

Back to the [README](../README.md).

The HTTP API exposes `/health` and `/v1/systemone`; MCP is available at `/mcp` on the
same port. `JEVJAM_API_KEY` protects the HTTP API and MCP when set. Keep the published
port on loopback for local use, or put a TLS reverse proxy in front for remote clients.

## `GET /health`

Always unauthenticated. Reports whether MCP is ready and which checkpoints are
resident. An empty `loaded` means cold or asleep; see [Sleeping on idle](configuration.md#sleeping-on-idle).

```json
{"status": "ok", "loaded": ["english"], "device": "auto", "mcp_ready": true}
```

That is a server whose last request went to Laya's English checkpoint, and which has
not gone idle yet.

## `POST /v1/systemone`

The decision endpoint, wire-compatible with TypeSafe Jev.

| Field | Required | Description |
| --- | --- | --- |
| `state` | yes | The text or JSON object to decide on. Strings and objects both work. |
| `questions` | yes | Object of `name -> question`. 1 to 64 questions per request. |
| `model` | no | Force a checkpoint: `english`, `multilingual`, `typed-decisions`, or the Hub ids `convaiinnovations/laya-multilingual` / `convaiinnovations/laya-typed-decisions`. `julia-1` (or `julia`, `SupersonicLabs/Julia-1`, any case) asks Julia-1, and `clef-flash` (or `clef`, `Cloudflare/clef-flash`) asks clef-flash. Anything else (e.g. a Jev model id) is ignored and Laya's router auto-selects. |
| `images` | no | clef-flash only. A list of images, each base64, a `data:` URI, or an `http(s)` URL the server fetches. |
| `videos` | no | clef-flash only. A list of video files (MP4 and the like), each base64, a `data:` URI, or an `http(s)` URL the server fetches. |

Unknown fields are ignored, so existing Jev clients keep working. All questions in one
request are answered in a single forward pass.

### Question types

| Type | Fields | Answer shape |
| --- | --- | --- |
| `choice` | `instructions`, `criteria` (object of `label -> description`, or a list) | `choice` (top label), `probabilities` per option, `confidence`, `answer_confidence` |
| `score` | `instructions`, `criteria` (ordered list of levels, low to high) | `score` (expected level as a number), `legend`, `probabilities`, `confidence`, `answer_confidence` |
| `noul` | `instructions`; optional `criteria` keyed exactly `true`/`false` | `noul` (probability that the answer is true), `confidence`, `answer_confidence` |

`confidence` is the model's confidence in its own answer; `answer_confidence` is the
probability of the chosen option. `action.act_probability` is present but carries no
usable signal today.

### JSON Schema

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
        "SupersonicLabs/Julia-1",
        "clef-flash",
        "clef",
        "Cloudflare/clef-flash"
      ]
    },
    "images": {
      "description": "clef-flash only: base64, data: URIs, or http(s) URLs.",
      "type": "array",
      "items": {"type": "string"}
    },
    "videos": {
      "description": "clef-flash only: base64, data: URIs, or http(s) URLs of video files.",
      "type": "array",
      "items": {"type": "string"}
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

### Example

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

### With authentication

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Authorization: Bearer secret' \
  -H 'Content-Type: application/json' \
  -d '{"state": "I was charged twice. Please refund this.", "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}'
```

### Limits and errors

| Condition | Status |
| --- | --- |
| Body larger than 2 MiB (20 MiB for clef-flash), more than 64 questions, or state longer than 50,000 characters | `413` |
| Body is not JSON, or is not an object with a `questions` field | `400` |
| Missing/invalid bearer token when `JEVJAM_API_KEY` is set | `401` |
| Question validation error; the message names the question and the fix | `422` |
| Images or videos for a model other than clef-flash, or media that is not base64, a `data:` URI or an `http(s)` URL | `422` |
| Inference failure (OOM and similar) | `500` |
| clef-flash cannot load on this machine; the message says what was tried | `503` |

The HTTP API and MCP share one queue and run one request at a time, in arrival order.
Combine related questions in one call: a request waits for every request ahead of it.
