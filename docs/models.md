# Models

jevjam serves three model families and keeps one checkpoint in VRAM at a time. Back to the [README](../README.md).

| Model | By | Size | Reads | Free VRAM | Answers |
| --- | --- | --- | --- | --- | --- |
| [Laya](https://huggingface.co/convaiinnovations/laya) | Convai Innovations | 3 checkpoints, ~1.2B in all | text, JSON | ~2.5 GB measured | every request by default |
| [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | Supersonic Labs | 144M (550 MB) | text, JSON | small (not measured) | requests that name it |
| [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | Cloudflare | 9B (19 GB) | text, JSON, images, video | 8 to 20 GB, or split with CPU | requests that name it |

A request for a model that is not resident frees the one that is, then loads its own.
Requests wait in one queue, so a load never runs under another request.

## Laya

[Laya](https://huggingface.co/convaiinnovations/laya) answers every request that does
not name another model. Its router picks one of three checkpoints per request:
`english`, `multilingual` or `typed-decisions` (the last only when `model` names it,
or with `LAYA_AUTO_TASK=1`). `routing` in the answer says which one it chose and why.
Laya also runs the MCP presets (`guard`, `moderation`, `triage`, `model_router`).

## Julia-1

[Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) answers a request only when
its `model` field (or `jevjam_predict`'s `model` argument) names it. Auto-routing,
presets and `jevjam_route` stay Laya-only. Its first request downloads 550 MB into
`/models`. It shares `JEVJAM_DEVICE`, `JEVJAM_THREADS`, the queue and the idle timer
with the other models, and on CUDA it runs in BF16. Julia pins `transformers<5.1`
for a fast path that newer releases broke; jevjam turns that path off, which gives
the same probabilities about 1 ms slower, and runs Julia on the newer transformers
clef-flash needs.

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

## clef-flash

[clef-flash](https://huggingface.co/Cloudflare/clef-flash) answers a request only when
its `model` field (or `jevjam_predict`'s `model` argument) names it. It is the one
model here that reads `images` and `videos` as well as `state`.

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"clef-flash\", \"state\": \"Is it safe to leave the kitchen?\",
       \"images\": [\"$(base64 -w0 kitchen.jpg)\"],
       \"questions\": {\"stove_on\": {\"type\": \"noul\", \"instructions\": \"Is a stove burner on?\"}}}"
```

Its first request downloads 19 GB into `/models`, and every switch to it from another
model rebuilds it, so give clients a long timeout. It loads in the first of these
modes that fits in free VRAM, falling through to the next if a mode runs out of memory
anyway:

| Mode | Free VRAM needed | Notes |
| --- | --- | --- |
| `bf16` | ~20 GB | As Cloudflare ships it. |
| `8bit` | ~12 GB | bitsandbytes LLM.int8. Picked on a 16 GB card. |
| `4bit` | ~8 GB | bitsandbytes NF4. |
| `offload` | any | BF16, with the layers that do not fit run from CPU memory by accelerate. |

Measured on an RTX 4070 Ti SUPER (16 GB), with the weights already in the page cache; reading them from disk first adds about 15 s:

| Mode | VRAM | Load | Warm text request | Largest gap from BF16 |
| --- | --- | --- | --- | --- |
| `8bit` | 10.9 GB | 14 s | 123 ms | 0.017 |
| `4bit` | 7.9 GB | 4 s | 126 ms | 0.077 |
| `offload` | 10.5 GB | 3 s | 653 ms | (is BF16) |

The last column is the largest difference in answer probability from `offload`, over
six answers to text, image and video questions. That is a smoke test, not a benchmark:
Cloudflare's benchmarks are for BF16, and jevjam has not measured how much the
quantized modes lose on them. Set `JEVJAM_CLEF_QUANT` to force a mode. When no mode loads, the
request gets `503` and a message naming what was tried. Without a GPU it runs on CPU,
at seconds per request.

Images and videos may be base64, `data:` URIs or `http(s)` URLs. The server fetches
URLs itself, so anyone who can reach the API can make it fetch from your network;
this is meant for self-hosting. Anything else, such as a file path, gets `422`.

The answer has the same shape as Laya's: `model` and `routing.model` are
`clef-flash`, `confidence` and `answer_confidence` are both the chosen option's
probability, and there is no `action` field.

## Notes on the models

Laya returns calibrated-intent probabilities, but the shipped checkpoints are a base to
evaluate, not a drop-in oracle: validate thresholds on your own data, prefer semantic
labels over `true`/`false` in `choice` questions, and expect degraded accuracy on choice
questions with more than ~20 options. The
[model card](https://huggingface.co/convaiinnovations/laya) documents the benchmarks and
the known limits. Julia-1's
[model card](https://huggingface.co/SupersonicLabs/Julia-1) does the same; it scores
73% on typed decisions, and weaker on unfamiliar domains and long label lists.
clef-flash's [model card](https://huggingface.co/Cloudflare/clef-flash) reports its
Decision Index results.
