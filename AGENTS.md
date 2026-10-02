# Agent notes

## Checks
- `uv sync --locked` installs dependencies; run the CI test command with `uv run --locked pytest -q`. For a focused run, use `uv run --locked pytest -q tests/test_mcp_server.py`; to run one test, append `::test_name` to its file path.
- Tests use a stub router and need no GPU, model download, or Hub access. The locked environment still installs CUDA 12.8 PyTorch wheels and uses substantial disk space.
- If you change dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`; CI uses `uv sync --locked`.

## Wiring
- The console entry point `jevjam` is `src/jevjam/__init__.py`; `build_app()` combines Laya's HTTP app and streamable-HTTP MCP endpoint.
- `src/jevjam/models.py` holds `Models`, which puts Laya's Router and other backends behind the Router's own surface. It runs every inference and unload on one FIFO queue shared by HTTP and MCP, and keeps one checkpoint resident across all families. A new model family is a `Backend` subclass (see `Julia` there and `Clef` in `clef.py`), added to `build_app`'s default `others`.
- `build_app` serves its own `/health` and `/v1/systemone` instead of laya's `create_app`: laya's handler drops `images`/`videos` and caps every body at 2 MiB. It still reuses laya's request checks.
- clef-flash's code is vendored unchanged in `src/jevjam/vendor/joint_schema_model.py`; its header commit must equal `REVISION` in `clef.py` (a test checks this). Media strings must stay limited to base64, `data:` URIs and http(s) URLs: the HF processor opens any other string as a server path. Its processor needs torchvision (from the CUDA index, matching torch); video goes through PyAV because transformers otherwise wants torchcodec and system FFmpeg.
- Keep one `Models` shared across HTTP and MCP. MCP tools use `laya.mcp.server._ROUTER`; this wrapper sets it and `_ensure_router` so MCP calls use the same idle watcher and model cache. `tests/test_mcp_server.py` covers this wiring.
- Laya is pinned to 0.3.20 because the wrapper uses private `laya.serve` helpers (`_resolve_model`, `_check_request_limits`), laya's MCP globals (`VALID_MODELS`), and the MCP server's tool manager (to rename tools to `jevjam_*`). Before changing the pin, verify those APIs and run the tests.
- Julia-1's code comes from its Hugging Face repo as a uv git source. Its `rev` in `pyproject.toml` must equal `REVISION` in `models.py`; a test checks this. Julia pins `transformers<5.1`; `[tool.uv] override-dependencies` lifts that, and `load_julia` turns off the fast path that needs it.

## Runtime and image
- Keep the `JEVJAM_IDLE_TIMEOUT` default in Python. Compose passes unset values as empty strings, which the config parser treats as defaults.
- Settings are `JEVJAM_*`; `apply_env()` accepts the old `LAYA_*` names with a warning and copies the new values back to the `LAYA_*` names laya reads itself. Don't set `JEVJAM_*` defaults in the Dockerfile: they would shadow an operator's old `LAYA_*` values.
- PyTorch comes from the explicit CUDA 12.8 index in `pyproject.toml`; the Docker build checks `torch.version.cuda == "12.8"`. Update the index, lockfile, and Docker validation together if changing the CUDA/PyTorch version.

## Agent skills

### Issue tracker

Issues and specs live in this repo's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout. See `docs/agents/domain.md`.
