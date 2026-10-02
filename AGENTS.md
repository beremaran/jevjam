# Agent notes

## Checks
- `uv sync --locked` installs dependencies; run the CI test command with `uv run --locked pytest -q`. For a focused run, use `uv run --locked pytest -q tests/test_mcp_server.py`; to run one test, append `::test_name` to its file path.
- Tests use a stub router and need no GPU, model download, or Hub access. The locked environment still installs CUDA 12.8 PyTorch wheels and uses substantial disk space.
- If you change dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`; CI uses `uv sync --locked`.

## Wiring
- The console entry point `jevjam` is `src/jevjam/__init__.py`; `build_app()` combines Laya's HTTP app and streamable-HTTP MCP endpoint.
- `src/jevjam/models.py` holds `Models`, which puts Laya's Router and other backends (now Julia-1) behind the Router's own surface and enforces one resident limit across them. A new model family is a new backend there, passed in `build_app`'s `others`.
- Keep one `Models` shared across HTTP and MCP. MCP tools use `laya.mcp.server._ROUTER`; this wrapper sets it and `_ensure_router` so MCP calls use the same idle watcher and model cache. `tests/test_mcp_server.py` covers this wiring.
- Laya is pinned to 0.3.20 because the wrapper uses private `laya.serve` helpers (`_resolve_model` is replaced so other backends can claim a `model`), laya's MCP globals (`VALID_MODELS`), and the MCP server's tool manager (to rename tools to `jevjam_*`). Before changing the pin, verify those APIs and run the tests.
- Julia-1's code comes from its Hugging Face repo as a uv git source. Its `rev` in `pyproject.toml` must equal `REVISION` in `models.py`; a test checks this. Julia pins `transformers<5.1`; `[tool.uv] override-dependencies` lifts that, and `load_julia` turns off the fast path that needs it.

## Runtime and image
- Keep `JEVJAM_IDLE_TIMEOUT` and `JEVJAM_MAX_LOADED` defaults in Python. Compose passes unset values as empty strings, which the config parser treats as defaults.
- Settings are `JEVJAM_*`; `apply_env()` accepts the old `LAYA_*` names with a warning and copies the new values back to the `LAYA_*` names laya reads itself. Don't set `JEVJAM_*` defaults in the Dockerfile: they would shadow an operator's old `LAYA_*` values.
- PyTorch comes from the explicit CUDA 12.8 index in `pyproject.toml`; the Docker build checks `torch.version.cuda == "12.8"`. Update the index, lockfile, and Docker validation together if changing the CUDA/PyTorch version.

## Agent skills

### Issue tracker

Issues and specs live in this repo's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout. See `docs/agents/domain.md`.
