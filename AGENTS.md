# Agent notes

## Checks
- `uv sync --locked` installs dependencies; run the CI test command with `uv run --locked pytest -q`. For a focused run, use `uv run --locked pytest -q tests/test_mcp_server.py`; to run one test, append `::test_name` to its file path.
- Tests use a stub router and need no GPU, model download, or Hub access. The locked environment still installs CUDA 12.8 PyTorch wheels and uses substantial disk space.
- If you change dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`; CI uses `uv sync --locked`.

## Wiring
- The console entry point `laya-idle-serve` is `src/laya_idle_serve/__init__.py`; `build_app()` combines Laya's HTTP app and streamable-HTTP MCP endpoint.
- Keep one `Router` shared across HTTP and MCP. MCP tools use `laya.mcp.server._ROUTER`; this wrapper sets it and `_ensure_router` so MCP calls use the same idle watcher and model cache. `tests/test_mcp_server.py` covers this wiring.
- Laya is pinned to 0.3.20 because the wrapper uses private `laya.serve` helpers and MCP server globals. Before changing the pin, verify those APIs and run the MCP tests.

## Runtime and image
- Keep `LAYA_IDLE_TIMEOUT` and `LAYA_MAX_LOADED` defaults in Python. Compose passes unset values as empty strings, which the config parser treats as defaults.
- PyTorch comes from the explicit CUDA 12.8 index in `pyproject.toml`; the Docker build checks `torch.version.cuda == "12.8"`. Update the index, lockfile, and Docker validation together if changing the CUDA/PyTorch version.
