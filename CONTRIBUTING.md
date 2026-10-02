# Contributing

Issues and pull requests are welcome. For anything larger than a small fix, open an
issue first so we can agree on the shape before you write it.

## Set up and test

```bash
uv sync --locked
uv run --locked pytest -q
```

The tests use a stub model and need no GPU, model download or Hub access. The locked
environment still installs CUDA 12.8 PyTorch, so expect several GB of disk use.

If you change dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`;
CI installs with `--locked`.

## Before you change

[AGENTS.md](AGENTS.md) lists the pins and the parts that must move together, such as
the Laya version, Julia-1's and clef-flash's commits, and the torch and CUDA versions.
Read it before you touch those.

## Pull requests

- Keep each one to a single change, with tests for what it fixes or adds.
- Update the README or `docs/` when you change what a user sees.
- By opening one, you agree your work is licensed under [Apache-2.0](LICENSE).
