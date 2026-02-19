# GitHub Copilot Instructions — dpnp

## Source of truth
This file is canonical for Copilot behavior in this repository.

For project context and directory routing, read root `AGENTS.md` first.

`AGENTS.md` files provide directory scope/guardrails and should not duplicate Copilot policy text.

## Mandatory flow
1. Read root `AGENTS.md`.
2. Read nearest local `AGENTS.md` for edited files.
3. Most specific AGENTS wins when multiple apply.

## Contribution expectations
- Keep diffs minimal and scoped to the task.
- Preserve NumPy-compatible API behavior by default.
- For behavior/API changes: update tests, and docs/examples when user-visible.
- For bug fixes: prefer adding regression coverage.

## Authoring rules
- Use source-of-truth files for mutable details.
- Do not invent or hardcode versions, flags, or matrix values.

## Source-of-truth files
All paths relative to project root:
- Build/config: `CMakeLists.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`
- CI/checks: `.github/workflows/`
- Style/lint: `.pre-commit-config.yaml`, `.clang-format`, `.flake8`
- API/package paths: `dpnp/`, `doc/`
- Examples/benchmarks context: `examples/`, `benchmarks/`
