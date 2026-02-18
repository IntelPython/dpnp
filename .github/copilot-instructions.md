# GitHub Copilot Instructions — dpnp

## Source of truth
This file is canonical for Copilot behavior in this repository.

`AGENTS.md` files provide directory scope/guardrails for AGENTS-aware tools and should not duplicate Copilot policy text.

## Mandatory flow
Read root `AGENTS.md`, then nearest local `AGENTS.md` for edited files; most specific wins.

## Contribution expectations
- Keep diffs minimal and scoped to the task.
- Preserve API behavior by default.
- For behavior/API changes: update tests, and docs/examples when user-visible.
- For bug fixes: prefer adding regression coverage.

## Authoring rules
- Use source-of-truth files for mutable details.
- Do not invent or hardcode versions, flags, or matrix values.

## Source-of-truth files
- Build/config: `CMakeLists.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`
- CI/checks: `.github/workflows/`
- Style/lint: `.pre-commit-config.yaml`, `.clang-format`, `.flake8`
- API/package paths: `dpnp/`, `doc/`
- Examples/benchmarks context: `examples/`, `benchmarks/`
