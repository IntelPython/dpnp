# GitHub Copilot Instructions — dpnp

## Source of truth
This file is canonical for Copilot behavior in this repository.

For project context and directory routing, read root `AGENTS.md` first.

`AGENTS.md` files provide directory scope/guardrails and should not duplicate Copilot policy text.

## Precedence hierarchy (conflict resolution)
When guidance conflicts across files, apply in this order:
1. `.github/copilot-instructions.md` (this file)
2. Applicable `.github/instructions/*.instructions.md` files
3. Nearest local `AGENTS.md` for edited path
4. Root `AGENTS.md`

## Mandatory flow
1. Read root `AGENTS.md`.
2. For each edited file, read its nearest local `AGENTS.md`.
3. If multiple directories changed, apply each scope's local AGENTS to corresponding files.
4. Most specific AGENTS wins for each file.

## Contribution expectations
- Keep diffs minimal and scoped to the task.
- Preserve NumPy-compatible API behavior by default.
- For behavior/API changes: update tests, and docs/examples when user-visible (API signature/behavior, errors, examples output).
- For bug fixes: add regression test coverage.

## Test and validation requirements
- Add unit/integration tests for changed code paths.
- For bug fixes: include regression tests.
- Keep tests deterministic and scoped to changed behavior.

## Authoring rules
- Use source-of-truth files for mutable details.
- Do not invent or hardcode versions, flags, or matrix values.
- Prefer single-purpose commits; avoid opportunistic refactors.

## Source-of-truth files
All paths relative to project root:
- Build/config: `CMakeLists.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`
- CI/checks: `.github/workflows/`
- Style/lint: `.pre-commit-config.yaml`, `.clang-format`, `.flake8`
- API/package paths: `dpnp/`, `doc/`
- Examples/benchmarks context: `examples/`, `benchmarks/`
