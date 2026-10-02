# AGENTS.md

Entry point for agent context in this repo.

## What this repository is
`dpnp` is the Data Parallel Extension for NumPy: a NumPy-compatible Python API with accelerated execution backends and cross-platform packaging/CI.

For detailed documentation, build instructions, and contribution guidelines, see:
- `README.md` (project overview and quickstart)
- `CONTRIBUTING.md` (if exists, or refer to project docs)

## How to work in this repo
- Keep changes small and single-purpose (bug fix, perf, docs, or infra).
- Preserve NumPy-compatible API behavior by default; call out intentional behavior changes in PR text.
- Pair user-visible changes with tests and docs/examples updates.
- Avoid duplicating mutable details (versions/flags/matrix) in instruction docs.

For Copilot behavior policy (flow, source-of-truth, edit hygiene), see:
- `.github/copilot-instructions.md` (canonical for Copilot behavior)

## Directory map
Below directories have local `AGENTS.md` for deeper context:
- `.github/AGENTS.md` — CI/workflow/automation context
- `dpnp/AGENTS.md` — core NumPy-compatible implementation guardrails
- `doc/AGENTS.md` — documentation updates and consistency
- `examples/AGENTS.md` — runnable user-facing examples
- `benchmarks/AGENTS.md` — reproducible performance evaluation
- `scripts/AGENTS.md` — maintenance/developer scripts
- `tests_external/AGENTS.md` — external/integration validation scope
