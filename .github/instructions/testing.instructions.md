---
applyTo: "{dpnp/tests/**,tests_external/**}"
---

# Testing Instructions for GitHub Copilot

## Test coverage requirements
- Add unit/integration tests for all changed code paths.
- For bug fixes: include regression tests that cover the fixed scenario.

## Test quality
- Keep tests deterministic and scoped to changed behavior.
- Prefer smallest test surface proving correctness.

## Validation workflow
Ensure tests pass locally before PR submission:
```bash
pytest dpnp/tests/        # core tests
pytest tests_external/    # integration tests
```
