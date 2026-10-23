# AGENTS.md - CI Workflows (.github/)

## Rules for Changes
These keep zizmor (`zizmor.yml`, fails on medium or higher findings) and the OpenSSF Scorecard clean.
- Pin every third-party action to a full commit SHA with a `# <release tag>` comment.
- Every workflow declares top-level `permissions` (`read-all`, `contents: read` or `{}`). Grant more per job, with a comment saying why.
- Every `actions/checkout` sets `persist-credentials: false`.
- Never inline untrusted contexts (`github.event.*`, `github.head_ref`, `inputs.*`) in `run:`; pass them through `env:`. Prefer `env:` for every value in new code.
- Every job sets `timeout-minutes`; for a job that calls a reusable workflow, set it inside the called workflow.
- `actionlint` runs in pre-commit.
- `workflows/array-api-skips.txt` lists array-api-tests node IDs skipped by the conformance job; precede each with a comment naming the upstream issue.
- `bump-sycl-deps.yml` rewrites the `DRIVER_PATH`, `OCLCPUEXP_FN` and `TBB_*` pins in `os-llvm-sycl-build.yml` with `sed`; keep their format.
- Dependabot (`dependabot.yml`) bumps actions, `environments/` pins and pre-commit hooks weekly with a 7-day cooldown.
