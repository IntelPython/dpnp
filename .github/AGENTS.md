# AGENTS.md - CI Workflows (.github/)

## Rules for Changes
These keep zizmor (`zizmor.yml`, fails on medium or higher findings) and the OpenSSF Scorecard clean.
- Pin every third-party action to a full commit SHA with a `# <release tag>` comment.
- Every workflow declares top-level `permissions` (`read-all`, `contents: read` or `{}`). Grant more per job, with a comment saying why.
- Every `actions/checkout` sets `persist-credentials: false`.
- Never inline untrusted contexts (`github.event.*`, `github.head_ref`, `inputs.*`) in `run:`; pass them through `env:`. Prefer `env:` for every value in new code.
- Every job sets `timeout-minutes`; for a job that calls a reusable workflow, set it inside the called workflow.
- `actionlint` runs in pre-commit.
- Set up conda with the local `.github/actions/setup-miniconda` action, which retries a failed setup once, instead of calling `conda-incubator/setup-miniconda` directly; a job that checks dpnp out into a subdirectory uses it from there (`./source/.github/actions/setup-miniconda`).
- `workflows/array-api-skips.txt` lists array-api-tests node IDs skipped by the conformance job; precede each with a comment naming the upstream issue.
- `bump-sycl-deps.yml` rewrites the `DRIVER_PATH`, `OCLCPUEXP_FN`, `TBB_URL`, `TBB_INSTALL_DIR` and `TBB_FN` pins in `.github/sycl-build-deps.env` with `sed`, and `os-llvm-sycl-build.yml` reads exactly those five; keep one unquoted `KEY=VALUE` per line, starting at column 0.
- Every write to `gh-pages` (`build-sphinx.yml`, `prune-pr-docs.yml`) goes through `.github/scripts/update_gh_pages.sh`, which retries a push rejected by a concurrent writer; don't push to `gh-pages` any other way.
- Dependabot (`dependabot.yml`) bumps actions, `environments/` pins and pre-commit hooks weekly with a 7-day cooldown. After a `dpctl` release it opens a PR that raises the `dpctl` floor in `environments/dpctl_pkg.txt` only; don't merge it. Instead, by hand, move the dev-series pins (`[project].dependencies` in `pyproject.toml`, `environments/dpctl_pkg.{txt,yml}`, `conda-recipe/meta.yaml`, `conda-recipe/rattler_recipe.yaml`) to the next `X.Y.0dev0`, and the release-only pins (`[build-system].requires` in `pyproject.toml`, `conda-recipe-cf/meta.yaml`) to the new release.
