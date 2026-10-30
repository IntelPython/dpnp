# AGENTS.md - dpnp

dpnp is a NumPy-compatible array library for SYCL devices. Its main namespace, `dpnp.fft` and `dpnp.linalg` also conform to the Python Array API standard (2025.12, `doc/reference/array_api.rst`). The Python layer (`dpnp/`) is built on `dpnp.tensor` (Array API, `dpnp/tensor/`) and pybind11 SYCL extensions (`dpnp/backend/extensions/`); `dpctl` provides devices, queues and USM memory.

## Rules for Changes

These come from recurring maintainer review comments. Directory-specific rules are in the nearest `AGENTS.md`.

- Match NumPy, or SciPy for `dpnp.scipy`: shape, dtype, memory layout, edge cases (empty, 0-d, NaN/inf) and error and warning text. Following CuPy instead is an exception. Any deliberate difference is documented in the docstring and covered by a test with a comment.
- Don't break Array API conformance. Where NumPy and the spec differ, raise it in the PR rather than silently picking one (precedent: `ndarray.T` on non-2-D arrays was deprecated to match the spec).
- Code and tests must work on devices without fp64 support (e.g. Iris Xe). Don't hardcode float64/complex128; use `dpnp.default_float_type()` or the dtype helpers in `dpnp/tests/helper.py`.
- Compute follows data: allocate results and temporaries on the inputs' SYCL queue with their coerced USM type, and raise when inputs computed together are on different queues (`get_usm_allocations` raises `ValueError`, `dpnp.tensor` raises `ExecutionPlacementError`). Never move inputs silently.
- Avoid hidden host transfers and queue syncs: `int()`/`bool()` on device arrays, `.wait()`, `asnumpy`, redundant full-size copies.
- Accept both `dpnp.ndarray` and `usm_ndarray`; validate and unwrap with `dpnp.check_supported_arrays_type` and `dpnp.get_usm_ndarray`, not ad-hoc `isinstance` or method calls.
- Every user-visible change, including build, recipe and C-API changes, gets a one-line `CHANGELOG.md` entry (format below).
- A new public function needs an export in `dpnp/__init__.py` (or in its subpackage's `__init__.py`), a numpydoc docstring, an entry in `doc/reference/*.rst`, tests (including `test_sycl_queue.py` and `test_usm_type.py`) and a changelog entry. Don't export names that NumPy/SciPy don't.
- Deprecate before removing: keep the old API for at least one release with a `DeprecationWarning`, and add a `Deprecated` changelog entry and a test for the warning. Removing something NumPy itself has removed can land directly with a `Removed` entry.
- A removal, rename or version bump also updates every other place that names or relies on the old thing: grep the whole repository, workflows and recipes included. A site kept on purpose gets a `TODO`, unless a directory `AGENTS.md` says to keep it. Past `CHANGELOG.md` entries stay.
- Every new branch and error path has a test; a bug fix comes with a test that fails without it. Use `# pragma: no cover` only for branches the Linux coverage job can't reach (CUDA, HIP, Windows-only) or that can't happen by construction.
- A workaround, skip or xfail gets a comment with the full issue URL so it can be reverted later. In `dpnp/tests/third_party/`, a skip for an unsupported feature needs only its reason string.
- Keep comments short and accurate: one line of why. Don't narrate the change, restate the code, or mention "this PR".
- One PR, one change. Search for an existing helper before adding one.
- New files carry the BSD-3 license header from a neighbouring file with the current year; never change the year in existing files. Test files directly in `dpnp/tests/` and under `dpnp/tests/third_party/` have no header.
- Don't edit generated files: `dpnp/_version.py`, `dpnp/include/dpnp/tensor/_usmarray*.h`, `dpnp/resources/cmake/`, `doc/reference/generated/`, `doc/reference/*.inc`.

## Verification Before You Push

```bash
pre-commit run --all-files         # the required `Check` job; pylint must be on PATH
python scripts/build_locally.py    # in-place build + editable install
python -m pytest -ra dpnp/tests/test_<area>.py
```

The build needs the packages in `environments/build_with_oneapi.yml`, `environments/oneapi_pkgs.yml` and `environments/dpctl_pkg.yml`. The `no-commit-to-branch` hook fails on `master` and `maintenance/*`, so work on a branch. Full-suite commands and test env vars are in `dpnp/tests/AGENTS.md`; build options in `dpnp/backend/AGENTS.md`.

### Changelog format

Append to the end of the matching `### Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security` list in the top `## [X.Y.Z] - YYYY-MM-DD` section. Past tense, code in backticks, no trailing period:

```
* Added support for the `"nuc"` (nuclear) norm in `dpnp.linalg.cond` [#3084](https://github.com/IntelPython/dpnp/pull/3084)
```

### Where the checks live

| Check | Workflow | Required on master |
| --- | --- | --- |
| pre-commit (`Check`) | `.github/workflows/pre-commit.yml` | yes |
| conda build and tests, Linux and Windows, py3.13-3.15 and 3.14t (`Test (...)`) | `conda-package.yml` | yes |
| Internal Jenkins (`jenkins-test-dpnp-3.13-linux-dev`, `...-3.14-...`) | external | yes |
| Array API conformance of `dpnp`, not yet `dpnp.tensor` (array-api-tests; never fails, results posted as a PR comment on non-fork PRs) | `conda-package.yml` | no |
| pybind11 examples | `conda-package.yml` | no |
| conda-forge build and tests | `conda-package-cf.yml`, matrix in `conda-build-test.yml` | no |
| oneMath build, open-source LLVM SYCL build | `check-onemath.yaml`, `os-llvm-sycl-build.yml` | no |
| Sphinx docs (`make spelling`, `make html`) | `build-sphinx.yml` | no |
| Coverage (Coveralls), zizmor | `generate_coverage.yaml`, `zizmor.yml` | no |

GitHub-hosted runners have no GPU, so tests there run on a CPU device: a green run says nothing about GPUs or devices without fp64.

## Reviewing

Report source-confirmed problems with NumPy compatibility, Array API conformance, compute-follows-data, devices without fp64, SYCL event ordering and lifetimes, error handling and test coverage. Don't report what pre-commit enforces (formatting, import order, spelling), and don't report that `@pytest.mark.slow` tests are skipped in CI: that is by design.

## Directory Guides

Read the `AGENTS.md` nearest the files you change. When editing these files, every command, path and name must match the repository; delete what can't be verified rather than soften it.

- [dpnp/AGENTS.md](dpnp/AGENTS.md): Python layer, adding a function, docstrings
- [dpnp/tests/AGENTS.md](dpnp/tests/AGENTS.md): test helpers, fp64, CuPy tests, running the suite
- [dpnp/backend/AGENTS.md](dpnp/backend/AGENTS.md): pybind11 SYCL extensions, C++ rules, build options
- [dpnp/tensor/AGENTS.md](dpnp/tensor/AGENTS.md): `dpnp.tensor` and libtensor
- [doc/AGENTS.md](doc/AGENTS.md): Sphinx docs
- [benchmarks/AGENTS.md](benchmarks/AGENTS.md): ASV benchmarks
- [.github/AGENTS.md](.github/AGENTS.md): workflow hardening rules

## Further Reading

- `doc/quick_start_guide.rst`: build from source, CUDA/HIP targets
- `.github/pull_request_template.md`: the PR checklist
