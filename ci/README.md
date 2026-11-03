# How Intel Python CI tests dpnp

Intel Python CI (gh-ci) builds and tests dpnp's conda packages and wheels in GitHub Actions. Intel's Jenkins then tests each package again on Intel machines, GPUs included, and scans it, and on a release tag publishes the packages it tested as the tag's GitHub Release. Everything it needs from this repository is in this directory, one workflow, the recipe, and `pyproject.toml`.

| File | Role |
|---|---|
| `../conda-recipe/` | The conda recipe, built against conda-forge only. `conda_build_config.yaml` lists the Python versions. The recipe skips macOS. Its own tests are `pip check` and an import; the test suite runs in `test.py`. |
| `../pyproject.toml` | What the wheels are built from. Its Python classifiers list the wheels' Python versions, and its Operating System classifiers their platforms; 3.15 is not classified because dpctl has no 3.15 wheel to build against yet. `[tool.cibuildwheel]` holds the wheel build settings, and `[tool.gh-ci]` holds the run-time requirements the wheels need on top of `[project]`'s, which the conda packages get from the recipe. |
| `../.github/workflows/intel-ci.yml` | One job reads both matrices. One job per platform and Python version builds and uploads `dpnp <Platform> Python <version>`, and one builds `dpnp <Platform> Wheels Python <version>`. Then one job per package tests it. Pushes to `master` and tags then upload the packages to the `dev` label of `dppy` on anaconda.org and remove old ones; no other event does. |
| `matrix.py` | Reads the platforms and Python versions from the recipe, or with `--system wheel` from `pyproject.toml`. A pull request gets the oldest and the newest regular versions. |
| `build.py` | Builds one Python version: with conda-build, which runs the recipe's own tests, or with `--system wheel` with cibuildwheel, in a manylinux container on Linux. |
| `hooks.py` | The workflow's test jobs: installs one package into a clean environment with the requirements in `config.yml`, and runs `test.py` the way Jenkins does. A conda package's environment comes from conda-forge; a wheel's has only Python from conda-forge and everything else from PyPI. |
| `install_icx.py` | Installs Intel's icx and the oneMKL, oneDPL and oneTBB development packages from conda-forge for the Windows wheel builds: `[tool.cibuildwheel.windows]` in `../pyproject.toml` runs it first. The Linux wheel builds get icpx from Intel's yum repository and the libraries from conda-forge with micromamba. |
| `config.yml` | Tells gh-ci which workflow builds the packages, what its artifacts are called, and what the tests and benchmarks need. |
| `test.py` | Runs the installed package's test suites with pytest, in Actions and again in Jenkins: `dpnp.tests`, then `dpnp.tests.tensor` on its own, which needs a lot of memory. It sets the SYCL and OpenCL variables itself, because the test environment has none. |
| `benchmark.py` | Runs the asv suite in `../benchmarks/` against the installed package, in Jenkins only, on a bare-metal benchmarking machine. Jenkins requires it because `../benchmarks/` exists. |

The scripts need only the Python standard library and are the same in every repository gh-ci tests. Scanning is not the repository's to define: Jenkins runs its own scans.

**Devices.** GitHub's runners have no GPU, so `test.py` runs there on the OpenCL CPU device of `intel-opencl-rt`. On Windows it registers that runtime's `intelocl64.dll` with the OpenCL loader in the machine's registry for the length of the tests, only if none is registered, as the runner has none and the loader ignores `OCL_ICD_FILENAMES`; on Linux it merges the environment's and the system's ICD files for the loader. Neither happens in a GPU cell on Windows, which uses the registered driver. Jenkins sets `GH_CI_DEVICE=gpu` for the cells it runs on GPU machines. `test.py` prints the SYCL devices, fails if there is none, and in a GPU cell fails if none is a GPU, so a cell that falls back to the CPU cannot pass. Workers of `pytest-xdist` run only on a CPU device, as they would share one GPU.

**Wheels.** They are built against the dpctl wheel on PyPI, with icpx 2026.2. They bundle none of the SYCL, oneMKL or compiler libraries: `[tool.gh-ci] wheel-requires` adds the wheels that provide them to the metadata, as the wheels on PyPI declare. Nothing is repaired: Linux only retags the wheels to `manylinux_2_28` without an `auditwheel` check, and Windows has no repair step. The Linux wheels link GLIBCXX_3.4.32, so they need the libstdc++ of GCC 13 or newer, as the dpctl wheels need 3.4.26. A wheel's test environment adds `intel-opencl-rt` from PyPI for the CPU device.

**A pull request into `master`.** Actions builds the oldest and the newest regular Python versions on Linux and Windows, as a conda package and as a wheel, and tests each package. Jenkins notices the pull request within 3 minutes, waits for that whole run, and then tests and scans every package at once, and benchmarks the newest Linux conda package, sampling each benchmark once. Each cell posts its own status on the pull request, and `intel-ci/pr` sums them up.

**A release tag.** Actions builds every declared version, and Jenkins tests and scans all of them, and measures the benchmarks properly on the newest Linux conda package, before it publishes the release. A benchmark that fails fails the run; a slower one does not, as nothing compares the timings with a baseline yet.

To support another Python, add it to `conda-recipe/conda_build_config.yaml` for the conda packages and as a classifier in `pyproject.toml` for the wheels; nothing else lists versions. Try it locally from a conda base environment with conda-build, or with cibuildwheel for wheels:

```bash
python ci/matrix.py                           # every platform and Python version
python ci/matrix.py --event pull_request      # what a pull request builds
python ci/build.py --python VERSION --output /tmp/out     # VERSION: one matrix.py lists
python ci/hooks.py --python VERSION --artifacts /tmp/out --work /tmp/hooks
python ci/build.py --system wheel --python VERSION --output /tmp/wheels
python ci/hooks.py --kind wheel --python VERSION --artifacts /tmp/wheels --work /tmp/wheel-hooks
```
