# dpnp benchmarks

Benchmarking dpnp using Airspeed Velocity.
Read more about ASV [here](https://asv.readthedocs.io/en/stable/index.html).

## Usage

Unlike a pure-Python project, dpnp is a SYCL/DPC++ extension that requires the
Intel oneAPI compiler and a lengthy build, so ASV does not build dpnp itself:
`build_command` in `asv.conf.json` is empty and the benchmarks are run against
an **existing environment** that already has dpnp installed.

Create an environment
[following these instructions](https://intelpython.github.io/dpnp/quick_start_guide.html)
and install the benchmarking tooling into it.

Install the tooling directly, which leaves the already-built dpnp untouched:

```bash
conda install -c conda-forge asv scipy
```

Do **not** use `pip install ".[benchmark]"` for an environment that already has
dpnp: dpnp is a scikit-build project, so pip reinstalls the `dpnp` package
itself and triggers a full oneAPI/DPC++ rebuild of the backend just to pull in
two pure-Python dependencies. The `benchmark` extra exists for the case where
dpnp is being built from source anyway, e.g.:

```bash
pip install --no-build-isolation --no-deps -e ".[benchmark]"
```

Then activate the environment and run the benchmarks against it. The simplest
way is to point ASV at the currently active environment with `--python=same`:

```bash
conda activate dpnp_env
asv run --python=same --launch-method spawn --quick HEAD^!
```

Alternatively, point ASV explicitly at an environment's python binary:

```bash
asv run --environment existing:/full/conda/path/envs/dpnp_env/bin/python \
    --launch-method spawn
```

Compare two commits or check for regressions:

```bash
asv continuous --python=same --launch-method spawn HEAD~1 HEAD
```

**Always pass `--launch-method spawn`.** ASV defaults to a forkserver, which
`fork()`s a process that has already initialized a SYCL runtime; the SYCL
runtime is multi-threaded and not fork-safe, so benchmarks may hang until
`default_benchmark_timeout` expires (reported as `failed`) or fail with
`USM Allocation` errors on `level_zero` devices. `spawn` starts a fresh
interpreter per benchmark and avoids this entirely.

By default, dpnp selects a default SYCL device. Use the `ONEAPI_DEVICE_SELECTOR`
environment variable to target a specific device, e.g.:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu asv run \
    --launch-method spawn \
    --python=same
```

## Benchmarks

### `bench_dpbench.py` -- dpBench workloads

`bench_dpbench.py` runs a set of dpnp workloads vendored from
[dpBench](https://github.com/IntelPython/dpbench). The kernels, their data
initialization, and the data-size presets are copied from dpBench and live in
`benchmarks/dpbench/workloads`. Each workload is exposed as its own benchmark
class (e.g. `BlackScholes.time_black_scholes`) and is parametrized by the
dpBench data-size preset (`S`, `M16Gb`, `M`, `L`) and by floating-point
precision (`single`, `double`).

Currently vendored workloads:

| Workload            | Domain             |
| ------------------- | ------------------ |
| `black_scholes`     | Finance            |
| `l2_norm`           | Distance Compute   |
| `pairwise_distance` | Distance Compute   |
| `rambo`             | Particle Physics   |
| `gpairs`            | Astrophysics       |

Host input data is generated and copied to the device exactly the way dpBench
does, and each kernel ends with `dpnp.synchronize_array_data`, so a single call
blocks until the device work has finished. The `time_*` methods invoke the
workload once and let ASV wall-clock-time it (handling repeats, samples and
statistics natively) -- the same end-to-end quantity dpBench itself measures,
and the same plain `time_*` style used by the mkl_fft ASV benchmarks.

**Precision.** Both `single` and `double` are benchmarked. Devices without fp64
support (common on iGPUs) skip the `double` parametrization via
`SkipNotImplemented` rather than failing the run, so such a device still
produces `single`-precision results. dpBench's own configs request `double`
throughout; that value is kept in each workload's `PRECISION` for reference.

**Preset selection.** Presets are chosen per device instead of being hard-coded:
`_dpbench_runner.select_presets` keeps every preset whose estimated peak device
footprint (each workload's `peak_elements`) fits within a fraction of the
device's `global_mem_size`. A large discrete GPU therefore exercises the bigger
problem sizes automatically, while a small iGPU stays on `S`. Note that dpBench's
preset names are not ordered by size -- `M16Gb` is *smaller* than `M`.

Prefer the largest preset your device fits when looking for regressions. The
smallest sizes are dominated by per-call dispatch overhead and are noticeably
noisier: on a CPU device the run-to-run spread of the median at `S` was measured
at 4-14%, against the 20% `regressions_thresholds` in `asv.conf.json`, whereas
the larger presets settled to a few percent. Timings at `S` are still useful for
a quick smoke test, and ASV's repeat/sample handling absorbs part of the noise.

**Validation.** Each workload also ships the NumPy `reference` implementation
from dpBench, and every benchmark's `setup` compares the dpnp results for all
`OUTPUT_ARGS` against it (mirroring dpBench's
`infrastructure/benchmark_validation.py`, same `1e-05` relative-error
tolerance). A numerically wrong kernel therefore fails the benchmark instead of
being silently timed. Validation runs outside the timed region and does not
affect the reported numbers.

### Other benchmark modules

The remaining `bench_*.py` modules (`bench_linalg.py`, `bench_elementwise.py`,
`bench_random.py`) are plain ASV benchmarks comparing dpnp against NumPy.

## Writing new benchmarks

Read ASV's guidelines for writing benchmarks
[here](https://asv.readthedocs.io/en/stable/writing_benchmarks.html).

To add another dpBench workload, copy its `<name>_dpnp.py` kernel,
`<name>_numpy.py` reference (as `reference`) and `<name>_initialize.py`
initializer into a new module under `benchmarks/dpbench/workloads`, translate its
`bench_info` TOML presets into the module's `PRESETS` and argument-metadata
constants, add a `peak_elements` estimate (see the existing workloads for the
exact shape), and add the module to `WORKLOADS` in
`benchmarks/dpbench/workloads/__init__.py`. `bench_dpbench.py` will generate a
benchmark class for it automatically.
