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
and install the benchmarking tooling into it. Either install the `benchmark`
extra from the repo:

```bash
pip install ".[benchmark]"
```

or install `asv` directly:

```bash
conda install -c conda-forge asv
```

Then activate the environment and run the benchmarks against it. The simplest
way is to point ASV at the currently active environment with `--python=same`:

```bash
conda activate dpnp_env
asv run --python=same --quick HEAD^!
```

Alternatively, point ASV explicitly at an environment's python binary:

```bash
asv run --environment existing:/full/conda/path/envs/dpnp_env/bin/python
```

Compare two commits or check for regressions:

```bash
asv continuous --python=same HEAD~1 HEAD
```

For `level_zero` devices, you might see `USM Allocation` errors unless you use
the `asv run` command with `--launch-method spawn`.

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
dpBench data-size preset (`S`, `M16Gb`, `M`, `L`).

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
and the same plain `time_*` style used by the mkl_fft ASV benchmarks. By
default only the small `S` preset is exercised; edit `ASV_PRESETS` in a workload
module to benchmark larger problem sizes (which may require several GiB of
device memory).

### Other benchmark modules

The remaining `bench_*.py` modules (`bench_linalg.py`, `bench_elementwise.py`,
`bench_random.py`) are plain ASV benchmarks comparing dpnp against NumPy.

## Writing new benchmarks

Read ASV's guidelines for writing benchmarks
[here](https://asv.readthedocs.io/en/stable/writing_benchmarks.html).

To add another dpBench workload, copy its `<name>_dpnp.py` kernel and
`<name>_initialize.py` initializer into a new module under
`benchmarks/dpbench/workloads`, translate its `bench_info` TOML presets into the
module's `PRESETS`/`ASV_PRESETS` and argument-metadata constants (see the
existing workloads for the exact shape), and add the module to `WORKLOADS` in
`benchmarks/dpbench/workloads/__init__.py`. `bench_dpbench.py` will generate a
benchmark class for it automatically.
