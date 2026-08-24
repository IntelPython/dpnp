# dpnp ASV Benchmarks

Performance benchmarks for [dpnp](https://github.com/IntelPython/dpnp) using
[Airspeed Velocity (ASV)](https://asv.readthedocs.io/en/stable/).

## Coverage

| File | API | Benchmarks | Params | Sizes |
|------|-----|------------|--------|-------|
| `bench_dpbench.py` | `dpnp` (end-to-end workloads) | `BlackScholes`, `L2Norm`, `PairwiseDistance`, `Rambo`, `Gpairs` | `preset`, `precision` | dpBench presets `S`, `M16Gb`, `M`, `L` |
| `bench_elementwise.py` | `dpnp` vs `numpy` | `Elementwise` (26 unary math functions) | `executor`, `size`, `dtype` | 2^16, 2^20, 2^24 |
| `bench_linalg.py` | `dpnp` vs `numpy` (`dot`, `matmul`, `inner`, `einsum`) | `MatMul` | `executor`, `order`, `dtype` | 16 to 1024 square |
| `bench_random.py` | `dpnp.random` vs `numpy.random` | `Sample` (`rand`, `randn`, `random_sample`) | `executor`, `size` | 2^16, 2^20, 2^24 |

### dpBench workloads

`bench_dpbench.py` runs a set of dpnp workloads derived from
[dpBench](https://github.com/IntelPython/dpbench), which live in
`benchmarks/benchmarks/dpbench/workloads`. They measure the end-to-end time of a
whole workload rather than of an individual API call.

| Workload            | Domain             |
| ------------------- | ------------------ |
| `black_scholes`     | Finance            |
| `l2_norm`           | Distance Compute   |
| `pairwise_distance` | Distance Compute   |
| `rambo`             | Particle Physics   |
| `gpairs`            | Astrophysics       |

Host input data is generated and copied to the device the way dpBench does, and
each kernel ends with `dpnp.synchronize_array_data`, so a single call blocks
until the device work has finished. dpBench is not a dependency. See
[`benchmarks/dpbench/README.md`](benchmarks/dpbench/README.md) for the
source-to-module mapping and the intended differences.

## Device and precision

dpnp allocates on the default SYCL device. Use `ONEAPI_DEVICE_SELECTOR` to
target a specific one:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu asv run \
    --python=same \
    --launch-method spawn \
    --quick
```

**The parameter matrix is the same on every machine.** All four dpBench presets
are declared statically, so a given benchmark has the same parameter set
everywhere and results are comparable across devices and across the CI pool.
What varies per device is which of those points *run*: `setup` calls
`_dpbench_runner.preset_fits` and raises `SkipNotImplemented` for any preset
whose estimated peak element count (the workload's `peak_elements`, taken at the
wider of the two precisions) exceeds **0.25** of the device's `global_mem_size`.
So a large discrete GPU exercises the bigger problem sizes automatically while a
small iGPU reports `S` and skips the rest, and a skipped point stays visible as a
skip rather than vanishing from the matrix.

The cheapest preset always runs. If even that does not fit, it is attempted
anyway so the failure is a loud allocation error rather than silence.

Note that dpBench's preset names are not ordered by size: `M16Gb` is *smaller*
than `M` for every workload except `rambo`, where it is larger and equal to `L`.
Anything that needs the cheapest preset sorts explicitly rather than relying on
declaration order.

**Both precisions are benchmarked.** Devices without fp64 support (common on
iGPUs) skip the `double` points via `SkipNotImplemented` rather than failing, so
such a device still produces `single` results. The `float64` points of
`bench_elementwise.py` and `bench_linalg.py` skip the same way for the `dpnp`
executor; the `numpy` executor is unaffected. dpBench's own configs request
`double` throughout, and that value is kept in each workload's `PRECISION` for
reference.

No benchmark module opens a SYCL queue at import time, so benchmark discovery
and `asv check` work on a machine with no usable device; only `setup` needs one.

One caveat on comparability: ASV keys results by machine, commit and
environment, not by device. Benchmarking two devices on the same host therefore
overwrites one set of results with the other. Give each device its own machine
name when you do that:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu asv run --python=same \
    --launch-method spawn --machine "$(hostname)-gpu"
```

## Notes on Measurement

### Process launch method

**Always pass `--launch-method spawn`.** ASV defaults to a forkserver, which
`fork()`s a process that has already initialized a SYCL runtime; the SYCL
runtime is multi-threaded and not fork-safe, so benchmarks may hang until
`default_benchmark_timeout` expires (reported as `failed`) or fail with
`USM Allocation` errors on `level_zero` devices. `spawn` starts a fresh
interpreter per benchmark and avoids this entirely.

### Asynchronous execution

**Every timed body that runs dpnp work must block on it.** dpnp enqueues to a
SYCL queue and returns before the kernel has run, so a body that does not block
measures submission rather than execution. Unsynchronized, a 1024x1024 float32
`dot` measured **0.4 ms** against **36 ms** synchronized on a CPU device -- which
would have reported dpnp as an order of magnitude faster than NumPy on work
where it is in fact slightly slower.

The dpBench workloads each end with `dpnp.synchronize_array_data`, and the
comparison suites obtain a synchronizer from `_utils.make_synchronizer` in
`setup` and pass every result through it (`self.sync(...)`). For the `numpy`
executor the synchronizer does nothing.

### First-call costs

The first call on a fresh queue pays SYCL kernel/JIT and allocator warmup.
`WorkloadRunner.setup` therefore runs each workload once before ASV starts
timing it, so the dpBench suite is warmed explicitly. The `bench_elementwise.py`,
`bench_linalg.py` and `bench_random.py` suites do **not** warm up and rely on
ASV's default `warmup_time`.

### Validation

Each workload ships the NumPy `reference` implementation from dpBench. On the
cheapest preset, `setup` compares the dpnp results for all `OUTPUT_ARGS`
against it (mirroring dpBench's `infrastructure/benchmark_validation.py`, same
`1e-05` relative-error tolerance). A numerically wrong kernel therefore fails
the benchmark instead of being silently timed. Validation runs outside the
timed region and does not affect the reported numbers.

Only the cheapest preset is validated: the reference runs on the host and at the
larger presets costs far more than the benchmark it guards -- tens of seconds
for `pairwise_distance` at `M16Gb` -- while checking numerics that do not depend
on the problem size.

### Noise at small presets

The smallest sizes are dominated by per-call dispatch overhead and are
noticeably noisier. On a CPU device the run-to-run spread of the median at `S`
was measured between **2%** and **25%** across workloads, against the **20%**
`regressions_thresholds` in `asv.conf.json`, whereas the larger presets settled
to a few percent. Treat `S` as a smoke-test size only and do not use it for
regression gating; prefer the largest preset the device fits.

## Running Benchmarks

ASV cannot build dpnp -- it is a SYCL/DPC++ extension that requires the Intel
oneAPI compiler and a lengthy build -- so the benchmarks always run against an
**existing environment** that already has dpnp installed. A bare `asv run` is
not supported; always pass `--python=same` or `--environment existing:<python>`.

Create an environment
[following these instructions](https://intelpython.github.io/dpnp/quick_start_guide.html),
then install the benchmarking tooling into it:

```bash
conda install -c conda-forge asv scipy
```

`scipy` is needed because `scipy.special.erf` is used by the NumPy reference
that the `black_scholes` benchmark validates its dpnp results against.

Do **not** use `pip install ".[benchmark]"` for an environment that already has
dpnp: dpnp is a scikit-build project, so pip reinstalls the `dpnp` package
itself and triggers a full oneAPI/DPC++ rebuild of the backend just to pull in
two pure-Python dependencies. The `benchmark` extra in `pyproject.toml` records
those two dependencies for the case where dpnp is being built from source
anyway; note that the usual editable-install invocation passes `--no-deps`, so
it does *not* install them:

```bash
pip install --no-build-isolation --no-deps -e .
conda install -c conda-forge asv scipy
```

All commands below are run from the `benchmarks/` directory, where
`asv.conf.json` lives.

Register the machine once. Without this a non-interactive or CI run aborts with
`No information stored about machine`:

```bash
asv machine --yes
```

Validate the whole suite without running it. This is cheap and catches broken
signatures and import errors; it accepts no `--bench`, so it is all-or-nothing:

```bash
asv check --python=same
```

Smoke-run the benchmarks, optionally scoped with `--bench`:

```bash
asv run --python=same --launch-method spawn --quick --bench bench_dpbench
```

This only *prints* results. Without `--set-commit-hash` ASV discards them, so
`asv compare` and `asv publish` will see nothing.

To record results, assert which revision the installed dpnp corresponds to:

```bash
asv run --python=same --launch-method spawn --set-commit-hash HEAD
```

ASV does not verify that claim -- it is your assertion -- and the
`For dpnp commit ...` progress line prints the branch head rather than the
value passed, so trust the result filename or `asv show`.

Pointing ASV at an interpreter explicitly works too:

```bash
asv run --environment existing:/full/conda/path/envs/dpnp_env/bin/python \
    --launch-method spawn
```

`asv.conf.json` sets `branches` to `HEAD` rather than to named branches, so that
results recorded on a feature branch are picked up. With named branches
`asv publish` reports `Couldn't find <hash> in branches (...)` and silently
drops them.

### Comparing two revisions

`asv continuous` and any `<commit>` range spec cannot be used here: ASV refuses
a range spec when it cannot install the project into the environment. Compare
two recorded runs instead. Rebuild dpnp in the same environment between them,
and omit `--quick` so the statistics path engages:

```bash
# against the old build
asv run --python=same --launch-method spawn --set-commit-hash <old-rev>
# rebuild/reinstall dpnp, then
asv run --python=same --launch-method spawn --set-commit-hash <new-rev>
asv compare <old-rev> <new-rev>
```

View recorded results in a browser:

```bash
asv publish
asv preview
```

## Writing new benchmarks

Read ASV's guidelines for writing benchmarks
[here](https://asv.readthedocs.io/en/stable/writing_benchmarks.html).

Parameter axes shared by two or more `bench_*` modules live in `_utils.py`;
single-use axes stay in the module that needs them. Two rules keep results
usable:

* Keep parameter values plain strings, numbers or tuples. A live module or dtype
  object renders as `<module 'dpnp' from '/...'>` in the result tables and embeds
  a local path in the result identity.
* Keep `params` static. Deriving an axis from the machine makes rows
  incomparable between devices; decide per-device behaviour in `setup` instead,
  by raising `SkipNotImplemented` (see `bench_dpbench._Workload.setup`).
* Block on dpnp work inside the timed body, or you are timing submission -- see
  [Asynchronous execution](#asynchronous-execution).

To add another dpBench workload, follow
[`benchmarks/dpbench/README.md`](benchmarks/dpbench/README.md), then add a
benchmark class for it to `bench_dpbench.py`. Copy an existing one: it is a
banner, a docstring, a `WORKLOAD` attribute and a one-line `time_*` method --
the parameter axes are inherited from `_Workload`.
