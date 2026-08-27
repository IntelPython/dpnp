## dpBench-derived workloads

The modules under `workloads/` reproduce benchmarks from
[dpBench](https://github.com/IntelPython/dpbench), so that dpnp is measured on
the same quantity: the end-to-end time of a whole workload rather than of a
single API call. dpBench is not a dependency; `_dpbench_runner.py` re-implements
the parts ASV needs (data initialization, host-to-device transfer, execution and
reference validation).

Reference version: dpBench `0.2.0+79.g4501644`.

Per workload, three modules from `dpbench/benchmarks/default/<name>/` and one
config from `dpbench/configs/bench_info/` map onto one module here:

| dpnp module                      | dpBench sources                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| `workloads/black_scholes.py`     | `black_scholes_{dpnp,numpy,initialize}.py`, `black_scholes.toml`                     |
| `workloads/l2_norm.py`           | `l2_norm_{dpnp,numpy,initialize}.py`, `l2_norm.toml`                                 |
| `workloads/pairwise_distance.py` | `pairwise_distance_{dpnp,numpy,initialize}.py`, `pairwise_distance.toml`             |
| `workloads/rambo.py`             | `rambo_{dpnp,numpy,initialize}.py`, `rambo.toml`                                     |
| `workloads/gpairs.py`            | `gpairs_{dpnp,numpy,initialize}.py`, `gpairs.toml`                                   |

`<name>_dpnp.py` became `<NAME>()`, `<name>_numpy.py` became `reference()` and
`<name>_initialize.py` became `initialize()`. From the TOML, `[benchmark]` gives
`INPUT_ARGS` / `ARRAY_ARGS` / `OUTPUT_ARGS`, `[benchmark.init]` gives
`INIT_INPUT_ARGS` / `INIT_OUTPUT_ARGS` / `PRECISION`, and
`[benchmark.parameters.*]` gives `PRESETS`.

### Intended differences

1. `black_scholes` calls `dpnp.scipy.special.erf`; dpBench calls `dpnp.erf`.
2. Every kernel ends with `dpnp.synchronize_array_data(<output>)`. ASV times the
   `time_*` method directly, so the kernel has to block or only host-side
   dispatch is measured.
3. `rambo.initialize` draws its random block in one `numpy.random.rand` call
   rather than element by element. This consumes the same RNG stream in the same
   order, so the data is bit-identical, but it is far faster -- which matters
   because ASV re-runs `setup` for every round.
4. `peak_elements(params)` is new: it estimates a preset's peak element count so
   `_dpbench_runner.preset_fits` can skip presets too large for the device.

### Adding a workload

Add a module under `workloads/` exposing the same interface as the existing ones,
translate its `bench_info` TOML into the metadata constants, add a
`peak_elements` estimate, and record it in the table above.

Then add a benchmark class to `bench_dpbench.py` -- that is what puts the
workload into the suite. `WORKLOADS` in `workloads/__init__.py` is only a
registry; adding to it alone has no effect.
