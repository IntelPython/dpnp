# AGENTS.md - Benchmarks (benchmarks/)

ASV suite that times dpnp only; `README.md` has the full setup.

## Rules for Changes
- NumPy is not timed: `executor` stays the single value `'dpnp'`. Don't edit the leftover `numpy` branches in `setup`: ASV versions a benchmark by the source of `setup` and `time_*`, so any edit makes its results incomparable with the recorded history.
- Keep `params` static and their values plain strings, numbers or tuples; skip unsupported devices in `setup` with `SkipNotImplemented`.
- Timed bodies block on the result (`self.sync(...)` or `dpnp.synchronize_array_data`); otherwise they time only the submission.
- Parameter axes shared by two or more modules live in `benchmarks/benchmarks/_utils.py`.
- A new non-ASV dependency goes in both `requirements.txt` (what CI installs) and the `benchmark` extra in `pyproject.toml`; ASV itself is only in the extra.

## Verification
From `benchmarks/`, against an environment that already has dpnp:
```bash
asv machine --yes
asv check --python=same
asv run --python=same --launch-method spawn --quick --bench bench_elementwise
```
Always pass `--launch-method spawn`: the SYCL runtime is not fork-safe. Don't `pip install ".[benchmark]"`; it rebuilds dpnp.
