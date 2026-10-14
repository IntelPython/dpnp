# AGENTS.md - Tests (dpnp/tests/)

## Layout
- `test_<area>.py`: one file per dpnp module, comparing dpnp with NumPy; `ndarray` methods and attributes go in `test_ndarray.py`
- `test_sycl_queue.py`, `test_usm_type.py`: compute-follows-data, i.e. the result's queue and USM type follow the inputs
- `test_strides.py`: strided and negative-stride inputs; `test_umath.py` tests every `numpy.ufunc` that dpnp has, automatically
- `third_party/cupy/`, `third_party/cupyx/`: CuPy's tests ported with `import dpnp as cupy`; `numpy_cupy_*` decorators run each test with `xp=dpnp` and `xp=numpy`
- `tensor/`: `dpnp.tensor` tests (see `dpnp/tensor/AGENTS.md`)
- `skipped_tests.tbl` (non-GPU default device), `skipped_tests_gpu.tbl` (GPU, or `DPNP_QUEUE_GPU=1`), plus `skipped_tests_cuda.tbl` on CUDA: one exact node ID per line, chosen by `conftest.py`; there is no comment syntax
- `tests_perf/` is never collected

## Rules for Changes
- Compare with `assert_dtype_allclose(result, expected)` and build inputs with `generate_random_numpy_array` (seeded) from `helper.py`. Don't hand-write dtype lists or use unseeded `numpy.random`.
- Compare against NumPy, or SciPy for `dpnp.scipy`. Where the reference's result depends on how it was built, assert the value the spec gives instead and say why in a comment.
- Parametrize dtypes with `get_all_dtypes()`, `get_float_dtypes()`, `get_float_complex_dtypes()` and similar. They drop float64/complex128 on devices without fp64, so build the NumPy and dpnp inputs with the same parametrized dtype. Never hardcode a float64 NumPy reference: it passes on CPU and fails on GPUs without fp64. Guard fp64-only tests with `has_support_aspect64()`.
- `get_all_dtypes()` includes `None`; pass `no_none=True` unless the default dtype is under test. `get_integer_dtypes()` returns only int32/int64 unless `DPNP_TEST_ALL_INT_TYPES` is set.
- `dpnp/tests/__init__.py` replaces `numpy.testing.assert_*` with wrappers that accept dpnp arrays and default to `strict=True` (dtype and shape must match) with NumPy 2 on fp64 devices. Pass `strict=False` only when a dtype difference is expected.
- Don't copy dpnp results to the host just to compare two dpnp arrays.
- Group tests in a `TestXxx` class with shared parametrizations; extend existing tests rather than duplicating them.
- Error cases run on both libraries: parametrize over `xp` in `[dpnp, numpy]` so both must raise. Gate NumPy-version-dependent behaviour with `numpy_version()` (or `testing.with_requires` in CuPy tests).
- Compute-follows-data error tests expect `(ValueError, ExecutionPlacementError)`: dpnp raises either, depending on the helper.
- A new function is added to the matching parametrized lists in `test_sycl_queue.py` and `test_usm_type.py` (`test_1in_1out`, `test_2in_1out`, ...); elementwise ones also to `test_strides.py`.
- A test that may exceed the device's maximum allocation size skips on `USMAllocationError` (from `dpnp.exceptions`), not `MemoryError`, which it does not derive from.
- Markers are strict and only `slow` and `multi_gpu` are registered. Tests that need huge allocations are `@pytest.mark.slow`; they never run in CI by design.
- Skip lists are for device- or backend-specific failures only; use the full parametrized node ID.
- CI runs the installed package (`--pyargs dpnp.tests`), so a new test directory or data file needs a matching glob in `package_data` in `setup.py`, or CI silently skips it.

## CuPy Tests (third_party/)
- They check parity with CuPy and mirror upstream: no dpnp-only tests (those go in `test_*.py`), and code that doesn't apply is commented out, not deleted.
- When a feature lands, remove its `pytest.mark.skip("... is not supported yet")` and any matching `.tbl` entries.
- Use `type_check=has_support_aspect64()` in `numpy_cupy_*` decorators.

## Running
```bash
python -m pytest -ra dpnp/tests/test_mathematical.py                    # one file, in-place build
SKIP_TENSOR_TESTS=1 python -m pytest -n auto -ra --pyargs dpnp.tests    # as CI; -n needs pytest-xdist
python -m pytest -n auto -ra --pyargs dpnp.tests.tensor                  # tensor tests, a separate CI step
python -m pytest -ra -m slow dpnp/tests/test_fft_gil.py                  # slow tests only
python -m pytest -ra -m "slow or not slow" dpnp/tests/test_fft_gil.py     # all tests, slow included
```
- Env flags in `config.py` are off when unset, empty, `0`, `false`, `no` or `off`; any other value turns them on.
- `python -m dpctl -f` lists the available devices.
