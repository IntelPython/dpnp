# AGENTS.md - dpnp.tensor (dpnp/tensor/)

## Purpose
`dpnp.tensor` is the Array API namespace with `usm_ndarray`, migrated from `dpctl.tensor` in 0.20.0. `dpnp.ndarray` and most dpnp functions are built on it.

## Layout
- `*.py`: Array API functions; `_elementwise_funcs.py` holds the elementwise function objects
- `*.pyx`: Cython modules, all built by glob (`*.pxd`: their declarations). `_usmarray.pyx` defines `usm_ndarray`; the `public` and `api` declarations in it and `_usmarray.pxd` generate `dpnp/include/dpnp/tensor/_usmarray*.h`
- C++ kernels and bindings (`_tensor_impl`, `_tensor_elementwise_impl`, `_tensor_reductions_impl`, `_tensor_sorting_impl`, `_tensor_accumulation_impl`, `_tensor_linalg_impl`) are in `dpnp/libtensor/` (see `dpnp/libtensor/AGENTS.md`) and imported as `dpnp.libtensor._tensor_*_impl`
- Tests: `dpnp/tests/tensor/`

## Rules for Changes
- `dpnp.tensor` follows the Array API standard, with no NumPy extensions; `dpnp` builds NumPy compatibility on top of it and also stays Array API conformant. Put NumPy-only behaviour in the `dpnp` layer unless the kernel itself is wrong.
- A libtensor change also changes `dpnp`: run `dpnp.tests.tensor` and the affected `dpnp/tests/test_*.py`.
- C++ rules are in `dpnp/libtensor/AGENTS.md`.
- New elementwise function: kernel in `dpnp/libtensor/include/kernels/elementwise_functions/<name>.hpp`; `populate_<name>_dispatch_vectors` (unary) or `populate_<name>_dispatch_tables` (binary) and `init_<name>` (`true_divide.cpp` has `init_divide`) in `dpnp/libtensor/source/elementwise_functions/<name>.cpp`; register it in `dpnp/libtensor/source/elementwise_functions/elementwise_common.cpp`; list the source in `_elementwise_sources` in `dpnp/libtensor/source/CMakeLists.txt`; add the Python object to `_elementwise_funcs.py` and export it in `__init__.py`.
- Tests in `dpnp/tests/tensor/` get a queue with `get_queue_or_skip()` and skip unsupported dtypes with `skip_if_dtype_not_supported(dt, q)` (`dpnp/tests/tensor/helper/`).

## Verification
```bash
python -m pytest -n auto -ra --pyargs dpnp.tests.tensor
```
The Array API conformance job in `conda-package.yml` tests only `dpnp` (`ARRAY_API_TESTS_MODULE: 'dpnp'`); a check for `dpnp.tensor` itself is not added yet, so verify spec compliance of `dpnp.tensor` changes against the spec. A libtensor change can still move the `dpnp` results, and the job never fails the build, so read its PR comment (posted for PRs from IntelPython/dpnp branches only, and only when the tests produced a report) or its log.
