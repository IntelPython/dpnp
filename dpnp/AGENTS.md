# AGENTS.md - Python layer (dpnp/)

## Layout
- `dpnp_iface_<category>.py`: public functions, grouped like NumPy's routines reference
- `dpnp_array.py`: `dpnp.ndarray`, a wrapper over `dpnp.tensor.usm_ndarray`; `get_array()` unwraps, `_create_from_usm_ndarray()` wraps
- `dpnp_algo/dpnp_elementwise_common.py`: ufunc wrapper classes (`DPNPUnaryFunc`, `DPNPBinaryFunc`, two-output and specialised variants)
- `dpnp_utils/`: shared helpers (`get_usm_allocations`, `map_dtype_to_device`, reduction wrappers)
- `linalg/`, `fft/`, `random/`, `scipy/`, `lib/`: subpackages, each exporting through its own `__init__.py`
- `random/`: `_bit_generator.py` (`BitGenerator`, `Philox4x3210`, `PCG64DXSM`, `MRG32k3a`; `SeedSequence` is NumPy's), `_generator.py` (`Generator`, which implements every distribution), `dpnp_random_state.py` (`RandomState`, the same samplers with NumPy's legacy checks) and `dpnp_iface_random.py` (module functions, one `RandomState` per queue). The kernels are in `dpnp.libtensor._rng_impl`.
- Legacy, don't extend: the Cython `dpnp_algo` path and `dpnp_backend_c` (still used by `dpnp.partition`). New kernels go into `dpnp/libtensor` (see `dpnp/libtensor/AGENTS.md`).

## Adding a Public Function
1. Implement it in the matching `dpnp_iface_*.py` or subpackage module. Iface modules have no `__all__`.
2. Export it in `dpnp/__init__.py` in two places, both under its NumPy routine category: the `from .dpnp_iface_<category> import (...)` block and the `__all__ += [...]` block. Functions in `linalg`, `fft`, `random`, `scipy.*` and `lib` are exported from that subpackage's `__init__.py` instead (import block and `__all__`). A new subpackage also goes in `packages=` in `setup.py`.
3. Add it to `doc/reference/` (see `doc/AGENTS.md`), tests (see `dpnp/tests/AGENTS.md`) and `CHANGELOG.md`.

## Patterns
- Ufuncs are module-level wrapper instances, e.g. `cbrt = DPNPUnaryFunc("cbrt", ti._cbrt_result_type, ti._cbrt, _CBRT_DOCSTRING, mkl_fn_to_call=..., mkl_impl_fn=...)`. The first argument is the public name, with no leading underscore. Kernels come from `dpnp.libtensor._tensor_elementwise_impl`; the oneMKL VM path runs first when `mkl_fn_to_call` is given and supported.
- Other functions: `dpnp.check_supported_arrays_type(...)`, unwrap with `dpnp.get_usm_ndarray`, call `dpnp.tensor` (`import dpnp.tensor as dpt`) or an extension, and wrap with `dpnp_array._create_from_usm_ndarray`. Pass `out` straight to the `dpnp.tensor` or extension call when it accepts one; otherwise `dpnp.get_result_array(res, out)` copies the result into it.
- Get the result queue and USM type with `get_usm_allocations([...])` (raises `ValueError` on mixed queues) or `dpt.get_execution_queue(...)` (returns `None`; raise `ExecutionPlacementError`). Creation functions take `device`, `usm_type` and `sycl_queue` (normalised by `dpnp.get_normalized_queue_device`); `*_like` functions inherit them from the input.
- Extension calls are ordered with `_manager = dpctl.utils.SequentialOrderManager[exec_q]`: pass `depends=_manager.submitted_events`, then `_manager.add_event_pair(ht_ev, ev)`. Don't add blocking waits.
- Use `dpnp.default_float_type(...)` or `map_dtype_to_device(...)` for the default float type.
- Deprecated ufuncs use `DPNPDeprecatedUnaryFunc(..., deprecated_msg=...)`.
- A `dpnp.random` distribution is a `Generator` method: it calls `self._check_queues(...)` on its array parameters (raises `ExecutionPlacementError`), checks host parameters as NumPy does, and draws through `_cont`, `_float_sample` or `_discrete_sample` into `BitGenerator._fill`. `RandomState` and the module function delegate to it. Parameters on the device are checked only by `RandomState` (`_check_device`), as the check waits for the device. With `usm_type=None`, the output and the temporaries take the coerced USM type of the array parameters (`Generator._sample`).
- Unsupported NumPy keywords raise `NotImplementedError` (`dpnp.check_limitations`) and are listed under `Limitations` in the docstring.
- The Array API spec version is set in three places that change together: `__array_api_version__` in `dpnp/tensor/_array_api.py`, `ARRAY_API_TESTS_VERSION` in `.github/workflows/conda-package.yml` and `doc/reference/array_api.rst` (plus the version test in `dpnp/tests/test_ndarray.py`). Inspection utilities come from `__array_namespace_info__` in `dpnp_array_api_info.py`.
- `dpnp.tensor` replaces `dpctl.tensor`, which no longer exists here; `SyclQueue`, `SyclDevice`, `dpctl.memory` and `SequentialOrderManager` still come from `dpctl`.

## Docstrings
numpydoc, rendered by `doc/conf.py`. `doc/docstring_template.py` shows the layout; model a new docstring on an existing function.
- Sections in the order `doc/conf.py` renders them: summary, ``For full documentation refer to :obj:`numpy.<name>`.``, Parameters, Returns, Raises, Warns, Warnings, Limitations, See Also, Notes, References, Examples.
- Array parameters are typed `{dpnp.ndarray, usm_ndarray}`, optional ones `{None, dpnp.ndarray, usm_ndarray}, optional`; the return is `out : dpnp.ndarray`.
- Every keyword with a default ends with a blank line and ``Default: ``<value>``.``
- Literals in double backticks (``` ``None`` ```), argument names in single backticks; link with `:obj:`.
- Copy the `device` / `usm_type` / `sycl_queue` parameter text from an existing creation function such as `dpnp.empty`.
- Examples start with `>>> import dpnp as np`. A docstring that contains `\` is raw (`r"""`).
- An `ndarray` method that wraps a function has a one-line summary, a blank line, then ``Refer to :obj:`dpnp.<name>` for full documentation.``
