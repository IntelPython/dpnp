# AGENTS.md - C++ Backend (dpnp/backend/)

## Layout
- `extensions/<name>/`: one pybind11 module `_<name>_impl` per directory (`blas`, `fft`, `indexing`, `lapack`, `sparse`, `statistics`, `ufunc`, `vm`, `window`), imported as `dpnp.backend.extensions.<name>._<name>_impl`
- `extensions/common/ext/`: dispatch helpers (`init_dispatch_vector`, `init_dispatch_table`, `DispatchTable`) and validation helpers (`validation_utils.hpp`)
- `extensions/elementwise_functions/`: copy of the libtensor elementwise Python-API helpers, compiled into `ufunc` and `vm`
- `kernels/{elementwise_functions,indexing,statistics,window}/`: header-only SYCL functors in namespaces under `dpnp::kernels`
- Legacy: `src/`, `include/` and `kernels/dpnp_krnl_*.cpp` build `dpnp_backend_c` (still used by `dpnp.partition`); `tests/` (GTest, `-DDPNP_BACKEND_TESTS=ON`) and `examples/` are not built in CI
- `cmake/Modules/`: vendored `IntelSYCL`, `MKL`, `TBB` and `oneDPL` config modules

## Adding a ufunc Kernel (model: `fabs`)
1. Functor in `kernels/elementwise_functions/<name>.hpp`.
2. Binding in `extensions/ufunc/elementwise_functions/<name>.{hpp,cpp}`: output-type map built from `td_ns::TypeMapResultEntry<...>` ending in `td_ns::DefaultResultEntry<void>`, `MACRO_POPULATE_DISPATCH_VECTORS(<name>)`, and `init_<name>(m)` defining `_<name>` and `_<name>_result_type`.
3. Register `init_<name>(m)` in `extensions/ufunc/elementwise_functions/common.cpp` and add the source to `_elementwise_sources` in `extensions/ufunc/CMakeLists.txt`.
4. Wrap it in Python (see `dpnp/AGENTS.md`).

A new extension's `CMakeLists.txt` lists its sources and calls `build_dpnp_pybind11_ext(<module> <install dir> SOURCES ... [INCLUDE_DIRS ...] [LARGE_RANGES])` from `dpnp/CMakeLists.txt`, which sets the options all extensions share; only oneMKL/oneMath linking and the `-DUSE_ONEMATH*` definitions stay in the extension. Add it with `add_subdirectory` in `dpnp/CMakeLists.txt`, and put an option every extension needs into the helper, not into one extension.

## Rules for Changes
These also apply to `dpnp/tensor/libtensor/`.
- C++17. Include what you use (`<cstdint>`, `<vector>`, ...), use `std::`-qualified fixed-width types (`std::int64_t`, `std::size_t`), make by-value parameters and locals `const` (`constexpr` where possible), and pass non-output parameters as `const T &`.
- Headers use `#pragma once`.
- Python-facing functions take `sycl::queue &exec_q` and `const std::vector<sycl::event> &depends`, and return `std::pair<sycl::event, sycl::event>` (keep-alive host-task event, compute event). Keep arguments alive with `dpnp::utils::keep_args_alive`.
- Validate inputs before launching: queue compatibility, contiguity, writability, memory overlap (including the same logical tensor), dtype and ndim. Reuse the helpers neighbouring files use: `check_queue`, `check_c_contig`, `check_no_overlap`, `check_writable` from `extensions/common/ext/validation_utils.hpp` (`indexing`, `statistics`, `ufunc`), or `dpnp::utils::queues_are_compatible`, `dpnp::tensor::overlap::MemoryOverlap` and `dpnp::tensor::validation::CheckWritable` (other extensions and libtensor). Don't silently accept strided input.
- Throw `py::value_error` for bad arguments and `std::runtime_error` for internal failures; catch oneMKL and SYCL exceptions and rethrow them.
- Guard double, complex<double> and half with `dpnp::tensor::type_utils::validate_type_for_device<T>(q)`; don't use double literals or casts in kernels that must run without fp64.
- `blas`, `fft`, `lapack` and `sparse` build against either oneMKL or oneMath (`USE_ONEMATH`); `vm` has kernels only with oneMKL: with oneMath it builds empty and `_is_available()` returns `False`.
- Release the GIL around blocking oneMKL calls (`py::gil_scoped_release`).
- An extension whose kernels may be launched over more than `INT_MAX` work-items passes `LARGE_RANGES` to `build_dpnp_pybind11_ext`, which adds `-fno-sycl-id-queries-fit-in-int` at compile and link time (DPC++ 2026.2 and later need it at link time). Outside the helper (`dpnp_backend_c`, `dpnp/tensor`) pass it to both `target_compile_options` and `target_link_options`.
- clang-format 23 formats C++ (`.clang-format`: LLVM base, 4-space indent, 80 columns); gersemi formats CMake at 88 columns, except `cmake/Modules/`.

## Build
```bash
python scripts/build_locally.py                  # Release, Ninja, icx/icpx, then pip install -e
python scripts/build_locally.py --clean --debug  # remove _skbuild first; Debug build
python scripts/build_locally.py --onemath        # oneMath instead of oneMKL
python scripts/build_locally.py --target-cuda    # plus NVIDIA sm_50 (or --target-cuda=sm_80); --target-hip=<gfx> for AMD
```
`--target-cuda`, `--target-hip` and `--onemath-dir` imply oneMath. Extra CMake options go through `--cmake-opts="..."`; the `--onemkl_interfaces*` options are deprecated. The in-place build copies the `.so` files into the source tree; they are gitignored.
