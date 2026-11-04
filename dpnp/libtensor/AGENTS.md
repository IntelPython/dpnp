# AGENTS.md - libtensor and pybind11 extensions (dpnp/libtensor/)

## Layout
- `include/kernels/`: header-only SYCL kernels in namespaces under `dpnp::tensor::kernels`; `include/utils/`: type dispatch, type, offset and memory-overlap helpers (`type_dispatch.hpp`, `type_dispatch_building.hpp`, `type_utils.hpp`, `memory_overlap.hpp`, ...); `include/ext/`: dispatch helpers (`init_dispatch_vector`, `init_dispatch_table`, `DispatchTable`) and validation helpers (`validation_utils.hpp`). The headers ship with dpnp; `dpnp.get_include()` and `python -m dpnp --tensor-include-dir` return `include/`.
- `source/`: bindings built by `source/CMakeLists.txt` into `_tensor_impl`, `_tensor_elementwise_impl`, `_tensor_reductions_impl`, `_tensor_sorting_impl`, `_tensor_accumulation_impl`, `_tensor_linalg_impl`, `_statistics_impl` and `_window_impl`. The functions dpnp adds beyond the Array API are listed in the `_dpnp_*_sources` variables there and compiled into the same modules.
- `extensions/{blas,fft,lapack,rng,sparse,vm}/`: oneMKL/oneMath extensions, one module `_<name>_impl` per directory.
- Every module installs into `dpnp/libtensor` and is imported as `dpnp.libtensor._<name>_impl`. A new kernel goes into `source/` and an existing module; only code that needs oneMKL gets a new extension.

## Adding a ufunc Kernel (model: `fabs`)
1. Functor in `include/kernels/elementwise_functions/<name>.hpp`, in namespace `dpnp::tensor::kernels::<name>`.
2. Binding in `source/elementwise_functions/<name>.{hpp,cpp}`: output-type map built from `td_ns::TypeMapResultEntry<...>` ending in `td_ns::DefaultResultEntry<void>`, `MACRO_POPULATE_DISPATCH_VECTORS(<name>)` (`MACRO_POPULATE_DISPATCH_TABLES` for binary) from `populate.hpp`, and `init_<name>(m)` defining `_<name>` and `_<name>_result_type`.
3. Register `init_<name>(m)` in `source/elementwise_functions/elementwise_common.cpp` and add the source to `_dpnp_elementwise_sources` in `source/CMakeLists.txt`, which compiles it with `-fno-approx-func -fno-finite-math-only`; the Array API functions in `_elementwise_sources` get `-fno-fast-math` instead.
4. Wrap it in Python (see `dpnp/AGENTS.md`).

A new extension's `CMakeLists.txt` lists its sources and calls `build_dpnp_pybind11_ext(<module> SOURCES ... [INCLUDE_DIRS ...] [LARGE_RANGES])` from `dpnp/CMakeLists.txt`, which sets the options all extensions share and installs the module into `dpnp/libtensor`; only oneMKL/oneMath linking and the `-DUSE_ONEMATH*` definitions stay in the extension. Add it with `add_subdirectory` in `dpnp/CMakeLists.txt`, and put an option every extension needs into the helper, not into one extension.

## Rules for Changes
- C++17. Include what you use (`<cstdint>`, `<vector>`, ...), use `std::`-qualified fixed-width types (`std::int64_t`, `std::size_t`), make by-value parameters and locals `const` (`constexpr` where possible), and pass non-output parameters as `const T &`.
- Headers use `#pragma once`. New files in `include/` and `source/` add a `/// \file` block after the license header.
- Python-facing functions take `sycl::queue &exec_q` and `const std::vector<sycl::event> &depends`, and return `std::pair<sycl::event, sycl::event>` (keep-alive host-task event, compute event). Keep arguments alive with `dpnp::utils::keep_args_alive`.
- Validate inputs before launching: queue compatibility, contiguity, writability, memory overlap (including the same logical tensor), dtype and ndim. Reuse the helpers neighbouring files use: `check_queue`, `check_c_contig`, `check_no_overlap`, `check_writable` from `include/ext/validation_utils.hpp` (`putmask`, `statistics`, `isclose`, `interpolate`), or `dpnp::utils::queues_are_compatible`, `dpnp::tensor::overlap::MemoryOverlap` and `dpnp::tensor::validation::CheckWritable` (extensions and the Array API functions). Don't silently accept strided input.
- Throw `py::value_error` for bad arguments and `std::runtime_error` for internal failures; catch oneMKL and SYCL exceptions and rethrow them.
- Guard double, complex<double> and half with `dpnp::tensor::type_utils::validate_type_for_device<T>(q)`; don't use double literals or casts in kernels that must run without fp64.
- `blas`, `fft`, `lapack` and `sparse` build against either oneMKL or oneMath (`USE_ONEMATH`); `vm` has kernels only with oneMKL: with oneMath it builds empty and `_is_available()` returns `False`; `rng` uses the header-only oneMKL device RNG API and links no oneMKL library.
- Release the GIL around blocking oneMKL calls (`py::gil_scoped_release`).
- `rng` draws depend only on the engine state and the number of elements, never on the device or the layout chosen for it (`extensions/rng/kernels/sample.hpp`): raw bits and uniform floats are bit-identical across devices, samplers using transcendental functions agree to a few ULP, or differ where a rejection sampler then takes another candidate. Each engine's `items_per_work_item` (`kernels/engines.hpp`) is part of its stream definition. Parameters on the device are unchecked, so a rejection sampler returns NaN or the smallest `int64` for the inputs that would make it loop forever, including values that are valid but round badly in fp32. Use `umul_hi` from `kernels/umul_hi.hpp` instead of `sycl::mul_hi`, which DPC++ 2026.2 miscompiles on Intel GPUs.
- An extension whose kernels may be launched over more than `INT_MAX` work-items passes `LARGE_RANGES` to `build_dpnp_pybind11_ext`, which adds `-fno-sycl-id-queries-fit-in-int` at compile and link time (DPC++ 2026.2 and later need it at link time). `source/CMakeLists.txt` and `dpnp_backend_c` pass it to both `target_compile_options` and `target_link_options`.
- clang-format 23 formats C++ (`.clang-format`: LLVM base, 4-space indent, 80 columns); gersemi formats CMake at 88 columns, except `dpnp/backend/cmake/Modules/`.

## Build
```bash
python scripts/build_locally.py                  # Release, Ninja, icx/icpx, then pip install -e
python scripts/build_locally.py --clean --debug  # remove _skbuild first; Debug build
python scripts/build_locally.py --onemath        # oneMath instead of oneMKL
python scripts/build_locally.py --target-cuda    # plus NVIDIA sm_50 (or --target-cuda=sm_80); --target-hip=<gfx> for AMD
```
`--target-cuda`, `--target-hip` and `--onemath-dir` imply oneMath. Extra CMake options go through `--cmake-opts="..."`; the `--onemkl_interfaces*` options are deprecated. The in-place build copies the `.so` files into the source tree; they are gitignored.
