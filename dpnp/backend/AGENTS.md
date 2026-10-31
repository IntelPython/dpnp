# AGENTS.md - Legacy C++ Backend (dpnp/backend/)

Legacy, don't extend: new kernels and bindings go into `dpnp/libtensor` (see `dpnp/libtensor/AGENTS.md`, which also holds the C++ rules and build options).

## Layout
- `src/`, `include/` and `kernels/dpnp_krnl_*.cpp` build `dpnp_backend_c` (still used by `dpnp.partition`); `tests/` (GTest, `-DDPNP_BACKEND_TESTS=ON`) and `examples/` are not built in CI
- `cmake/Modules/`: vendored `IntelSYCL`, `MKL`, `TBB` and `oneDPL` config modules, used by the whole build
