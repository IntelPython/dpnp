# Building dpnp_algo_fft Cython extension

build_dpnp_cython_ext_with_backend(
  dpnp_algo_fft
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_fft.pyx
  dpnp/fft
  )
