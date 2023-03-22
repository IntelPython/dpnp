# Building dpnp_algo_utils Cython extension

build_dpnp_cython_ext_with_backend(
  dpnp_algo_utils
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_utils.pyx
  dpnp/dpnp_utils
  )
