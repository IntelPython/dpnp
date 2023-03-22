# Building dpnp_algo_linalg Cython extension

build_dpnp_cython_ext_with_backend(
  dpnp_algo_linalg
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_linalg.pyx
  dpnp/linalg
  )
