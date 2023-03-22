# Building dpnp_algo_random Cython extension

build_dpnp_cython_ext_with_backend(
  dpnp_algo_random
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_random.pyx
  dpnp/random
  )
