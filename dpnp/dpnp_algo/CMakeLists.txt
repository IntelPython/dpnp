
set(dpnp_algo_pyx_deps
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_linearalgebra.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_manipulation.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_counting.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_statistics.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_trigonometric.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_sorting.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_arraycreation.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_mathematical.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_searching.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_indexing.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_logic.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_bitwise.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_special.pyx
  )

build_dpnp_cython_ext_with_backend(
  dpnp_algo
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo.pyx
  dpnp/dpnp_algo
  )

add_custom_target(_dpnp_algo_deps DEPENDS ${dpnp_algo_pyx_deps})
add_dependencies(dpnp_algo _dpnp_algo_deps)
