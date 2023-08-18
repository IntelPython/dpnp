
set(dpnp_algo_pyx_deps
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_linearalgebra.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_manipulation.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_counting.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_statistics.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_trigonometric.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_sorting.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_arraycreation.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_mathematical.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_searching.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_indexing.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_logic.pxi
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo_special.pxi
  )

build_dpnp_cython_ext_with_backend(
  dpnp_algo
  ${CMAKE_CURRENT_SOURCE_DIR}/dpnp_algo.pyx
  dpnp/dpnp_algo
  )

add_custom_target(_dpnp_algo_deps DEPENDS ${dpnp_algo_pyx_deps})
add_dependencies(dpnp_algo _dpnp_algo_deps)
