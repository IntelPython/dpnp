
function(build_dpnp_cython_ext _trgt _src _dest)
  set(options SYCL)
  cmake_parse_arguments(BUILD_DPNP_EXT "${options}" "" "" ${ARGN})
  add_cython_target(${_trgt} ${_src} CXX OUTPUT_VAR _generated_src)
  message(STATUS "Using ${_trgt}")

  Python_add_library(${_trgt} MODULE WITH_SOABI ${_generated_src})
  set(_trgt_deps "${_trgt}_deps")
  add_custom_target(${_trgt_deps} DEPENDS ${_src})
  add_dependencies(${_trgt} ${_trgt_deps})

  if (BUILD_DPNP_EXT_SYCL)
    add_sycl_to_target(TARGET ${_trgt} SOURCES ${_generated_src})
  endif()

  if (DPNP_GENERATE_COVERAGE)
    target_compile_definitions(${_trgt} PRIVATE CYTHON_TRACE=1 CYTHON_TRACE_NOGIL=1)
    target_compile_options(${_trgt} PRIVATE "-fno-sycl-use-footer")
  endif()

  # NumPy
  target_compile_definitions(${_trgt} PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
  target_include_directories(${_trgt} PRIVATE ${NumPy_INCLUDE_DIR})

  # Dpctl
  target_include_directories(${_trgt} PRIVATE ${Dpctl_INCLUDE_DIR})
  target_link_directories(${_trgt} PRIVATE ${Dpctl_INCLUDE_DIR}/..)
  target_link_libraries(${_trgt} PRIVATE DPCTLSyclInterface)

  set(_linker_options "LINKER:${DPNP_LDFLAGS}")
  target_link_options(${_trgt} PRIVATE ${_linker_options})

  if (DPNP_GENERATE_COVERAGE)
       set(_copy_cxx_trgt "${_trgt}_copy_cxx")
       add_custom_target(
           ${_copy_cxx_trgt} ALL COMMAND ${CMAKE_COMMAND}
           -DSOURCE_FILE=${_generated_src}
           -DDEST=${CMAKE_CURRENT_SOURCE_DIR}
           -P ${CMAKE_SOURCE_DIR}/dpnp/cmake/copy_existing.cmake
           DEPENDS ${_trgt}
           VERBATIM COMMENT "Copying Cython-generated source for target ${_trgt} to dpnp source layout"
       )
  endif()
  install(TARGETS ${_trgt} LIBRARY DESTINATION ${_dest})
endfunction()

function(build_dpnp_cython_ext_with_backend _trgt _src _dest)
  build_dpnp_cython_ext(${_trgt} ${_src} ${_dest})
  target_link_libraries(${_trgt} PRIVATE dpnp_backend_library)
  if (UNIX)
    set_target_properties(${_trgt} PROPERTIES INSTALL_RPATH "$ORIGIN/..")
  endif()
endfunction()


build_dpnp_cython_ext_with_backend(dparray ${CMAKE_CURRENT_SOURCE_DIR}/dparray.pyx dpnp)
add_subdirectory(backend)
add_subdirectory(backend/extensions/blas)
add_subdirectory(backend/extensions/lapack)
add_subdirectory(backend/extensions/vm)
add_subdirectory(backend/extensions/sycl_ext)

add_subdirectory(dpnp_algo)
add_subdirectory(dpnp_utils)
add_subdirectory(fft)
add_subdirectory(linalg)
add_subdirectory(random)
