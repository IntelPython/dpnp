
function(build_dpnp_cython_ext _trgt _src _dest)
  add_cython_target(${_trgt} ${_src} CXX OUTPUT_VAR _generated_src)
  message(STATUS "Using ${_trgt}")
  add_library(${_trgt} MODULE ${_generated_src})
  set(_trgt_deps "${_trgt}_deps")
  add_custom_target(${_trgt_deps} DEPENDS ${_src})
  add_dependencies(${_trgt} ${_trgt_deps})
  if (DPNP_GENERATE_COVERAGE)
    target_compile_definitions(${_trgt} PRIVATE CYTHON_TRACE=1 CYTHON_TRACE_NOGIL=1)
  endif()
  # NumPy
  target_include_directories(${_trgt} PRIVATE ${NumPy_INCLUDE_DIR})
  # Dpctl
  target_include_directories(${_trgt} PRIVATE ${Dpctl_INCLUDE_DIR})
  target_link_directories(${_trgt} PRIVATE ${Dpctl_INCLUDE_DIR}/..)
  target_link_libraries(${_trgt} DPCTLSyclInterface)
  
  set(_linker_options "LINKER:${DPNP_LDFLAGS}")
  target_link_options(${_trgt} PRIVATE ${_linker_options})
  python_extension_module(${_trgt})
  
  install(TARGETS ${_trgt} LIBRARY DESTINATION ${_dest})
endfunction()

function(build_dpnp_cython_ext_with_backend _trgt _src _dest)
  build_dpnp_cython_ext(${_trgt} ${_src} ${_dest})
  target_link_libraries(${_trgt} dpnp_backend_library)
  if (UNIX)
    set_target_properties(${_trgt} PROPERTIES INSTALL_RPATH "$ORIGIN/..")
  endif()
endfunction()


build_dpnp_cython_ext_with_backend(dparray ${CMAKE_CURRENT_SOURCE_DIR}/dparray.pyx dpnp)
add_subdirectory(backend)

add_subdirectory(dpnp_algo)
add_subdirectory(dpnp_utils)
add_subdirectory(fft)
add_subdirectory(linalg)
add_subdirectory(random)

