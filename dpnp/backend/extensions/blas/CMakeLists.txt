# *****************************************************************************
# Copyright (c) 2024-2025, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


set(python_module_name _blas_impl)
set(_module_src
    ${CMAKE_CURRENT_SOURCE_DIR}/blas_py.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/syrk.cpp
)

pybind11_add_module(${python_module_name} MODULE ${_module_src})
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})

if(_dpnp_sycl_targets)
    # make fat binary
    target_compile_options(
        ${python_module_name}
        PRIVATE
        ${_dpnp_sycl_target_compile_options}
    )
    target_link_options(
        ${python_module_name}
        PRIVATE
        ${_dpnp_sycl_target_link_options}
    )
endif()

if (WIN32)
    if (${CMAKE_VERSION} VERSION_LESS "3.27")
        # this is a work-around for target_link_options inserting option after -link option, cause
        # linker to ignore it.
        set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -fsycl-device-code-split=per_kernel")
    endif()
endif()

set_target_properties(${python_module_name} PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE ON)

target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_include_directories(${python_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
target_include_directories(${python_module_name} PUBLIC ${Dpctl_TENSOR_INCLUDE_DIR})

if (WIN32)
  target_compile_options(${python_module_name} PRIVATE
    /clang:-fno-approx-func
    /clang:-fno-finite-math-only
    )
else()
  target_compile_options(${python_module_name} PRIVATE
    -fno-approx-func
    -fno-finite-math-only
    )
endif()

target_link_options(${python_module_name} PUBLIC -fsycl-device-code-split=per_kernel)

if (DPNP_GENERATE_COVERAGE)
    target_link_options(${python_module_name} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

if(_ues_onemath)
    target_link_libraries(${python_module_name} PRIVATE ${ONEMATH_LIB})
    target_compile_options(${python_module_name} PRIVATE -DUSE_ONEMATH)
    if(_ues_onemath_cuda)
        target_compile_options(${python_module_name} PRIVATE -DUSE_ONEMATH_CUBLAS)
    endif()
else()
    target_link_libraries(${python_module_name} PUBLIC MKL::MKL_SYCL::BLAS)
endif()

if (DPNP_WITH_REDIST)
    set_target_properties(${python_module_name} PROPERTIES INSTALL_RPATH "$ORIGIN/../../../../../../")
endif()

install(TARGETS ${python_module_name}
  DESTINATION "dpnp/backend/extensions/blas"
)
