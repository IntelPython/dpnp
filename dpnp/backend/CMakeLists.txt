# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

set(DPNP_SRC
    kernels/dpnp_krnl_arraycreation.cpp
    kernels/dpnp_krnl_bitwise.cpp
    kernels/dpnp_krnl_common.cpp
    kernels/dpnp_krnl_elemwise.cpp
    kernels/dpnp_krnl_fft.cpp
    kernels/dpnp_krnl_indexing.cpp
    kernels/dpnp_krnl_linalg.cpp
    kernels/dpnp_krnl_logic.cpp
    kernels/dpnp_krnl_manipulation.cpp
    kernels/dpnp_krnl_mathematical.cpp
    kernels/dpnp_krnl_random.cpp
    kernels/dpnp_krnl_reduction.cpp
    kernels/dpnp_krnl_searching.cpp
    kernels/dpnp_krnl_sorting.cpp
    kernels/dpnp_krnl_statistics.cpp
    src/constants.cpp
    src/dpnp_iface_fptr.cpp
    src/memory_sycl.cpp
    src/queue_sycl.cpp
    src/verbose.cpp
    src/dpnp_random_state.cpp
    )

find_package(MKL REQUIRED)
find_package(oneDPL REQUIRED)

set(_trgt "dpnp_backend_c")

add_library(${_trgt} SHARED ${DPNP_SRC})
set_target_properties(${_trgt} PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE ON)

target_include_directories(${_trgt} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${_trgt} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_options(${_trgt} PUBLIC -fsycl-device-code-split=per_kernel)
if(UNIX)
    # this option is support on Linux only
    target_link_options(${_trgt} PUBLIC -fsycl-link-huge-device-code)
endif()

target_link_libraries(${_trgt} PUBLIC MKL::MKL_DPCPP)
target_link_libraries(${_trgt} PUBLIC oneDPL)

# needed for STL headers with GCC < 11
target_compile_definitions(${_trgt} PUBLIC _GLIBCXX_USE_TBB_PAR_BACKEND=0)
target_compile_definitions(${_trgt} PUBLIC PSTL_USE_PARALLEL_POLICIES=0)
# work-around for Windows at exit crash with predefined policies
target_compile_definitions(${_trgt} PUBLIC ONEDPL_USE_PREDEFINED_POLICIES=0)

target_include_directories(${_trgt} PUBLIC ${Dpctl_INCLUDE_DIR})
target_link_directories(${_trgt} PUBLIC "${Dpctl_INCLUDE_DIR}/..")
target_link_libraries(${_trgt} PUBLIC DPCTLSyclInterface)

add_library(dpnp_backend_library INTERFACE IMPORTED GLOBAL)
target_include_directories(dpnp_backend_library BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dpnp_backend_library INTERFACE ${_trgt})

if(DPNP_BACKEND_TESTS)
  add_subdirectory(tests)
endif()


install(
  TARGETS ${_trgt}
  LIBRARY DESTINATION dpnp
  PERMISSIONS
  OWNER_WRITE
  OWNER_READ
  OWNER_EXECUTE
  GROUP_READ
  GROUP_EXECUTE
  WORLD_READ
  WORLD_EXECUTE
  )
