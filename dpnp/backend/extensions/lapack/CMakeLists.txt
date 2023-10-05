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


set(python_module_name _lapack_impl)
pybind11_add_module(${python_module_name} MODULE
    lapack_py.cpp
    heevd.cpp
    syevd.cpp
    gesv.cpp
)

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
if (UNIX)
    # this option is support on Linux only
    target_link_options(${python_module_name} PUBLIC -fsycl-link-huge-device-code)
endif()

if (DPNP_GENERATE_COVERAGE)
    target_link_options(${python_module_name} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

if (MKL_VERSION_2024)
    target_link_libraries(${python_module_name} PUBLIC MKL::MKL_SYCL::LAPACK)
else()
    target_link_libraries(${python_module_name} PUBLIC MKL::MKL_DPCPP)
endif()

install(TARGETS ${python_module_name}
  DESTINATION "dpnp/backend/extensions/lapack"
)
