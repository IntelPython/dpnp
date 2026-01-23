# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

if(NOT _use_onemath)
    set(_elementwise_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/abs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/acos.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/acosh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/arg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/asin.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/asinh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/atan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/atan2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/atanh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cbrt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ceil.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/conj.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/copysign.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cos.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cosh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/div.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/erf_funcs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/exp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/exp2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/expm1.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/floor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fmin.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fmod.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hypot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/i0.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/inv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ln.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/log10.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/log1p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/log2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/modf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nextafter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pow.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sin.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sinh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sqr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sqrt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sub.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tanh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trunc.cpp
    )
endif()

set(_module_src
    # TODO: remove sources from `elementwise_functions` folder
    ${CMAKE_CURRENT_SOURCE_DIR}/../elementwise_functions/elementwise_functions_type_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../elementwise_functions/simplify_iteration_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vm_py.cpp
    ${_elementwise_sources}
)

set(python_module_name _vm_impl)

pybind11_add_module(${python_module_name} MODULE ${_module_src})
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})

if(WIN32)
    if(${CMAKE_VERSION} VERSION_LESS "3.27")
        # this is a work-around for target_link_options inserting option after -link option, cause
        # linker to ignore it.
        set(CMAKE_CXX_LINK_FLAGS
            "${CMAKE_CXX_LINK_FLAGS} -fsycl-device-code-split=per_kernel"
        )
    endif()
endif()

set_target_properties(
    ${python_module_name}
    PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    ${python_module_name}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../
)
target_include_directories(
    ${python_module_name}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_include_directories(
    ${python_module_name}
    PRIVATE ${CMAKE_SOURCE_DIR}/dpctl/tensor/libtensor/include
)

target_include_directories(${python_module_name} PUBLIC ${Dpctl_INCLUDE_DIR})

if(WIN32)
    target_compile_options(
        ${python_module_name}
        PRIVATE /clang:-fno-approx-func /clang:-fno-finite-math-only
    )
else()
    target_compile_options(
        ${python_module_name}
        PRIVATE -fno-approx-func -fno-finite-math-only
    )
endif()

target_link_options(${python_module_name} PUBLIC -fsycl-device-code-split=per_kernel)

if(DPNP_GENERATE_COVERAGE)
    target_link_options(
        ${python_module_name}
        PRIVATE -fprofile-instr-generate -fcoverage-mapping
    )
endif()

if(_use_onemath)
    target_compile_options(${python_module_name} PRIVATE -DUSE_ONEMATH)
else()
    target_link_libraries(${python_module_name} PUBLIC MKL::MKL_SYCL::VM)
endif()

if(DPNP_WITH_REDIST)
    set_target_properties(
        ${python_module_name}
        PROPERTIES INSTALL_RPATH "$ORIGIN/../../../../../../"
    )
endif()

install(TARGETS ${python_module_name} DESTINATION "dpnp/backend/extensions/vm")
