# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

set(python_module_name _rng_impl)
set(_module_src
    ${CMAKE_CURRENT_SOURCE_DIR}/binomial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/discrete.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exponential.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gamma.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/integers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/noncentral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/normal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/poisson.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rng_py.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/uniform.cpp
)

build_dpnp_pybind11_ext(
    ${python_module_name}
    SOURCES ${_module_src}
)

# oneMKL device RNG API is header-only, so no oneMKL library is linked
target_include_directories(
    ${python_module_name}
    SYSTEM
    PRIVATE $<TARGET_PROPERTY:MKL::MKL_SYCL,INTERFACE_INCLUDE_DIRECTORIES>
)
