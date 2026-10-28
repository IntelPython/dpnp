# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2024, Intel Corporation
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

set(_elementwise_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/degrees.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/divmod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/erf_funcs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fabs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/float_power.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fmin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fmod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/frexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/gcd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/heaviside.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/i0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/isclose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/lcm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/ldexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logaddexp2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/modf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/nan_to_num.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/radians.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/sinc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/spacing.cpp
)

set(python_module_name _ufunc_impl)

set(_module_src
    # TODO: remove sources from `elementwise_functions` folder
    ${CMAKE_CURRENT_SOURCE_DIR}/../elementwise_functions/elementwise_functions_type_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../elementwise_functions/simplify_iteration_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ufunc_py.cpp
    ${_elementwise_sources}
)

build_dpnp_pybind11_ext(
    ${python_module_name}
    "dpnp/backend/extensions/ufunc"
    SOURCES ${_module_src}
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../ ${CMAKE_CURRENT_SOURCE_DIR}/../common
    LARGE_RANGES
)
