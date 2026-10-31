# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
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

set(python_module_name _lapack_impl)
set(_module_src
    ${CMAKE_CURRENT_SOURCE_DIR}/lapack_py.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/geqrf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/geqrf_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gesv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gesv_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gesvd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gesvd_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/getrf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/getrf_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/getri_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/getrs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/getrs_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/heevd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/heevd_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orgqr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orgqr_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/potrf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/potrf_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/potrs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/syevd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/syevd_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ungqr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ungqr_batch.cpp
)

build_dpnp_pybind11_ext(
    ${python_module_name}
    "dpnp/backend/extensions/lapack"
    SOURCES ${_module_src}
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(_use_onemath)
    target_link_libraries(${python_module_name} PRIVATE ${ONEMATH_LIB})
    target_compile_options(${python_module_name} PRIVATE -DUSE_ONEMATH)
else()
    target_link_libraries(${python_module_name} PUBLIC MKL::MKL_SYCL::LAPACK)
endif()
