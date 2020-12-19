# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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

# initial version, only Linux was checked, use from root of project like
# cmake dpnp/backend "-DDPCPPROOT=$DPCPPROOT" "-DMKLROOT=$MKLROOT" "-DTBBROOT=$TBBROOT"
# make VERBOSE=1

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project("DPNP: NumPy-like API accelerated with SYCL")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "dpnp")

if(UNIX)
    set(CMAKE_CXX_COMPILER "clang++")
elseif(WIN32)
    set(CMAKE_CXX_COMPILER "dpcpp")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

set(CMAKE_CXX_FLAGS "-fsycl -DDPNP_LOCAL_QUEUE=1 -DMKL_ILP64=1")
# TODO replace this hardcode with finding of libs
set(CMAKE_SHARED_LINKER_FLAGS "-lmkl_rt -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl")

add_library(dpnp_backend_c SHARED
    src/dpnp_iface_fptr.cpp
    kernels/dpnp_krnl_bitwise.cpp
    kernels/dpnp_krnl_common.cpp
    kernels/dpnp_krnl_elemwise.cpp
    kernels/dpnp_krnl_linalg.cpp
    kernels/dpnp_krnl_manipulation.cpp
    kernels/dpnp_krnl_mathematical.cpp
    kernels/dpnp_krnl_random.cpp
    kernels/dpnp_krnl_reduction.cpp
    kernels/dpnp_krnl_searching.cpp
    kernels/dpnp_krnl_sorting.cpp
    kernels/dpnp_krnl_statistics.cpp
    src/memory_sycl.cpp
    src/queue_sycl.cpp
)

target_include_directories(
    dpnp_backend_c
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)
