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

# Directory-scoped build flags shared by the dpnp.tensor Cython extensions and
# the libtensor pybind11 modules built in dpnp/libtensor/source. Include it from
# the CMakeLists.txt of a directory before defining any targets there.

# Tensor-specific flags

# dpctl doesn't add -fsycl globally
# only to pybind11 module sources via add_sycl_to_target()
string(REPLACE "-fsycl " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Use LLD linker (dpctl sets this at root level)
if(UNIX)
    add_link_options("-fuse-ld=lld")
endif()

# Remove global coverage flags for tensor
# use link-time only approach like dpctl
if(DPNP_GENERATE_COVERAGE)
    string(REPLACE "-fprofile-instr-generate " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-fcoverage-mapping " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-fno-sycl-use-footer " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Tensor-specific debug flags
# Disable device code debug info for Debug and Coverage builds to speed up linking
if(
    CMAKE_BUILD_TYPE STREQUAL "Debug"
    OR CMAKE_BUILD_TYPE STREQUAL "DEBUG"
    OR CMAKE_BUILD_TYPE STREQUAL "Coverage"
)
    if(WIN32)
        add_compile_options(-Xsycl-target-frontend=spir64 "-g0")
    elseif(UNIX)
        add_compile_options(-Xsycl-target-frontend=spir64 "-g0")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
            string(REPLACE "-g1" "-g" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
            string(REPLACE "-g1" "-g" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        endif()
    endif()
endif()

# Match dpctl warning flags
# Suppress unused parameter warnings
add_compile_options(-Wno-unused-parameter)
