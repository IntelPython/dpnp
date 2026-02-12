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

find_package(Python REQUIRED COMPONENTS NumPy)

# -t is to only Cythonize sources with timestamps newer than existing CXX files (if present)
# -w is to set working directory (and correctly set __pyx_f[] array of filenames)
set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

if(WIN32)
    string(
        CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-Wstrict-prototypes "
        "-Wno-unused-parameter "
    )
    string(CONCAT SDL_FLAGS "/GS " "/DynamicBase ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O0 -g1 -DDEBUG -Xsycl-target-frontend=spir64 \"-g0\""
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O0 -g1 -DDEBUG -Xsycl-target-frontend=spir64 \"-g0\""
    )
    set(CMAKE_C_FLAGS_COVERAGE
        "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O1 -g1 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_COVERAGE
        "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O1 -g1 -DDEBUG"
    )
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
    set(DPCTL_LDFLAGS "/NXCompat;/DynamicBase")
    mark_as_advanced(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_C_FLAGS_COVERAGE
        CMAKE_MODULE_LINKER_FLAGS_COVERAGE
    )
elseif(UNIX)
    string(
        CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-Wstrict-prototypes "
        "-Wno-unused-parameter "
        "-fdiagnostics-color=auto "
    )
    string(
        CONCAT SDL_FLAGS
        "-fstack-protector "
        "-fstack-protector-all "
        "-fpic "
        "-fPIC "
        "-D_FORTIFY_SOURCE=2 "
        "-Wformat "
        "-Wformat-security "
        #       "-fno-strict-overflow "    # no-strict-overflow is implied by -fwrapv
        "-fno-delete-null-pointer-checks "
        "-fwrapv "
    )
    string(CONCAT CFLAGS "${WARNING_FLAGS}" "${SDL_FLAGS}")
    string(CONCAT CXXFLAGS "${WARNING_FLAGS}" "${SDL_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${CXXFLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O0 -g -DDEBUG -Xsycl-target-frontend=spir64 \"-g0\""
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -O0 -g -DDEBUG -Xsycl-target-frontend=spir64 \"-g0\""
    )
    set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O1 -g1 -DDEBUG")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -O1 -g1 -DDEBUG")
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
    set(DPCTL_LDFLAGS "-z,noexecstack,-z,relro,-z,now")
    mark_as_advanced(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_C_FLAGS_COVERAGE
        CMAKE_MODULE_LINKER_FLAGS_COVERAGE
    )
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

# at build time create include/ directory and copy header files over
set(DPCTL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_INSTALL_RPATH "$ORIGIN")

add_subdirectory(tensor)
