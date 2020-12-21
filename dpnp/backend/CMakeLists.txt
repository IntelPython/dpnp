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
set(DPNP_VERSION 0.4.0)
set(DPNP_API_VERSION 0.4)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

#-----------------------------------------------------------------------------------------------
# Project build options...
#-----------------------------------------------------------------------------------------------
option(DPNP_STATIC_LIB_ENABLE "Enable build DPNP static library" FALSE)
option(DPNP_DEBUG_ENABLE "Enable output for DPNP_DEBUG statements" FALSE)

message(STATUS "DPNP_STATIC_LIB_ENABLE:          ${DPNP_STATIC_LIB_ENABLE}")
message(STATUS "DPNP_DEBUG_ENABLE:               ${DPNP_DEBUG_ENABLE}")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------
if(UNIX)
    set(CMAKE_CXX_COMPILER "dpcpp")
elseif(WIN32)
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

set(DPNP_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

add_definitions(-fsycl-device-code-split=per_kernel)
add_definitions(-fsycl)
add_definitions(-std=c++17)

if(DPNP_DEBUG_ENABLE)
    add_definitions(-g -O1 -W -Wextra -Wshadow -Wall -Wstrict-prototypes)
else()
    add_definitions(-O3 -DNDEBUG)
endif()

#-----------------------------------------------------------------------------------------------
# Auxilary building options...
#-----------------------------------------------------------------------------------------------
# sdl
add_definitions(-fstack-protector-strong -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks)
# set(DPNP_SDL_LDFLAGS "-Wl,-z,noexecstack,-z,relro,-z,now")

add_definitions(-DPSTL_USE_PARALLEL_POLICIES=0 -D_GLIBCXX_USE_TBB_PAR_BACKEND=0)

#-----------------------------------------------------------------------------------------------
# Print Global Options
#-----------------------------------------------------------------------------------------------
message(STATUS "Compile Flags:                   ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE          ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG            ${CMAKE_CXX_FLAGS_DEBUG}")

#-----------------------------------------------------------------------------------------------
# Building logic...
#-----------------------------------------------------------------------------------------------
set (DPNP_SRC
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

if (DPNP_STATIC_LIB_ENABLE)
    add_library(dpnp_backend_c STATIC ${DPNP_SRC})
else()
    add_library(dpnp_backend_c SHARED ${DPNP_SRC})
endif()

target_include_directories(dpnp_backend_c PUBLIC "${DPNP_INCLUDE_PATH}")

#-----------------------------------------------------------------------------------------------
# Dependencies logic...
#-----------------------------------------------------------------------------------------------
find_package(MathLib REQUIRED)
target_include_directories(dpnp_backend_c PUBLIC ${MATHLIB_INCLUDE_DIR})
target_link_directories(dpnp_backend_c PRIVATE ${MATHLIB_LIBRARY_DIR})
add_definitions(-DMKL_ILP64=1)
set(DPNP_DEP_LIBS mkl_rt mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl OpenCL pthread m dl)
target_link_libraries(dpnp_backend_c PUBLIC ${DPNP_DEP_LIBS})

find_package(PSTL REQUIRED)
target_include_directories(dpnp_backend_c PUBLIC ${PSTL_INCLUDE_DIR})

add_definitions(-DDPNP_LOCAL_QUEUE=1)

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/package_dpnp)

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_BINDIR "bin")
    set(CMAKE_INSTALL_INCLUDEDIR "include")
    set(CMAKE_INSTALL_DOCDIR "doc")
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if (DEFINED DPNP_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${DPNP_INSTALL_PREFIX})
endif()
message(STATUS "Installation directory:          ${CMAKE_INSTALL_PREFIX}")

set_target_properties(dpnp_backend_c PROPERTIES
                      VERSION ${DPNP_VERSION}
                      SOVERSION ${DPNP_API_VERSION})

install(TARGETS dpnp_backend_c DESTINATION "install")
