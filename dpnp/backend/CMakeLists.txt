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

# cmake-format -i CMakeLists.txt --line-width=120

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(DPNP_VERSION 0.4.0)
set(DPNP_API_VERSION 0.4)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# -----------------------------------------------------------------------------------------------
# Project build options...
# -----------------------------------------------------------------------------------------------
if(DEFINED ENV{ONEAPI_ROOT})
  set(DPNP_ONEAPI_ROOT "$ENV{ONEAPI_ROOT}" CACHE PATH "Folder contains oneapi tool set")
else()
  if(UNIX)
    set(DPNP_ONEAPI_ROOT "/opt/intel/oneapi" CACHE PATH "Folder contains oneapi tool set")
  elseif(WIN32)
    set(DPNP_ONEAPI_ROOT "C:\Program Files (x86)\Intel\oneAPI" CACHE PATH "Folder contains oneapi tool set")
  else()
    message(FATAL_ERROR "Unsupported system ${CMAKE_SYSTEM}")
  endif()
endif()

option(DPNP_STATIC_LIB_ENABLE "Enable build DPNP static library" FALSE)
option(DPNP_DEBUG_ENABLE "Enable output for DPNP_DEBUG statements" FALSE)

message(STATUS "DPNP_ONEAPI_ROOT:                ${DPNP_ONEAPI_ROOT}")
message(STATUS "DPNP_STATIC_LIB_ENABLE:          ${DPNP_STATIC_LIB_ENABLE}")
message(STATUS "DPNP_DEBUG_ENABLE:               ${DPNP_DEBUG_ENABLE}")

# -----------------------------------------------------------------------------------------------
# Compiler-specific logic...
# -----------------------------------------------------------------------------------------------
if(UNIX)
  set(CMAKE_CXX_COMPILER "dpcpp")
elseif(WIN32)
  set(CMAKE_CXX_COMPILER "dpcpp")
  # set(CMAKE_LINKER "lld-link")
  add_compile_options(/Ox)
  add_compile_options(/EHsc) # /Ox /W3 /GL /DNDEBUG /MD /EHsc
else()
  message(FATAL_ERROR "Unsupported system ${CMAKE_SYSTEM} in compiler selection case")
endif()

# set language version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SYCL related compile options
add_compile_options(-fsycl)
add_compile_options(-fsycl-device-code-split=per_kernel)

# debug/release compile definitions
set(DPNP_WARNING_FLAGS "-W -Wextra -Wshadow -Wall -Wstrict-prototypes" CACHE STRING "set of warning compiler switches")
if(DPNP_DEBUG_ENABLE)
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-O1 ${DPNP_WARNING_FLAGS})
else()
  set(CMAKE_BUILD_TYPE "Release")
  add_definitions(${DPNP_WARNING_FLAGS})
endif()

# -----------------------------------------------------------------------------------------------
# Auxilary building options...
# -----------------------------------------------------------------------------------------------
# sdl
add_definitions(-D_FORTIFY_SOURCE=2 -Wformat -Wformat-security)
if(NOT WIN32)
  add_compile_options(-fno-delete-null-pointer-checks -fstack-protector-strong -fno-strict-overflow)
  add_link_options(-Wl,-z,noexecstack,-z,relro,-z,now)
endif()

# disable PSTL policies due to compiler bug
add_definitions(-DPSTL_USE_PARALLEL_POLICIES=0 -D_GLIBCXX_USE_TBB_PAR_BACKEND=0)

project(dpnp_backend_c
        VERSION ${DPNP_VERSION}
        DESCRIPTION "DPNP: NumPy-like API accelerated with SYCL"
        LANGUAGES CXX)

# -----------------------------------------------------------------------------------------------
# Building logic...
# -----------------------------------------------------------------------------------------------
set(DPNP_SRC
    kernels/dpnp_krnl_bitwise.cpp
    kernels/dpnp_krnl_common.cpp
    kernels/dpnp_krnl_elemwise.cpp
    kernels/dpnp_krnl_fft.cpp
    kernels/dpnp_krnl_linalg.cpp
    kernels/dpnp_krnl_manipulation.cpp
    kernels/dpnp_krnl_mathematical.cpp
    kernels/dpnp_krnl_random.cpp
    kernels/dpnp_krnl_reduction.cpp
    kernels/dpnp_krnl_searching.cpp
    kernels/dpnp_krnl_sorting.cpp
    kernels/dpnp_krnl_statistics.cpp
    src/dpnp_iface_fptr.cpp
    src/memory_sycl.cpp
    src/queue_sycl.cpp)

target_include_directories(dpnp_backend_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(dpnp_backend_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(DPNP_STATIC_LIB_ENABLE)
  add_library(dpnp_backend_c STATIC ${DPNP_SRC})
else()
  add_library(dpnp_backend_c SHARED ${DPNP_SRC})
endif()

# -----------------------------------------------------------------------------------------------
# Dependencies logic...
# -----------------------------------------------------------------------------------------------
# Math library
find_package(MathLib REQUIRED)
add_definitions(-DMKL_ILP64=1)
target_include_directories(dpnp_backend_c PUBLIC ${MATHLIB_INCLUDE_DIR})
link_directories(dpnp_backend_c PUBLIC ${MATHLIB_LIBRARY_DIR})

if(UNIX)
# Link Line Advisor v6.13
# -DMKL_ILP64 -I"${MKLROOT}/include"
# -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
  set(DPNP_DEP_LIBS
      mkl_rt
      mkl_sycl
      mkl_intel_ilp64
      mkl_sequential
      mkl_core
      sycl
      OpenCL
      pthread
      m
      dl
      CACHE STRING "Set of libraries to link")
elseif(WIN32)
# Link Line Advisor v6.13
# -DMKL_ILP64 -I"%MKLROOT%\include"
# mkl_sycl_dll.lib mkl_intel_ilp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib sycl.lib OpenCL.lib
  set(DPNP_DEP_LIBS
      mkl_rt
      mkl_sycl #_dll
      mkl_intel_ilp64 #_dll
      mkl_sequential #_dll
      mkl_core #_dll
      sycl
      OpenCL
      CACHE STRING "Set of libraries to link")
else()
  message(FATAL_ERROR "Unsupported system ${CMAKE_SYSTEM} in MathLib libraries set")
endif()

target_link_libraries(dpnp_backend_c PUBLIC ${DPNP_DEP_LIBS})

# Parallel STL from DPC++
if(NOT WIN32)
  find_package(PSTL REQUIRED)
  target_include_directories(dpnp_backend_c PUBLIC ${PSTL_INCLUDE_DIR})
endif()

# SYCL queue manager
add_definitions(-DDPNP_LOCAL_QUEUE=1)

# -----------------------------------------------------------------------------------------------
# Installation logic...
# -----------------------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/package_dpnp)

if(DEFINED DPNP_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${DPNP_INSTALL_PREFIX})
endif()

set_target_properties(dpnp_backend_c PROPERTIES VERSION ${DPNP_VERSION} SOVERSION ${DPNP_API_VERSION})

install(TARGETS dpnp_backend_c DESTINATION "install")

# -----------------------------------------------------------------------------------------------
# Print Global Options
# -----------------------------------------------------------------------------------------------
message(STATUS "CMAKE_VERSION:                   ${CMAKE_VERSION}")
message(STATUS "CMAKE_GENERATOR:                 ${CMAKE_GENERATOR}")
message(STATUS "CMAKE_SYSTEM:                    ${CMAKE_SYSTEM}")
message(STATUS "CMAKE_SYSTEM_VERSION:            ${CMAKE_SYSTEM_VERSION}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR:          ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_HOST_SYSTEM_NAME:          ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "CXX_STANDARD:                    ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CXX_COMPILER:              ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_LINKER:                    ${CMAKE_LINKER}")
message(STATUS "DPNP_INSTALL_PREFIX:             ${CMAKE_INSTALL_PREFIX}")
