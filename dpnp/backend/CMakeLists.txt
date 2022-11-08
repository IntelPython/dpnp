# *****************************************************************************
# Copyright (c) 2016-2022, Intel Corporation
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

# set(DPNP_VERSION 0.10.3)
# set(DPNP_API_VERSION 0.10)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# -----------------------------------------------------------------------------------------------
# Project build options...
# -----------------------------------------------------------------------------------------------
if(DEFINED ENV{ONEAPI_ROOT})
  set(DPNP_ONEAPI_ROOT "$ENV{ONEAPI_ROOT}" CACHE PATH "Folder contains oneapi tool set")
endif()

option(DPNP_STATIC_LIB_ENABLE "Enable build DPNP static library" FALSE)
option(DPNP_DEBUG_ENABLE "Enable output for DPNP_DEBUG statements" FALSE)
option(DPNP_INSTALL_STRUCTURED "if FALSE, install package files into same directory" TRUE)
option(DPNP_SYCL_QUEUE_MGR_ENABLE "Use external manager for SYCL queue" FALSE)
option(DPNP_BACKEND_TESTS "Enable DPNP tests" FALSE)

if(DEFINED ENV{DPNP_DEBUG})
  set(DPNP_DEBUG_ENABLE $ENV{DPNP_DEBUG})
endif()

message(STATUS "CMAKE_VERSION:                   ${CMAKE_VERSION}")
message(STATUS "CMAKE_GENERATOR:                 ${CMAKE_GENERATOR}")
message(STATUS "CMAKE_HOST_SYSTEM_NAME:          ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "========== User controlled variables list ==========")
message(STATUS "DPNP_ONEAPI_ROOT:                ${DPNP_ONEAPI_ROOT}")
message(STATUS "DPNP_STATIC_LIB_ENABLE:          ${DPNP_STATIC_LIB_ENABLE}")
message(STATUS "DPNP_DEBUG_ENABLE:               ${DPNP_DEBUG_ENABLE}")
message(STATUS "DPNP_BACKEND_TESTS:              ${DPNP_BACKEND_TESTS}")
message(STATUS "DPNP_INSTALL_STRUCTURED:         ${DPNP_INSTALL_STRUCTURED}")
message(STATUS "DPNP_SYCL_QUEUE_MGR_ENABLE:      ${DPNP_SYCL_QUEUE_MGR_ENABLE}")
message(STATUS " |- DPNP_QUEUEMGR_INCLUDE_DIR:   ${DPNP_QUEUEMGR_INCLUDE_DIR}")
message(STATUS " |- DPNP_QUEUEMGR_LIB_DIR:       ${DPNP_QUEUEMGR_LIB_DIR}")
message(STATUS "======= End of user controlled variables list ======")

# -----------------------------------------------------------------------------------------------
# Compiler-specific logic...
# -----------------------------------------------------------------------------------------------

# cmake 3.19.1 has a bug in dpcpp compiler detection. Let's assume it is a clang
# set(CMAKE_CXX_COMPILER_ID "Clang")
# set(CMAKE_CXX_COMPILER_VERSION 12.0)
if (CMAKE_VERSION VERSION_EQUAL 3.19.1)
    message(FATAL_ERROR
      " Unsupported cmake version ${CMAKE_VERSION}\n"
      " Please use other cmake version, for example:\n"
      "in Linux:\n"
      " curl --output cmake_webimage.tar.gz --url https://cmake.org/files/v3.19/cmake-3.19.2-Linux-x86_64.tar.gz --retry 5 --retry-delay 5\n"
      " tar -xzf cmake_webimage.tar.gz\n"
      " rm -f cmake_webimage.tar.gz\n"
      " export PATH=`pwd`/cmake-3.19.2-Linux-x86_64/bin:$PATH\n"
      "in Windows:\n"
      " curl.exe --output cmake_webimage.zip --url https://cmake.org/files/v3.19/cmake-3.19.2-win64-x64.zip --retry 5 --retry-delay 5\n"
      " tar -xf cmake_webimage.zip\n"
      " del cmake_webimage.zip\n"
      " set PATH=%CD%\\cmake-3.19.2-win64-x64\\bin;%PATH%\n"
        )
endif()

# SYCL related compile options
string(CONCAT COMMON_COMPILE_FLAGS
  "-fsycl "
  "-fsycl-device-code-split=per_kernel "
  "-fno-approx-func "
)
string(CONCAT COMMON_LINK_FLAGS
  "-fsycl "
  "-fsycl-device-code-split=per_kernel "
)
if(UNIX)
  set(CMAKE_CXX_COMPILER "dpcpp")
  # add_compile_options(-fPIC)
elseif(WIN32)
  set(CMAKE_CXX_COMPILER "dpcpp")
  # set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld-link")
  # set(CMAKE_LINKER "lld-link")
  # include (Platform/Windows-Clang)
  # set(CMAKE_LINKER "dpcpp")
  # set(CMAKE_AR "llvm-ar")
  # set(CMAKE_RANLIB "llvm-ranlib")
  # set(CMAKE_CXX_FLAGS "/EHsc")

  string(APPEND COMMON_COMPILER_FLAGS
    "/EHsc "
#    "/Ox "
#    "/W3 "
#    "/GL "
#    "/DNDEBUG "
#    "/MD "
  )
else()
  message(FATAL_ERROR "Unsupported system ${CMAKE_SYSTEM} in compiler selection case")
endif()

# set language version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# warning flag set
string(CONCAT DPNP_WARNING_FLAGS
  "-W "
  "-Wextra "
  "-Wshadow "
  "-Wall "
  "-Wstring-prototypes "
  "-Wformat "
  "-Wformat-security "
)
string(APPEND COMMON_COMPILER_FLAGS
  "${DPNP_WARNING_FLAGS}"
)

# debug/release compile definitions
if(DPNP_DEBUG_ENABLE)
  set(CMAKE_BUILD_TYPE "Debug")
  string(APPEND COMMON_COMPILER_FLAGS
    "-O0 "
  )
else()
  set(CMAKE_BUILD_TYPE "Release")
  string(APPEND COMMON_COMPILER_FLAGS
    "-O3 "
  )
endif()

# -----------------------------------------------------------------------------------------------
# Auxilary building options...
# -----------------------------------------------------------------------------------------------
# sdl
string(CONCAT DPNP_DEFS
  "-D_FORTIFY_SOURCE=2 "
)
if(NOT WIN32)
  string(APPEND COMMON_COMPILER_FLAGS
    "-fno-delete-null-pointer-checks "
    "-fstack-protector-strong "
    "-fno-strict-overflow "
    )
  string(APPEND COMMON_LINK_FLAGS
    "LINKER:-z,noexecstack,-z,relro,-z,now "
  )
endif()

# disable PSTL policies due to compiler bug
string(APPEND DPNP_DEFS
  "-DPSTL_USE_PARALLEL_POLICIES=0 "
  "-D_GLIBCXX_USE_TBB_PAR_BACKEND=0 "
)

# disable PSTL predefined policies objects (global queues, prevent fail on Windows)
string(APPEND DPNP_DEFS
  "-DONEDPL_USE_PREDEFINED_POLICIES=0 "
)

# -----------------------------------------------------------------------------------------------
# Create project...
# -----------------------------------------------------------------------------------------------
# set(CMAKE_CXX_COMPILER "clang++")
project(dpnp_project
        # VERSION ${DPNP_VERSION}
        DESCRIPTION "DPNP: NumPy-like API accelerated with SYCL"
        HOMEPAGE_URL https://github.com/IntelPython/dpnp
        LANGUAGES CXX)
# set(CMAKE_CXX_COMPILER "dpcpp")

# -----------------------------------------------------------------------------------------------
# Building logic...
# -----------------------------------------------------------------------------------------------
set(DPNP_SRC
    kernels/dpnp_krnl_arraycreation.cpp
    kernels/dpnp_krnl_bitwise.cpp
    kernels/dpnp_krnl_common.cpp
    kernels/dpnp_krnl_elemwise.cpp
    kernels/dpnp_krnl_fft.cpp
    kernels/dpnp_krnl_indexing.cpp
    kernels/dpnp_krnl_linalg.cpp
    kernels/dpnp_krnl_logic.cpp
    kernels/dpnp_krnl_manipulation.cpp
    kernels/dpnp_krnl_mathematical.cpp
    kernels/dpnp_krnl_random.cpp
    kernels/dpnp_krnl_reduction.cpp
    kernels/dpnp_krnl_searching.cpp
    kernels/dpnp_krnl_sorting.cpp
    kernels/dpnp_krnl_statistics.cpp
    src/constants.cpp
    src/dpnp_iface_fptr.cpp
    src/memory_sycl.cpp
    src/queue_sycl.cpp
    src/verbose.cpp
    src/dpnp_random_state.cpp
    )

if(DPNP_STATIC_LIB_ENABLE)
  add_library(dpnp_backend_c STATIC ${DPNP_SRC})
else()
  add_library(dpnp_backend_c SHARED ${DPNP_SRC})
  set_target_properties(dpnp_backend_c PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(dpnp_backend_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(dpnp_backend_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

string(REPLACE " " ";" COMMON_COMPILE_FLAGS_AS_LIST ${COMMON_COMPILE_FLAGS})
target_compile_options(dpnp_backend_c PUBLIC ${COMMON_COMPILE_FLAGS_AS_LIST})
string(REPLACE " " ";" DPNP_DEFS_AS_LIST ${DPNP_DEFS})
target_compile_definitions(dpnp_backend_c PUBLIC ${DPNP_DEFS_AS_LIST})
string(REPLACE " " ";" COMMON_LINK_FLAGS_AS_LIST ${COMMON_LINK_FLAGS})
target_link_options(dpnp_backend_c PUBLIC ${COMMON_LINK_FLAGS_AS_LIST})


# -----------------------------------------------------------------------------------------------
# Testing logic...
# -----------------------------------------------------------------------------------------------
if(DPNP_BACKEND_TESTS)
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------------------------
# Dependencies logic...
# -----------------------------------------------------------------------------------------------
# Math library
find_package(MathLib REQUIRED)
target_compile_definitions(dpnp_backend_c PUBLIC -DMKL_ILP64=1)
target_include_directories(dpnp_backend_c PUBLIC ${MATHLIB_INCLUDE_DIR})

link_directories(dpnp_backend_c PUBLIC ${MATHLIB_LIBRARY_DIR}) # does not work with some cmake versions
target_link_directories(dpnp_backend_c PUBLIC ${MATHLIB_LIBRARY_DIR}) # duplicate link_directories

if(UNIX)
# Link Line Advisor v6.13
# -DMKL_ILP64 -I"${MKLROOT}/include"
# -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
  set(DPNP_MATHLIB_DEP_LIBS
      mkl_sycl
      mkl_intel_ilp64
      mkl_tbb_thread # mkl_sequential
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
  set(DPNP_MATHLIB_DEP_LIBS
      mkl_sycl_dll
      mkl_intel_ilp64_dll
      mkl_tbb_thread_dll # mkl_sequential_dll
      mkl_core_dll
      sycl
      OpenCL
      CACHE STRING "Set of libraries to link")
else()
  message(FATAL_ERROR "Unsupported system ${CMAKE_SYSTEM} in MathLib libraries set")
endif()

target_link_libraries(dpnp_backend_c PUBLIC ${DPNP_MATHLIB_DEP_LIBS})

# Parallel STL
find_package(DPL REQUIRED)
target_include_directories(dpnp_backend_c PUBLIC ${DPL_INCLUDE_DIR})

# SYCL queue manager
if(DPNP_SYCL_QUEUE_MGR_ENABLE)
    target_include_directories(dpnp_backend_c PUBLIC ${DPNP_QUEUEMGR_INCLUDE_DIR})
    target_link_directories(dpnp_backend_c PUBLIC ${DPNP_QUEUEMGR_LIB_DIR})
    target_link_libraries(dpnp_backend_c PUBLIC "DPCTLSyclInterface")

    # not sure but add runpath
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${DPNP_QUEUEMGR_LIB_DIR}")

    # disable stripping rpath in installation logic
    set_target_properties(dpnp_backend_c PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
    target_compiler_definitions(dpnp_backend_c PUBLIC -DDPNP_LOCAL_QUEUE=1)
endif()

# -----------------------------------------------------------------------------------------------
# Installation logic...
# -----------------------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/package_dpnp)

if(DEFINED DPNP_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${DPNP_INSTALL_PREFIX})
endif()

if(NOT DPNP_INSTALL_STRUCTURED)
  set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX})
  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX})
  set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX})
endif()

# set_target_properties(dpnp_backend_c PROPERTIES VERSION ${DPNP_VERSION} SOVERSION ${DPNP_API_VERSION})

install(TARGETS dpnp_backend_c
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )

# -----------------------------------------------------------------------------------------------
# Print Global Options
# -----------------------------------------------------------------------------------------------
message(STATUS "CMAKE_SYSTEM:                    ${CMAKE_SYSTEM}")
message(STATUS "CMAKE_SYSTEM_VERSION:            ${CMAKE_SYSTEM_VERSION}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR:          ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_BUILD_TYPE:                ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX_STANDARD:                    ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CXX_COMPILER_ID:           ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION:      ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER:              ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_LINKER:                    ${CMAKE_LINKER}")
message(STATUS "CMAKE_SOURCE_DIR:                ${CMAKE_SOURCE_DIR}")
message(STATUS "DPNP_INSTALL_PREFIX:             ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_VERBOSE_MAKEFILE:          ${CMAKE_VERBOSE_MAKEFILE}")
