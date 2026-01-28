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

cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(
    dpnp
    VERSION 0.20
    LANGUAGES CXX
    DESCRIPTION "NumPy-like API accelerated by SYCL."
)

option(DPNP_GENERATE_COVERAGE "Enable build DPNP with coverage instrumentation" OFF)
option(DPNP_BACKEND_TESTS "Enable building of DPNP backend test suite" OFF)
option(
    DPNP_WITH_REDIST
    "Build DPNP assuming DPC++ redistributable is installed into Python prefix"
    OFF
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

# find_package() search order in CONFIG mode:
# 1. <PackageName>_ROOT variables (CMake variable / environment)
# 2. CMake-specific cache variables (on the command line with -D<PackageName>_DIR=...)
# 3. CMake-specific environment variables (<PackageName>_DIR, CMAKE_PREFIX_PATH, etc.)
# 4. Paths specified by the HINTS option
# 5. Standard system environment variables (PATH, etc.)
# 6. Paths stored in the CMake User Package Registry
# 7. CMake variables defined in the Platform files for the current system (like in /usr/local for Linux)
# 8. Paths stored in the CMake System Package Registry
# 9. Paths specified by the PATHS option (assumed hard-coded guesses)
set(path_to_cmake_dir ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules)
find_package(IntelSYCL REQUIRED PATHS ${path_to_cmake_dir})
find_package(TBB REQUIRED PATHS ${path_to_cmake_dir})

set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "tbb_thread")
set(MKL_INTERFACE "ilp64")
find_package(
    MKL
    REQUIRED
    HINTS ${MKL_ROOT}/lib/cmake ${MKL_ROOT}/lib/cmake/mkl $ENV{MKLROOT}
    PATHS ${path_to_cmake_dir}
)

set(ONEDPL_PAR_BACKEND tbb)
find_package(oneDPL REQUIRED PATHS ${path_to_cmake_dir})

include(GNUInstallDirs)

# find Python before enabling pybind11
find_package(Python 3.10...<3.15 REQUIRED COMPONENTS Development.Module NumPy)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
    URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
    FIND_PACKAGE_ARGS NAMES pybind11
)
FetchContent_MakeAvailable(pybind11)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

find_package(Dpctl REQUIRED)
message(STATUS "Dpctl_INCLUDE_DIR=" ${Dpctl_INCLUDE_DIR})
message(STATUS "Dpctl_TENSOR_INCLUDE_DIR=" ${Dpctl_TENSOR_INCLUDE_DIR})

option(DPNP_USE_ONEMATH "Build DPNP with oneMath" OFF)
set(DPNP_TARGET_CUDA
    ""
    CACHE STRING
    "Build DPNP to target CUDA device. \
Set to a truthy value (e.g., ON, TRUE) to use default architecture (sm_50), \
or to a specific architecture like sm_80."
)
set(DPNP_TARGET_HIP "" CACHE STRING "HIP architecture for target")

set(_dpnp_sycl_targets)
set(_use_onemath OFF)
set(_use_onemath_cuda OFF)
set(_use_onemath_hip OFF)

set(_dpnp_sycl_target_compile_options)
set(_dpnp_sycl_target_link_options)

if("x${DPNP_SYCL_TARGETS}" STREQUAL "x")
    if(DPNP_TARGET_CUDA)
        set(_dpnp_cuda_arch)
        if(DPNP_TARGET_CUDA MATCHES "^sm_")
            set(_dpnp_cuda_arch ${DPNP_TARGET_CUDA})
        elseif(DPNP_TARGET_CUDA MATCHES "^(ON|TRUE|YES|Y|1)$")
            set(_dpnp_cuda_arch "sm_50")
        else()
            message(
                FATAL_ERROR
                "Invalid value for DPNP_TARGET_CUDA: \"${DPNP_TARGET_CUDA}\". "
                "Expected 'ON', 'TRUE', 'YES', 'Y', '1', or a CUDA architecture like 'sm_80'."
            )
        endif()
        set(_dpnp_sycl_targets "nvidia_gpu_${_dpnp_cuda_arch},spir64-unknown-unknown")
        set(_use_onemath_cuda ON)
    endif()

    if(DPNP_TARGET_HIP)
        if(DPNP_TARGET_HIP MATCHES "^gfx")
            if("x${_dpnp_sycl_targets}" STREQUAL "x")
                set(_dpnp_sycl_targets
                    "amd_gpu_${DPNP_TARGET_HIP},spir64-unknown-unknown"
                )
            else()
                set(_dpnp_sycl_targets
                    "amd_gpu_${DPNP_TARGET_HIP},${_dpnp_sycl_targets}"
                )
            endif()

            set(_use_onemath_hip ON)
            set(HIP_TARGETS
                ${DPNP_TARGET_HIP}
                CACHE STRING
                "HIP GPU targets for oneMath"
            )
        else()
            message(
                FATAL_ERROR
                "Invalid value for DPNP_TARGET_HIP: \"${DPNP_TARGET_HIP}\". "
                "Expected an architecture name starting with 'gfx', e.g. 'gfx1030'."
            )
        endif()
    endif()
else()
    set(_dpnp_sycl_targets ${DPNP_SYCL_TARGETS})

    if("${DPNP_SYCL_TARGETS}" MATCHES "(nvidia_gpu_sm_|nvptx64-nvidia-cuda)")
        set(_use_onemath_cuda ON)
    endif()

    if("${DPNP_SYCL_TARGETS}" MATCHES "amd_gpu_")
        if("x${DPNP_TARGET_HIP}" STREQUAL "x")
            message(
                FATAL_ERROR
                "DPNP_TARGET_HIP must be specified when using HIP backend"
            )
        endif()

        set(_use_onemath_hip ON)
        set(HIP_TARGETS ${DPNP_TARGET_HIP} CACHE STRING "HIP GPU targets for oneMath")
    endif()

    if("${DPNP_SYCL_TARGETS}" MATCHES "amdgcn-amd-amdhsa")
        message(
            FATAL_ERROR
            "Legacy target 'amdgcn-amd-amdhsa' is not supported. "
            "Use alias form 'amd_gpu_<arch>' instead"
        )
    endif()
endif()

if(_dpnp_sycl_targets)
    message(STATUS "Compiling for -fsycl-targets=${_dpnp_sycl_targets}")
    list(APPEND _dpnp_sycl_target_compile_options -fsycl-targets=${_dpnp_sycl_targets})
    list(APPEND _dpnp_sycl_target_link_options -fsycl-targets=${_dpnp_sycl_targets})
endif()

if(DPNP_USE_ONEMATH)
    set(_use_onemath ON)
else()
    if(DEFINED ENV{DPNP_USE_ONEMATH})
        set(_use_onemath ON)
    endif()
endif()

if(_use_onemath)
    set(BUILD_FUNCTIONAL_TESTS False)
    set(BUILD_EXAMPLES False)
    set(ENABLE_MKLGPU_BACKEND True)
    set(ENABLE_MKLCPU_BACKEND True)

    if(_use_onemath_cuda)
        set(ENABLE_CUBLAS_BACKEND True)
        set(ENABLE_CUSOLVER_BACKEND True)
        set(ENABLE_CUFFT_BACKEND True)
        # set(ENABLE_CURAND_BACKEND True)
    endif()
    if(_use_onemath_hip)
        set(ENABLE_ROCBLAS_BACKEND True)
        set(ENABLE_ROCSOLVER_BACKEND True)
        set(ENABLE_ROCFFT_BACKEND True)
        # set(ENABLE_ROCRAND_BACKEND True)
    endif()

    if(DPNP_ONEMATH_DIR)
        FetchContent_Declare(onemath_library SOURCE_DIR "${DPNP_ONEMATH_DIR}")
    else()
        FetchContent_Declare(
            onemath_library
            GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
            GIT_TAG
                6ff3a43e555dbb20357017d48f0f6c6263259895 # v0.9
        )
    endif()

    FetchContent_MakeAvailable(onemath_library)
    if(TARGET onemath)
        set(ONEMATH_LIB "onemath" CACHE INTERNAL "OneMath lib target")
    elseif(TARGET onemkl)
        set(ONEMATH_LIB "onemkl" CACHE INTERNAL "OneMKL lib target")
    else()
        message(FATAL_ERROR "Neither 'oneMath' nor 'oneMKL' found!")
    endif()
    message(STATUS "OneMath lib target used: ${ONEMATH_LIB}")
    set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib")
else()
    if(_use_onemath_cuda OR _use_onemath_hip)
        message(
            FATAL_ERROR
            "CUDA or HIP targets are enabled, but oneMath is not. "
            "Please set DPNP_USE_ONEMATH=ON to enable them."
        )
    endif()
endif()

if(WIN32)
    string(
        CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-Wno-unused-parameter "
    )
    string(CONCAT SDL_FLAGS "/GS " "/DynamicBase ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O0 -g1 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${SDL_FLAGS} -O0 -g1 -DDEBUG"
    )
    set(DPNP_LDFLAGS "/NXCompat;/DynamicBase")
elseif(UNIX)
    string(
        CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
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
        # "-fno-strict-overflow "   # implied by -fwrapv
        "-fno-delete-null-pointer-checks "
        "-fwrapv "
    )
    string(CONCAT CFLAGS "${WARNING_FLAGS}" "${SDL_FLAGS}")
    string(CONCAT CXXFLAGS "${WARNING_FLAGS}" "${SDL_FLAGS}" "-fsycl ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${CXXFLAGS}")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O0 -g1 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -O0 -g1 -DDEBUG")
    set(DPNP_LDFLAGS "-z,noexecstack,-z,relro,-z,now")

    # Enable LLVM Linker to reduce link time
    add_link_options("-fuse-ld=lld")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

# Define flags for CMAKE_BUILD_TYPE=Coverage
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG} -O1 -g1 -DDEBUG")
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} -O1 -g1 -DDEBUG")
set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}")
mark_as_advanced(
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_MODULE_LINKER_FLAGS_COVERAGE
)

if(DPNP_GENERATE_COVERAGE)
    string(
        CONCAT PROFILE_FLAGS
        "-fprofile-instr-generate "
        "-fcoverage-mapping "
        "-fno-sycl-use-footer "
        # "-save-temps=obj "
    )

    # Add profiling flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROFILE_FLAGS}")
endif()

if(DEFINED SKBUILD)
    set(_ignore_me ${SKBUILD})
endif()

add_subdirectory(dpnp)
