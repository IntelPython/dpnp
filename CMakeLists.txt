cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(dpnp
  VERSION 0.18
  LANGUAGES CXX
  DESCRIPTION "NumPy-like API accelerated by SYCL."
)

option(DPNP_GENERATE_COVERAGE "Enable build DPNP with coverage instrumentation" OFF)
option(DPNP_BACKEND_TESTS "Enable building of DPNP backend test suite" OFF)
option(DPNP_WITH_REDIST "Build DPNP assuming DPC++ redistributable is installed into Python prefix" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)


find_package(IntelSYCL REQUIRED PATHS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules NO_DEFAULT_PATH)
find_package(TBB QUIET)
if(TBB_FOUND)
    find_package(TBB REQUIRED)
else()
    find_package(TBB REQUIRED PATHS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules NO_DEFAULT_PATH)
endif()

set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "tbb_thread")
set(MKL_INTERFACE "ilp64")
find_package(MKL QUIET)
if(MKL_FOUND)
    find_package(MKL REQUIRED)
else()
    find_package(MKL REQUIRED PATHS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules NO_DEFAULT_PATH)
endif()

set(ONEDPL_PAR_BACKEND tbb)
find_package(oneDPL QUIET)
if(oneDPL_FOUND)
    if(oneDPL_VERSION VERSION_GREATER_EQUAL "2022.3.0")
        find_package(oneDPL REQUIRED)
    else()
        find_package(oneDPL REQUIRED PATHS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules NO_DEFAULT_PATH)
    endif()
else()
    find_package(oneDPL REQUIRED PATHS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules NO_DEFAULT_PATH)
endif()

include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    URL_HASH SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

find_package(Dpctl REQUIRED)
message(STATUS "Dpctl_INCLUDE_DIR=" ${Dpctl_INCLUDE_DIR})
message(STATUS "Dpctl_TENSOR_INCLUDE_DIR=" ${Dpctl_TENSOR_INCLUDE_DIR})

option(DPNP_TARGET_CUDA
    "Build DPNP to target CUDA devices"
    OFF
)
option(DPNP_USE_ONEMKL_INTERFACES
    "Build DPNP with oneMKL Interfaces"
    OFF
)
set(HIP_TARGETS "" CACHE STRING "HIP architecture for target")

set(_dpnp_sycl_targets)
set(_use_onemkl_interfaces OFF)
set(_use_onemkl_interfaces_cuda OFF)
set(_use_onemkl_interfaces_hip OFF)

set(_dpnp_sycl_target_compile_options)
set(_dpnp_sycl_target_link_options)

if ("x${DPNP_SYCL_TARGETS}" STREQUAL "x")
    if(DPNP_TARGET_CUDA)
        set(_dpnp_sycl_targets "nvptx64-nvidia-cuda,spir64-unknown-unknown")
        set(_use_onemkl_interfaces_cuda ON)
    endif()

    if (NOT "x${HIP_TARGETS}" STREQUAL "x")
        set(_use_onemkl_interfaces_hip ON)

        if ("x${_dpnp_sycl_targets}" STREQUAL "x")
            set(_dpnp_sycl_targets "amd_gpu_${HIP_TARGETS},spir64-unknown-unknown")
        else()
            set(_dpnp_sycl_targets "amd_gpu_${HIP_TARGETS},${_dpnp_sycl_targets}")
        endif()
    endif()
else()
    set(_dpnp_sycl_targets ${DPNP_SYCL_TARGETS})

    if ("${DPNP_SYCL_TARGETS}" MATCHES "nvptx64-nvidia-cuda")
        set(_use_onemkl_interfaces ON)
        set(_use_onemkl_interfaces_cuda ON)
    endif()

    if ("${DPNP_SYCL_TARGETS}" MATCHES "amd_gpu_")
        set(_use_onemkl_interfaces ON)
        set(_use_onemkl_interfaces_hip ON)

        if ("x${HIP_TARGETS}" STREQUAL "x")
            message(FATAL_ERROR "HIP_TARGETS must be specified when using HIP backend")
        endif()
    endif()

    if ("${DPNP_SYCL_TARGETS}" MATCHES "amdgcn-amd-amdhsa")
        message(FATAL_ERROR
            "Legacy target 'amdgcn-amd-amdhsa' is not supported. "
            "Use alias form 'amd_gpu_<arch>' instead"
        )
    endif()
endif()

if (_dpnp_sycl_targets)
   message(STATUS "Compiling for -fsycl-targets=${_dpnp_sycl_targets}")
   list(APPEND _dpnp_sycl_target_compile_options -fsycl-targets=${_dpnp_sycl_targets})
   list(APPEND _dpnp_sycl_target_link_options -fsycl-targets=${_dpnp_sycl_targets})
endif()

if(DPNP_USE_ONEMKL_INTERFACES)
    set(_use_onemkl_interfaces ON)
else()
    if(DEFINED ENV{DPNP_USE_ONEMKL_INTERFACES})
        set(_use_onemkl_interfaces ON)
    endif()
endif()

if(_use_onemkl_interfaces)
    set(BUILD_FUNCTIONAL_TESTS False)
    set(BUILD_EXAMPLES False)
    set(ENABLE_MKLGPU_BACKEND True)
    set(ENABLE_MKLCPU_BACKEND True)

    if(_use_onemkl_interfaces_cuda)
        set(ENABLE_CUBLAS_BACKEND True)
        set(ENABLE_CUSOLVER_BACKEND True)
        set(ENABLE_CUFFT_BACKEND True)
        # set(ENABLE_CURAND_BACKEND True)
    endif()
    if(_use_onemkl_interfaces_hip)
        set(ENABLE_ROCBLAS_BACKEND True)
        set(ENABLE_ROCSOLVER_BACKEND True)
        set(ENABLE_ROCFFT_BACKEND True)
        # set(ENABLE_ROCRAND_BACKEND True)
    endif()

    if(DPNP_ONEMKL_INTERFACES_DIR)
        FetchContent_Declare(onemkl_interfaces_library SOURCE_DIR "${DPNP_ONEMKL_INTERFACES_DIR}")
    else()
        FetchContent_Declare(
                onemkl_interfaces_library
                GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
                GIT_TAG 8f4312ef966420b9b8b4b82b9d5c22e2c91a1fe7  # v0.6
        )
    endif()

    FetchContent_MakeAvailable(onemkl_interfaces_library)
    if(TARGET onemath)
        set(MKL_INTERFACES_LIB "onemath" CACHE INTERNAL "OneMath lib target")
    elseif(TARGET onemkl)
        set(MKL_INTERFACES_LIB "onemkl" CACHE INTERNAL "OneMKL lib target")
    else()
        message(FATAL_ERROR "Neither 'oneMath' nor 'oneMKL' found!")
    endif()
    message(STATUS "MKL interfaces lib target used: ${MKL_INTERFACES_LIB}")
    set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib")
endif()

if(WIN32)
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-Wno-unused-parameter "
    )
    string(CONCAT SDL_FLAGS
        "/GS "
        "/DynamicBase "
    )
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
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-fdiagnostics-color=auto "
    )
    string(CONCAT SDL_FLAGS
        "-fstack-protector "
        "-fstack-protector-all "
        "-fpic "
        "-fPIC "
        "-D_FORTIFY_SOURCE=2 "
        "-Wformat "
        "-Wformat-security "
#        "-fno-strict-overflow "   # implied by -fwrapv
        "-fno-delete-null-pointer-checks "
        "-fwrapv "
    )
    string(CONCAT CFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
    )
    string(CONCAT CXXFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
        "-fsycl "
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${CXXFLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O0 -g1 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -O0 -g1 -DDEBUG"
    )
    set(DPNP_LDFLAGS "-z,noexecstack,-z,relro,-z,now")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

if (DPNP_GENERATE_COVERAGE)
    string(CONCAT PROFILE_FLAGS
        "-fprofile-instr-generate "
        "-fcoverage-mapping "
        "-fno-sycl-use-footer "
#        "-save-temps=obj "
    )

    # Add profiling flags
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${PROFILE_FLAGS}"
    )
endif()

if(DEFINED SKBUILD)
  set(_ignore_me ${SKBUILD})
endif()

add_subdirectory(dpnp)
