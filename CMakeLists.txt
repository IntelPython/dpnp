cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(dpnp
  DESCRIPTION "NumPy-like API accelerated by SYCL."
)

option(DPNP_GENERATE_COVERAGE "Enable build DPNP with coverage instrumentation" FALSE)
option(DPNP_BACKEND_TESTS "Enable building of DPNP backend test suite" FALSE)

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
set(MKL_VERSION_2024 FALSE)
find_package(MKL QUIET)
if(MKL_FOUND)
    if(MKL_VERSION VERSION_GREATER_EQUAL "2024.0.0")
        set(MKL_VERSION_2024 TRUE)
        set(MKL_INTERFACE "ilp64")
        find_package(MKL REQUIRED)
    endif()
endif()

if(NOT MKL_VERSION_2024)
    set(MKL_INTERFACE_FULL "intel_ilp64")
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
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
  URL_HASH SHA256=d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

find_package(Dpctl REQUIRED)
message(STATUS "Dpctl_INCLUDE_DIR=" ${Dpctl_INCLUDE_DIR})
message(STATUS "Dpctl_TENSOR_INCLUDE_DIR=" ${Dpctl_TENSOR_INCLUDE_DIR})

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

    # Enable LLVM Linker to reduce link time
    add_link_options("-fuse-ld=lld")
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
