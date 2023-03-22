cmake_minimum_required(VERSION 3.21...3.26 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

project(dpnp
  DESCRIPTION "NumPy-like API accelerated by SYCL."
)

option(DPNP_GENERATE_COVERAGE "Enable build DPNP with coverage instrumentation" FALSE)
option(DPNP_BACKEND_TESTS "Enable building of DPNP backend test suite" FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

if (NOT DEFINED DPCTL_MODULE_PATH)
  if (DEFINED ENV{DPCTL_MODULE_PATH})
    set(DPCTL_MODULE_PATH $ENV{DPCTL_MODULE_PATH})
  else ()
    message(FATAL_ERROR "Specify DPCTL_MODULE_PATH, either via cmake or as environment varibale")
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_MODULE_PATH})


find_package(IntelDPCPP REQUIRED)
find_package(TBB REQUIRED HINTS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules)

set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_INTERFACE_FULL "intel_ilp64")
set(MKL_THREADING "tbb_thread")
find_package(MKL REQUIRED)

find_package(oneDPL REQUIRED HINTS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules)

include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.2.tar.gz
  URL_HASH SHA256=93bd1e625e43e03028a3ea7389bba5d3f9f2596abc074b068e70f4ef9b1314ae
)
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)

set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)
find_package(Dpctl REQUIRED)

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

add_subdirectory(dpnp)
