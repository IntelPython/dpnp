#                       Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CMake find_package() module for the IntelSycl compiler and development
# environment.
#
# Example usage:
#
# find_package(IntelSyclCompiler)
#
# If successful, the following variables will be defined:
# IntelSyclCompiler_FOUND
# IntelSyclCompiler_VERSION
# IntelSyclCompiler_INCLUDE_DIR
# IntelSyclCompiler_C_COMPILER
# IntelSyclCompiler_CXX_COMPILER
# IntelSyclCompiler_SYCL_INCLUDE_DIR
# IntelSyclCompiler_LIBRARY_DIR
# IntelSyclCompiler_SYCL_LIBRARY
# IntelSyclCompiler_OPENCL_LIBRARY

include(FindPackageHandleStandardArgs)
if(NOT DEFINED IntelSYCL_FOUND OR NOT IntelSYCL_FOUND)
   find_package(IntelSYCL REQUIRED)
endif()

# We will extract the version information from the compiler
set(clangxx_cmd "${CMAKE_CXX_COMPILER}")
set(clangxx_arg "--version")

# Compares the two version string that are supposed to be in x.y.z format
# and reports if the argument VERSION_STR1 is greater than or equal than
# version_str2. The strings are compared lexicographically after conversion to
# lists of equal lengths, with the shorter string getting zero-padded.
function(versions_greater_equal VERSION_STR1 VERSION_STR2 OUTPUT)
    # Convert the strings to list
    string(REPLACE  "." ";" VL1 ${VERSION_STR1})
    string(REPLACE  "." ";" VL2 ${VERSION_STR2})
    # get lengths of both lists
    list(LENGTH VL1 VL1_LEN)
    list(LENGTH VL2 VL2_LEN)
    set(LEN ${VL1_LEN})
    # If they differ in size pad the shorter list with 0s
    if(VL1_LEN GREATER VL2_LEN)
        math(EXPR DIFF "${VL1_LEN} - ${VL2_LEN}" OUTPUT_FORMAT DECIMAL)
        foreach(IDX RANGE 1 ${DIFF} 1)
            list(APPEND VL2 "0")
        endforeach()
    elseif(VL2_LEN GREATER VL2_LEN)
        math(EXPR DIFF "${VL1_LEN} - ${VL2_LEN}" OUTPUT_FORMAT DECIMAL)
        foreach(IDX RANGE 1 ${DIFF} 1)
            list(APPEND VL2 "0")
        endforeach()
        set(LEN ${VL2_LEN})
    endif()
    math(EXPR LEN_SUB_ONE "${LEN}-1")
    foreach(IDX RANGE 0 ${LEN_SUB_ONE} 1)
        list(GET VL1 ${IDX} VAL1)
        list(GET VL2 ${IDX} VAL2)

        if(${VAL1} GREATER ${VAL2})
            set(${OUTPUT} TRUE PARENT_SCOPE)
            break()
        elseif(${VAL1} LESS ${VAL2})
            set(${OUTPUT} FALSE PARENT_SCOPE)
            break()
        else()
            set(${OUTPUT} TRUE PARENT_SCOPE)
        endif()
    endforeach()
endfunction(versions_greater_equal)

# Check if dpcpp is available
execute_process(
    COMMAND ${clangxx_cmd} ${clangxx_arg}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE clangxx_result
    OUTPUT_VARIABLE clangxx_ver
)

# If dpcpp is found then set the package variables
if(${clangxx_result} MATCHES "0")
    string(REPLACE "\n" ";" IntelSyclCompiler_VERSION_LIST "${clangxx_ver}")
    set(IDX 0)
    foreach(X ${IntelSyclCompiler_VERSION_LIST})
        message(STATUS "dpcpp ver[${IDX}]: ${X}")
        MATH(EXPR IDX "${IDX}+1")
    endforeach()
    list(GET IntelSyclCompiler_VERSION_LIST 0 VERSION_STRING)

    # Get the dpcpp version
    string(REGEX MATCH
        "[0-9]+\.[0-9]+\.[0-9]+"
        IntelSyclCompiler_VERSION
        ${VERSION_STRING}
    )

    # Split out the version into major, minor an patch
    string(REPLACE "." ";" IntelSyclCompiler_VERSION_LIST1 "${IntelSyclCompiler_VERSION}")
    list(GET IntelSyclCompiler_VERSION_LIST1 0 IntelSyclCompiler_VERSION_MAJOR)
    list(GET IntelSyclCompiler_VERSION_LIST1 1 IntelSyclCompiler_VERSION_MINOR)
    list(GET IntelSyclCompiler_VERSION_LIST1 2 IntelSyclCompiler_VERSION_PATCH)
    set(IntelSyclCompiler_INCLUDE_DIR ${SYCL_INCLUDE_DIR})
    set(IntelSyclCompiler_SYCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR}/sycl)
    set(IntelSyclCompiler_LIBRARY_DIR ${SYCL_LIBRARY_DIR})
    if("x${CMAKE_SYSTEM_NAME}" STREQUAL "xWindows")
        find_file(
            IntelSyclCompiler_SYCL_LIBRARY
            NAMES "sycl.lib" "sycl6.lib" "sycl7.lib" "sycl8.lib"
            PATHS ${IntelSyclCompiler_LIBRARY_DIR}
        )
        find_file(
            IntelSyclCompiler_OPENCL_LIBRARY
            NAMES "OpenCL.lib"
            PATHS ${IntelSyclCompiler_LIBRARY_DIR}
        )
    elseif("x${CMAKE_SYSTEM_NAME}" STREQUAL "xLinux")
        find_file(
            IntelSyclCompiler_SYCL_LIBRARY
            NAMES "libsycl.so"
            PATHS ${IntelSyclCompiler_LIBRARY_DIR}
        )
        find_file(
            IntelSyclCompiler_OPENCL_LIBRARY
            NAMES "libOpenCL.so"
            PATHS ${IntelSyclCompiler_LIBRARY_DIR}
        )
    endif()

endif()

# Check if a specific version of DPCPP is requested.
if(IntelSyclCompiler_FIND_VERSION AND (DEFINED IntelSyclCompiler_VERSION))
    set(VERSION_GT_FIND_VERSION FALSE)
    versions_greater_equal(
        ${IntelSyclCompiler_VERSION}
        ${IntelSyclCompiler_FIND_VERSION}
        VERSION_GT_FIND_VERSION
    )
    if(VERSION_GT_FIND_VERSION)
        set(IntelSyclCompiler_FOUND TRUE)
    else()
        set(IntelSyclCompiler_FOUND FALSE)
    endif()
else()
    set(IntelSyclCompiler_FOUND TRUE)
endif()

find_package_handle_standard_args(IntelSyclCompiler DEFAULT_MSG
    IntelSyclCompiler_FOUND
    IntelSyclCompiler_VERSION
    IntelSyclCompiler_INCLUDE_DIR
    IntelSyclCompiler_SYCL_INCLUDE_DIR
    IntelSyclCompiler_LIBRARY_DIR
    IntelSyclCompiler_SYCL_LIBRARY
    IntelSyclCompiler_OPENCL_LIBRARY
)
