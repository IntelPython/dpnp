#
# Modifications, Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute, disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the
# License.
#
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
IntelSYCLConfig
-------

Library to verify SYCL compatibility of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``IntelSYCL_FOUND``
  True if the system has the SYCL library.
``SYCL_COMPILER``
  The resolved SYCL compiler.
``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.
``SYCL_COMPILER_VERSION``
  The SYCL version of the supported Compiler.
``SYCL_INCLUDE_DIR``
  Include directories needed to use SYCL.
``SYCL_INCLUDE_SYCL_DIR``
  Include directories needed to use SYCL.
``SYCL_IMPLEMENTATION_ID``
  The SYCL compiler variant.
``SYCL_FLAGS``
  SYCL specific flags for the compiler.

``IntelSYCL::SYCL_CXX``
  Target for using Intel SYCL (DPC++).  The following properties are defined
  for the target: ``INTERFACE_COMPILE_OPTIONS``, ``INTERFACE_LINK_OPTIONS``,
  ``INTERFACE_INCLUDE_DIRECTORIES``, and ``INTERFACE_LINK_DIRECTORIES``

Cache Variables
^^^^^^^^^^^^^^^

The following cache variable may also be set:

``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.

``SYCL_COMPILER_VERSION``
  The SYCL version of the supported Compiler.

``SYCL_COMPILER``
  The supported SYCL Compiler.

.. Note::

  1. User needs to set -DCMAKE_CXX_COMPILER or environment of
  CXX pointing to SYCL compatible compiler  ( eg: icx, clang++, icpx)


  2. Add this package to user's Cmake config file.

  .. code-block:: cmake

    find_package(IntelSYCL REQUIRED)

  3. Add sources to target through add_sycl_to_target()

  .. code-block:: cmake

     # Compile specific sources for SYCL and build target for SYCL
     add_executable(target_proj A.cpp B.cpp offload1.cpp offload2.cpp)
     add_sycl_to_target(TARGET target_proj SOURCES offload1.cpp offload2.cpp)

#]=======================================================================]

if(__INTEL_SYCL_CONFIG)
  return()
endif()
set(__INTEL_SYCL_CONFIG 1)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  # TODO add dependency package module checks, if any
endif()


# TODO: can't use find_program to override the CMAKE_CXX_COMPILER as
# Platform/ files are executed, potentially for a different compiler.
# Safer approach is to make user to define CMAKE_CXX_COMPILER.

string(COMPARE EQUAL "${CMAKE_CXX_COMPILER}" "" nocmplr)
if(nocmplr)
  set(IntelSYCL_FOUND False)
  set(SYCL_REASON_FAILURE "CMAKE_CXX_COMPILER not set!!")
  set(IntelSYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

if(NOT "x${SYCL_HOST_COMPILER}" STREQUAL "x")
  set(sycl_host_compiler "-fsycl-host-compiler=${SYCL_HOST_COMPILER}")
else()
  set(sycl_host_compiler "")
endif()

message(VERBOSE "SYCL_HOST_COMPILER = ${SYCL_HOST_COMPILER}")

# Check if a Compiler ID is being set. project() should be set prior to find_package()

if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "x")
   set(IntelSYCL_FOUND False)
   set(SYCL_REASON_FAILURE "CMake CXX Compiler family is not set. Please make sure find_package(IntelSYCL) is called after project()!!")
   set(IntelSYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
   return()
endif()

# Check for known compiler family that supports SYCL

if( NOT "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang" AND
    NOT "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xIntelLLVM")
   set(IntelSYCL_FOUND False)
   set(SYCL_REASON_FAILURE "Unsupported compiler family ${CMAKE_CXX_COMPILER_ID} and compiler ${CMAKE_CXX_COMPILER}. Also default Intel SYCL compiler not found in the environment!!")

   #find_program(INTEL_SYCL_COMPILER NAMES icpx icx icx-cc icx-cl)
   if(WIN32 AND
      (NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "GNU"))
     find_program(INTEL_SYCL_COMPILER NAMES icx icx-cl)
   else()
     find_program(INTEL_SYCL_COMPILER icpx)
   endif()

   if(NOT INTEL_SYCL_COMPILER)
     message(FATAL_ERROR "Intel SYCL Compiler not found.")
     set(IntelSYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
     return()
   else()
     # Remove trailing semicolon if present
     string(STRIP "${INTEL_SYCL_COMPILER};" INTEL_SYCL_COMPILER)
     message(STATUS "Setting SYCL Compiler to default ${INTEL_SYCL_COMPILER}.")
     set(SYCL_COMPILER ${INTEL_SYCL_COMPILER})
     set(IntelSYCL_FOUND True)
   endif()
else()
  # Assume that CXX Compiler supports SYCL and then test to verify.
  set(SYCL_COMPILER ${CMAKE_CXX_COMPILER})
endif()

# Function to write a test case to verify SYCL features.
function(parse_compiler_version compiler_name version_number)
    execute_process(
        COMMAND ${compiler_name} --version
        OUTPUT_VARIABLE COMPILER_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Intel Compiler Regex
    string(REGEX MATCH "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)"
           INTEL_VERSION_STRING ${COMPILER_VERSION_STRING})

    if(INTEL_VERSION_STRING)
        # Parse Intel Compiler Version
        string(REGEX REPLACE "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\2"
               SYCL_VERSION_STRING_MATCH ${INTEL_VERSION_STRING})
        string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
        list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
        list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
        list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
        math(EXPR VERSION_NUMBER_MATCH "${VERSION_MAJOR} * 10000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
    endif()
    set(${version_number} "${VERSION_NUMBER_MATCH}" PARENT_SCOPE)
endfunction()

parse_compiler_version(${SYCL_COMPILER} SYCL_COMPILER_VERSION)

if(NOT SYCL_COMPILER_VERSION)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "Cannot parse sycl compiler version to get SYCL_COMPILER_VERSION!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

#set(SYCL_COMPILER_VERSION ${COMPILER_VERSION})
message(STATUS "SYCL Compiler version: ${SYCL_COMPILER_VERSION}")

function(SYCL_FEATURE_TEST_WRITE src)

  set(pp_if "#if")
  set(pp_endif "#endif")

  set(SYCL_TEST_CONTENT "")
  string(APPEND SYCL_TEST_CONTENT "#include <iostream>\nusing namespace std;\n")
  string(APPEND SYCL_TEST_CONTENT "int main(){\n")

  # Feature tests goes here

  string(APPEND SYCL_TEST_CONTENT "${pp_if} defined(SYCL_LANGUAGE_VERSION)\n")
  string(APPEND SYCL_TEST_CONTENT "cout << \"SYCL_LANGUAGE_VERSION=\"<<SYCL_LANGUAGE_VERSION<<endl;\n")
  string(APPEND SYCL_TEST_CONTENT "${pp_endif}\n")

  string(APPEND SYCL_TEST_CONTENT "return 0;}\n")

  file(WRITE ${src} "${SYCL_TEST_CONTENT}")

endfunction()

# Function to Build the feature check test case.

function(SYCL_FEATURE_TEST_BUILD TEST_SRC_FILE TEST_EXE)

  # Convert CXX Flag string to list
  set(SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS}")
  separate_arguments(SYCL_CXX_FLAGS_LIST NATIVE_COMMAND ${SYCL_CXX_FLAGS_LIST})

  # Spawn a process to build the test case.
  execute_process(
    COMMAND "${SYCL_COMPILER}"
    ${SYCL_CXX_FLAGS_LIST}
    ${TEST_SRC_FILE}
    "-o"
    ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    OUTPUT_FILE ${SYCL_TEST_DIR}/Compile.log
    RESULT_VARIABLE result
    TIMEOUT 60
    #COMMAND_ECHO STDOUT
    )

  # Verify if test case build properly.
  if(result)
    message("SYCL feature test compile failed!")
    message("compile output is: ${output}")
  endif()

  # TODO: what to do if it doesn't build

endfunction()

# Function to run the test case to generate feature info.

function(SYCL_FEATURE_TEST_RUN TEST_EXE)

  # Spawn a process to run the test case.

  execute_process(
    COMMAND ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  # Verify the test execution output.
  if(test_result)
    set(IntelSYCL_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: feature test execution failed!!")
  endif()
  # TODO: what if the result is false.. error or ignore?

  set( test_result "${result}" PARENT_SCOPE)
  set( test_output "${output}" PARENT_SCOPE)

endfunction()


# Function to extract the information from test execution.
function(SYCL_FEATURE_TEST_EXTRACT test_output)

  string(REGEX REPLACE "\n" ";" test_output_list "${test_output}")

  set(SYCL_LANGUAGE_VERSION "")
  foreach(strl ${test_output_list})
     if(${strl} MATCHES "^SYCL_LANGUAGE_VERSION=([A-Za-z0-9_]+)$")
       string(REGEX REPLACE "^SYCL_LANGUAGE_VERSION=" "" extracted_sycl_lang "${strl}")
       set(SYCL_LANGUAGE_VERSION ${extracted_sycl_lang})
     endif()
  endforeach()

  set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" PARENT_SCOPE)
endfunction()

if(SYCL_COMPILER)
  # TODO ensure CMAKE_LINKER and CMAKE_CXX_COMPILER are same/supports SYCL.
  # set(CMAKE_LINKER ${SYCL_COMPILER})

  # use REALPATH to resolve symlinks
  get_filename_component(_REALPATH_SYCL_COMPILER "${SYCL_COMPILER}" REALPATH)
  get_filename_component(SYCL_BIN_DIR "${_REALPATH_SYCL_COMPILER}" DIRECTORY)
  get_filename_component(SYCL_PACKAGE_DIR "${SYCL_BIN_DIR}" DIRECTORY CACHE)

  # Find Include path from binary
  find_file(SYCL_INCLUDE_DIR
    NAMES
      include
    HINTS
      ${SYCL_PACKAGE_DIR} $ENV{SYCL_INCLUDE_DIR_HINT}
    NO_DEFAULT_PATH
      )

  # Find include/sycl path from include path.
  find_file(
    SYCL_INCLUDE_SYCL_DIR
    NAMES sycl
    HINTS
    ${SYCL_PACKAGE_DIR} $ENV{SYCL_INCLUDE_DIR_HINT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    )
  message(STATUS "SYCL_INCLUDE_DIR: ${SYCL_INCLUDE_DIR}")

  # Find Library directory
  find_file(SYCL_LIBRARY_DIR
    NAMES
      lib lib64
    HINTS
      ${SYCL_PACKAGE_DIR} $ENV{SYCL_LIBRARY_DIR_HINT}
    NO_DEFAULT_PATH
      )
  #TODO Make an input file to configure and update the lib current version
  if(WIN32)
    set(sycl_lib_suffix "9")
  else()
    set(sycl_lib_suffix "")
  endif()
  if(NOT "x${SYCL_LIB_SUFFIX}" STREQUAL "x")
    set(sycl_lib_suffix "${SYCL_LIB_SUFFIX}")
  endif()
  find_library(
    SYCL_LIBRARY
    NAMES "sycl${sycl_lib_suffix}"
    HINTS ${SYCL_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  message(STATUS "SYCL_LIBRARY=${SYCL_LIBRARY}")
endif()


set(SYCL_FLAGS "")
set(SYCL_LINK_FLAGS "")

# Based on Compiler ID, add support for SYCL
if( "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang" OR
    "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xIntelLLVM" OR IntelSYCL_FOUND)
  list(APPEND SYCL_FLAGS "-fsycl")
  list(APPEND SYCL_LINK_FLAGS "-fsycl")
endif()

# TODO verify if this is needed
# Windows: Add Exception handling
if(WIN32)
  list(APPEND SYCL_FLAGS "/EHsc")
endif()

list(JOIN SYCL_FLAGS " " SYCL_FLAGS_STRING)
message(DEBUG "SYCL_FLAGS_STRING: ${SYCL_FLAGS_STRING}")
set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS_STRING}")

# And now test the assumptions.

# Create a clean working directory.
set(SYCL_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCL")
file(REMOVE_RECURSE ${SYCL_TEST_DIR})
file(MAKE_DIRECTORY ${SYCL_TEST_DIR})

# Create the test source file
set(TEST_SRC_FILE "${SYCL_TEST_DIR}/sycl_features.cpp")
set(TEST_EXE "${TEST_SRC_FILE}.exe")
SYCL_FEATURE_TEST_WRITE(${TEST_SRC_FILE})

# Build the test and create test executable
SYCL_FEATURE_TEST_BUILD(${TEST_SRC_FILE} ${TEST_EXE})

# Execute the test to extract information
SYCL_FEATURE_TEST_RUN(${TEST_EXE})

# Extract test output for information
SYCL_FEATURE_TEST_EXTRACT(${test_output})

# As per specification, all the SYCL compatible compilers should
# define macro  SYCL_LANGUAGE_VERSION
string(COMPARE EQUAL "${SYCL_LANGUAGE_VERSION}" "" nosycllang)
if(nosycllang)
  set(IntelSYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL: It appears that the ${SYCL_COMPILER} does not support SYCL")
  set(IntelSYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

if(NOT "x${sycl_host_compiler}" STREQUAL "x")
  list(APPEND SYCL_FLAGS ${sycl_host_compiler})
  list(APPEND SYCL_CXX_FLAGS ${sycl_host_compiler})
endif()

# Placeholder for identifying various implementations of SYCL compilers.
# for now, set to the CMAKE_CXX_COMPILER_ID
set(SYCL_IMPLEMENTATION_ID "${CMAKE_CXX_COMPILER_ID}")

message(DEBUG "The SYCL compiler is ${SYCL_COMPILER}")
message(DEBUG "The SYCL Flags are ${SYCL_FLAGS}")
message(DEBUG "The SYCL Language Version is ${SYCL_LANGUAGE_VERSION}")

add_library(IntelSYCL::SYCL_CXX INTERFACE IMPORTED)
set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY
  INTERFACE_COMPILE_OPTIONS ${SYCL_FLAGS})
set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY
  INTERFACE_LINK_OPTIONS ${SYCL_LINK_FLAGS})
set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${SYCL_INCLUDE_DIR})
set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY
  INTERFACE_LINK_DIRECTORIES ${SYCL_LIBRARY_DIR})
set_property(TARGET IntelSYCL::SYCL_CXX PROPERTY
  INTERFACE_LINK_LIBRARIES ${SYCL_LIBRARY})

find_package_handle_standard_args(
  IntelSYCL
  FOUND_VAR IntelSYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_FLAGS SYCL_COMPILER_VERSION SYCL_COMPILER SYCL_LIBRARY
  VERSION_VAR SYCL_LANGUAGE_VERSION
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}")

# Include in Cache
set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" CACHE STRING "SYCL Language version")
set(SYCL_COMPILER_VERSION "${SYCL_COMPILER_VERSION}" CACHE STRING "SYCL Compiler version")
set(SYCL_COMPILER "${SYCL_COMPILER}" CACHE STRING "SYCL Compiler")
set(SYCL_INCLUDE_DIR "${SYCL_INCLUDE_DIR}" CACHE STRING "SYCL Include Dir ")
set(SYCL_INCLUDE_SYCL_DIR "${SYCL_INCLUDE_SYCL_DIR}" CACHE STRING "SYCL Include SYCL Dir ")
set(SYCL_LIBRARY_DIR "${SYCL_LIBRARY_DIR}" CACHE STRING "SYCL Library Dir")
set(SYCL_LIBRARY "${SYCL_LIBRARY}" CACHE STRING "SYCL Library")

function(add_sycl_to_target)

  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(SYCL
    ""
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN})


    get_target_property(__sycl_cxx_options IntelSYCL::SYCL_CXX INTERFACE_COMPILE_OPTIONS)
    get_target_property(__sycl_cxx_include_directories IntelSYCL::SYCL_CXX INTERFACE_INCLUDE_DIRECTORIES)

    if(NOT ${ARGC})
      message(FATAL_ERROR " add_sycl_to_target() does not have any arguments")
    elseif(${ARGC} EQUAL 1)
      message(WARNING "add_sycl_to_target() have only one argument specified.. assuming the target to be ${ARGV}.
Adding sycl to all sources but that may effect compilation times")
      set(SYCL_TARGET ${ARGV})
    endif()

    if(NOT SYCL_TARGET)
      message(FATAL_ERROR "add_sycl_to_target() requires a TARGET argument, but none was passed.")
    endif()

    if(NOT SYCL_SOURCES)
      message(WARNING "add_sycl_to_target() does not have sources specified.. Adding sycl to all sources but that may effect compilation times")
      target_compile_options(${SYCL_TARGET} PUBLIC ${__sycl_cxx_options})
      target_include_directories(${SYCL_TARGET} PUBLIC ${__sycl_cxx_include_directories})
    endif()

    foreach(source ${SYCL_SOURCES})
      set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "${__sycl_cxx_options}")
      set_source_files_properties(${source} PROPERTIES INCLUDE_DIRECTORIES "${__sycl_cxx_include_directories}")
    endforeach()

    get_target_property(__sycl_link_options
        IntelSYCL::SYCL_CXX INTERFACE_LINK_OPTIONS)
    target_link_options(${SYCL_TARGET} PUBLIC "${__sycl_link_options}")
    get_target_property(__sycl_link_directories
        IntelSYCL::SYCL_CXX INTERFACE_LINK_DIRECTORIES)
    target_link_directories(${SYCL_TARGET} PUBLIC "${__sycl_link_directories}")
endfunction(add_sycl_to_target)
