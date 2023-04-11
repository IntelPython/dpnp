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

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

if(GTEST_FOUND)
  message(STATUS "Found GTest:                     (include: ${GTEST_INCLUDE_DIRS}, library: ${GTEST_LIBRARIES})")
else()
  message(FATAL_ERROR "Cannot find Google Test Framework!")
endif()

if(NOT Threads_FOUND)
  message(FATAL_ERROR "Cannot find Threads library!")
endif()

# Emulate autotools like make check target to build tests
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
enable_testing()

include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIB_DIR})

# TODO split
add_executable(dpnpc_tests
               test_broadcast_iterator.cpp
               test_main.cpp
               test_random.cpp
               test_utils.cpp
               test_utils_iterator.cpp)
target_link_libraries(dpnpc_tests GTest::GTest GTest::Main pthread dpnp_backend_library)

# TODO split
add_test(dpnpc_tests dpnpc_tests)
gtest_discover_tests(dpnpc_tests)
add_dependencies(check dpnpc_tests)
