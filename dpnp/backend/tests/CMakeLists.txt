find_package(GTest REQUIRED)

find_package(Threads REQUIRED)

if(GTEST_FOUND)
  message(STATUS "Found GTest:     (include: ${GTEST_INCLUDE_DIRS}, library: ${GTEST_LIBRARIES})")
else()
  message(FATAL_ERROR "Cannot find Google Test Framework!")
endif()

if(NOT Threads_FOUND)
  message(FATAL_ERROR "Cannot find Threads library!")
endif()

# Emulate autotools like make check target to build tests
# set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure -j 4)

enable_testing()

include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIB_DIR})

add_executable(dpnpc_tests test_random.cpp)
target_link_libraries(dpnpc_tests GTest::GTest GTest::Main pthread dpnp_backend_c)
