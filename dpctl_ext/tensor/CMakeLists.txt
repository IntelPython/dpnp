# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

find_package(Python COMPONENTS Development.Module)

if(WIN32)
    if(${CMAKE_VERSION} VERSION_LESS "3.27")
        # this is a work-around for target_link_options inserting option after -link option, cause
        # linker to ignore it.
        set(CMAKE_CXX_LINK_FLAGS
            "${CMAKE_CXX_LINK_FLAGS} -fsycl-device-code-split=per_kernel"
        )
    endif()
endif()

# TODO: reuse this library for dpnp ufunc extension build
set(_static_lib_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/simplify_iteration_space.cpp
)
set(_tensor_impl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/tensor_ctors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/accumulators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_and_cast_usm_to_usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_as_contig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_numpy_ndarray_into_usm_ndarray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_for_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_for_roll.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/linear_sequences.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/integer_advanced_indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/boolean_advanced_indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/eye_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/full_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/zeros_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/triul_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/where.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/device_support_queries.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/repeat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/clip.cpp
)

set(_static_lib_trgt simplify_iteration_space)

add_library(${_static_lib_trgt} STATIC ${_static_lib_sources})
target_include_directories(
    ${_static_lib_trgt}
    PRIVATE
        # ${Python_INCLUDE_DIRS}
        # ${Dpctl_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/include
)
target_link_libraries(${_static_lib_trgt} PRIVATE pybind11::headers Python::Module)
set_target_properties(${_static_lib_trgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_py_trgts)

set(python_module_name _tensor_impl)
pybind11_add_module(${python_module_name} MODULE ${_tensor_impl_sources})
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_tensor_impl_sources})
target_link_libraries(${python_module_name} PRIVATE ${_static_lib_trgt})
list(APPEND _py_trgts ${python_module_name})

set(_clang_prefix "")
if(WIN32)
    set(_clang_prefix "/clang:")
endif()

set(_no_fast_math_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_and_cast_usm_to_usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/full_ctor.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/linear_sequences.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/clip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/where.cpp
)
#list(
#APPEND _no_fast_math_sources
# ${_elementwise_sources}
# ${_reduction_sources}
# ${_sorting_sources}
# ${_linalg_sources}
# ${_accumulator_sources}
#)

foreach(_src_fn ${_no_fast_math_sources})
    get_source_file_property(_cmpl_options_prop ${_src_fn} COMPILE_OPTIONS)
    set(_combined_options_prop ${_cmpl_options_prop} "${_clang_prefix}-fno-fast-math")
    set_source_files_properties(
        ${_src_fn}
        PROPERTIES COMPILE_OPTIONS "${_combined_options_prop}"
    )
endforeach()

set(_compiler_definitions "")

set(_linker_options "LINKER:${DPNP_LDFLAGS}")
foreach(python_module_name ${_py_trgts})
    target_compile_options(
        ${python_module_name}
        PRIVATE -fno-sycl-id-queries-fit-in-int
    )
    target_link_options(
        ${python_module_name}
        PRIVATE -fsycl-device-code-split=per_kernel
    )
    # TODO: expand DPCTL_OFFLOAD_COMPRESS to the whole dpnp level
    if(DPCTL_OFFLOAD_COMPRESS)
        target_link_options(${python_module_name} PRIVATE --offload-compress)
    endif()

    target_include_directories(
        ${python_module_name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/dpnp/backend/include
            ${Dpctl_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/include
            ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/
    )
    target_link_options(${python_module_name} PRIVATE ${_linker_options})
    if(DPCTL_GENERATE_COVERAGE)
        if(DPCTL_GENERATE_COVERAGE_FOR_PYBIND11_EXTENSIONS)
            target_compile_options(
                ${python_module_name}
                PRIVATE -fprofile-instr-generate -fcoverage-mapping
            )
        endif()
        target_link_options(
            ${python_module_name}
            PRIVATE -fprofile-instr-generate -fcoverage-mapping
        )
    endif()
    if(_dpnp_sycl_targets)
        # make fat binary
        target_compile_options(
            ${python_module_name}
            PRIVATE ${_dpnp_sycl_target_compile_options}
        )
        target_link_options(
            ${python_module_name}
            PRIVATE ${_dpnp_sycl_target_link_options}
        )
    endif()
    # TODO: update source so they reference individual libraries instead of
    #   dpctl4pybind11.hpp. It will allow to simplify dependency tree
    # NOTE: dpctl C-API is resolved at runtime via Python
    # target_link_libraries(${python_module_name} PRIVATE DpctlCAPI)
    if(DPNP_WITH_REDIST)
        set_target_properties(
            ${python_module_name}
            PROPERTIES INSTALL_RPATH "$ORIGIN/../../../.."
        )
    endif()
    # TODO: revert to `DESTINATION "dpctl/tensor"`
    install(TARGETS ${python_module_name} DESTINATION "dpctl_ext/tensor")
endforeach()
