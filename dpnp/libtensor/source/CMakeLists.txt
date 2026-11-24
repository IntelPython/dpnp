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

# Builds the pybind11 modules of libtensor. All of them are installed into
# dpnp/libtensor, next to the MKL-based extensions.

include(${CMAKE_SOURCE_DIR}/dpnp/cmake/tensor_build_flags.cmake)

get_filename_component(
    _libtensor_include_dir
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ABSOLUTE
)

# Sources of the functions dpnp implements beyond the array API. They are
# compiled into the same modules as the libtensor functions but keep the
# coverage settings of the former dpnp extensions.
# They come first in each module: the linker keeps the first copy of an inline
# function both groups instantiate, so coverage builds keep the instrumented one.
set(_dpnp_elementwise_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/degrees.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/divmod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/erf_funcs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fabs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/float_power.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fmin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/fmod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/frexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/gcd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/heaviside.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/i0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/isclose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/lcm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/ldexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logaddexp2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/modf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/nan_to_num.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/radians.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/sinc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/spacing.cpp
)
set(_dpnp_indexing_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/choose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/putmask.cpp
)
set(_dpnp_reduction_sources ${CMAKE_CURRENT_SOURCE_DIR}/reductions/minmax.cpp)
set(_dpnp_sorting_sources ${CMAKE_CURRENT_SOURCE_DIR}/sorting/partition.cpp)
set(_statistics_impl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/bincount.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/histogram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/histogramdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/histogram_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/median.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/sliding_dot_product1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/sliding_window1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statistics/statistics_py.cpp
)
set(_window_impl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/window/kaiser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window/window_py.cpp
)
set(_dpnp_sources
    ${_dpnp_elementwise_sources}
    ${_dpnp_indexing_sources}
    ${_dpnp_reduction_sources}
    ${_dpnp_sorting_sources}
    ${_statistics_impl_sources}
    ${_window_impl_sources}
)

set(_static_lib_sources ${CMAKE_CURRENT_SOURCE_DIR}/simplify_iteration_space.cpp)
set(_tensor_impl_sources
    ${_dpnp_indexing_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_ctors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accumulators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_and_cast_usm_to_usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_as_contig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_numpy_ndarray_into_usm_ndarray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_for_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_for_roll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_sequences.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/integer_advanced_indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/boolean_advanced_indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eye_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/full_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/zeros_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/triul_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/where.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_support_queries.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/repeat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/clip.cpp
)
set(_accumulator_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/accumulators/accumulators_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accumulators/cumulative_logsumexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accumulators/cumulative_prod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accumulators/cumulative_sum.cpp
)
set(_elementwise_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/elementwise_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/elementwise_functions_type_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/abs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/acos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/acosh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/angle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/asin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/asinh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/atan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/atan2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/atanh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_invert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_left_shift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_right_shift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/bitwise_xor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/cbrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/ceil.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/conj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/copysign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/cos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/cosh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/exp2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/expm1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/floor_divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/floor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/greater_equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/greater.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/hypot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/imag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/isfinite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/isinf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/isnan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/less_equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/less.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/log1p.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/log2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/log10.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logaddexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logical_and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logical_not.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logical_or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/logical_xor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/maximum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/minimum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/negative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/nextafter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/not_equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/positive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/pow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/proj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/real.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/reciprocal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/remainder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/round.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/rsqrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/sign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/signbit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/sin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/sinh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/sqrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/square.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/subtract.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/tan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/tanh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/true_divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/trunc.cpp
)
set(_reduction_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/reduction_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/any.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/argmin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/logsumexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/min.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/prod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/reduce_hypot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reductions/sum.cpp
)
set(_sorting_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/isin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/merge_sort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/merge_argsort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/radix_sort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/radix_argsort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/searchsorted.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sorting/topk.cpp
)
set(_linalg_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/elementwise_functions_type_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linalg_functions/dot.cpp
)
set(_tensor_accumulation_impl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_accumulation.cpp
    ${_accumulator_sources}
)
set(_tensor_elementwise_impl_sources
    ${_dpnp_elementwise_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_elementwise.cpp
    ${_elementwise_sources}
)
set(_tensor_reductions_impl_sources
    ${_dpnp_reduction_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_reductions.cpp
    ${_reduction_sources}
)
set(_tensor_sorting_impl_sources
    ${_dpnp_sorting_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_sorting.cpp
    ${_sorting_sources}
)
set(_tensor_linalg_impl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_linalg.cpp
    ${_linalg_sources}
)

set(_static_lib_trgt simplify_iteration_space)

add_library(${_static_lib_trgt} STATIC ${_static_lib_sources})
target_include_directories(${_static_lib_trgt} PRIVATE ${_libtensor_include_dir})
target_link_libraries(${_static_lib_trgt} PRIVATE pybind11::headers Python::Module)
set_target_properties(${_static_lib_trgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_py_trgts
    _tensor_impl
    _tensor_accumulation_impl
    _tensor_elementwise_impl
    _tensor_reductions_impl
    _tensor_sorting_impl
    _tensor_linalg_impl
    _statistics_impl
    _window_impl
)
foreach(python_module_name ${_py_trgts})
    build_dpnp_pybind11_ext(
        ${python_module_name}
        SOURCES ${${python_module_name}_sources}
        INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        LARGE_RANGES
    )
endforeach()
foreach(
    python_module_name
    _tensor_impl
    _tensor_accumulation_impl
    _tensor_elementwise_impl
    _tensor_reductions_impl
    _tensor_sorting_impl
    _tensor_linalg_impl
)
    target_link_libraries(${python_module_name} PRIVATE ${_static_lib_trgt})
endforeach()

if(DPNP_GENERATE_COVERAGE)
    # dpnp sources of the modules with the libtensor radix select kernels
    set(_dpnp_radix_select_sources ${_dpnp_sorting_sources} ${_statistics_impl_sources})
    set(_dpnp_device_coverage_sources ${_dpnp_sources})
    list(REMOVE_ITEM _dpnp_device_coverage_sources ${_dpnp_radix_select_sources})
    set_property(
        SOURCE ${_dpnp_device_coverage_sources}
        APPEND
        PROPERTY
            COMPILE_OPTIONS
                -fprofile-instr-generate
                -fcoverage-mapping
                -fno-sycl-use-footer
    )

    # libtensor code the dpnp functions use, measured like the dpnp sources
    set_property(
        SOURCE ${_static_lib_sources}
        APPEND
        PROPERTY COMPILE_OPTIONS -fprofile-instr-generate -fcoverage-mapping
    )
    # Host only: the device coverage records of a kernel-less TU break sycl-link
    # (a source property, unlike target options, keeps the repeated -Xarch_host).
    # The radix select modules are host only too: every kernel image of a module
    # gets all its device counters as device globals, and their init events
    # crash the OpenCL CPU runtime at random in the kernel bundle launches.
    # The device counts never reach the profile anyway
    set_property(
        SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_functions/elementwise_functions_type_utils.cpp
            ${_dpnp_radix_select_sources}
        APPEND
        PROPERTY
            COMPILE_OPTIONS
                -Xarch_host
                -fprofile-instr-generate
                -Xarch_host
                -fcoverage-mapping
                -fno-sycl-use-footer
    )
endif()

if(DPNP_GENERATE_COVERAGE AND DPNP_TENSOR_GENERATE_COVERAGE_FOR_PYBIND11_EXTENSIONS)
    foreach(python_module_name ${_py_trgts})
        target_compile_options(
            ${python_module_name}
            PRIVATE -fprofile-instr-generate -fcoverage-mapping
        )
    endforeach()
endif()
