//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

/*
 * This header file contains single argument element wise functions definitions
 *
 * Macro `MACRO_1ARG_2TYPES_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 * - mkl operation used to calculate the result
 *
 */

#ifndef MACRO_1ARG_2TYPES_OP
#error "MACRO_1ARG_2TYPES_OP is not defined"
#endif

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)                                                  \
    /** @ingroup BACKEND_API                                                                                         */ \
    /** @brief Per element operation function __name__                                                               */ \
    /**                                                                                                              */ \
    /** Function "__name__" executes operator "__operation1__" over each element of the array                        */ \
    /**                                                                                                              */ \
    /** @param[in]  array1   Input array.                                                                            */ \
    /** @param[out] result1  Output array.                                                                           */ \
    /** @param[in]  size     Number of elements in the input array.                                                  */ \
    template <typename _DataType_input, typename _DataType_output>                                                      \
    void __name__(void* array1, void* result1, size_t size);

#endif

MACRO_1ARG_2TYPES_OP(dpnp_acos_c, cl::sycl::acos(input_elem), oneapi::mkl::vm::acos(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_acosh_c,
                     cl::sycl::acosh(input_elem),
                     oneapi::mkl::vm::acosh(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_asin_c, cl::sycl::asin(input_elem), oneapi::mkl::vm::asin(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_asinh_c,
                     cl::sycl::asinh(input_elem),
                     oneapi::mkl::vm::asinh(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_atan_c, cl::sycl::atan(input_elem), oneapi::mkl::vm::atan(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_atanh_c,
                     cl::sycl::atanh(input_elem),
                     oneapi::mkl::vm::atanh(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_cbrt_c, cl::sycl::cbrt(input_elem), oneapi::mkl::vm::cbrt(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_ceil_c, cl::sycl::ceil(input_elem), oneapi::mkl::vm::ceil(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(__dpnp_copyto_c, input_elem, DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_cos_c, cl::sycl::cos(input_elem), oneapi::mkl::vm::cos(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_cosh_c, cl::sycl::cosh(input_elem), oneapi::mkl::vm::cosh(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_degrees_c, cl::sycl::degrees(input_elem), DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_ediff1d_c, array1[i + 1] - input_elem, DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_exp2_c, cl::sycl::exp2(input_elem), oneapi::mkl::vm::exp2(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_exp_c, cl::sycl::exp(input_elem), oneapi::mkl::vm::exp(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_expm1_c,
                     cl::sycl::expm1(input_elem),
                     oneapi::mkl::vm::expm1(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_fabs_c, cl::sycl::fabs(input_elem), oneapi::mkl::vm::abs(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_floor_c,
                     cl::sycl::floor(input_elem),
                     oneapi::mkl::vm::floor(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_log10_c,
                     cl::sycl::log10(input_elem),
                     oneapi::mkl::vm::log10(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_log1p_c,
                     cl::sycl::log1p(input_elem),
                     oneapi::mkl::vm::log1p(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_log2_c, cl::sycl::log2(input_elem), oneapi::mkl::vm::log2(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_log_c, cl::sycl::log(input_elem), oneapi::mkl::vm::ln(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_radians_c, cl::sycl::radians(input_elem), DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_sin_c, cl::sycl::sin(input_elem), oneapi::mkl::vm::sin(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_sinh_c, cl::sycl::sinh(input_elem), oneapi::mkl::vm::sinh(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_sqrt_c, cl::sycl::sqrt(input_elem), oneapi::mkl::vm::sqrt(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_tan_c, cl::sycl::tan(input_elem), oneapi::mkl::vm::tan(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_tanh_c, cl::sycl::tanh(input_elem), oneapi::mkl::vm::tanh(DPNP_QUEUE, size, array1, result))
MACRO_1ARG_2TYPES_OP(dpnp_trunc_c,
                     cl::sycl::trunc(input_elem),
                     oneapi::mkl::vm::trunc(DPNP_QUEUE, size, array1, result))

#undef MACRO_1ARG_2TYPES_OP
