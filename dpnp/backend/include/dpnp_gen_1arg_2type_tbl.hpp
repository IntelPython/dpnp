//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)         \
    /** @ingroup BACKEND_API */                                                \
    /** @brief Per element operation function __name__ */                      \
    /** */                                                                     \
    /** Function "__name__" executes operator "__operation1__" over each       \
     * element of the array                        */                          \
    /** */                                                                     \
    /** @param[in]  q_ref              Reference to SYCL queue. */             \
    /** @param[out] result_out         Output array. */                        \
    /** @param[in]  result_size        Output array size. */                   \
    /** @param[in]  result_ndim        Number of output array dimensions.      \
     */                                                                        \
    /** @param[in]  result_shape       Output array shape. */                  \
    /** @param[in]  result_strides     Output array strides. */                \
    /** @param[in]  input1_in          Input array 1. */                       \
    /** @param[in]  input1_size        Input array 1 size. */                  \
    /** @param[in]  input1_ndim        Number of input array 1 dimensions.     \
     */                                                                        \
    /** @param[in]  input1_shape       Input array 1 shape. */                 \
    /** @param[in]  input1_strides     Input array 1 strides. */               \
    /** @param[in]  where              Where condition. */                     \
    /** @param[in]  dep_event_vec_ref  Reference to vector of SYCL events.     \
     */                                                                        \
    template <typename _DataType_input, typename _DataType_output>             \
    DPCTLSyclEventRef __name__(                                                \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType_input, typename _DataType_output>             \
    void __name__(                                                             \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where);

#endif

MACRO_1ARG_2TYPES_OP(dpnp_acos_c,
                     sycl::acos(input_elem),
                     oneapi::mkl::vm::acos(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_acosh_c,
    sycl::acosh(input_elem),
    oneapi::mkl::vm::acosh(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_asin_c,
                     sycl::asin(input_elem),
                     oneapi::mkl::vm::asin(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_asinh_c,
    sycl::asinh(input_elem),
    oneapi::mkl::vm::asinh(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_atan_c,
                     sycl::atan(input_elem),
                     oneapi::mkl::vm::atan(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_atanh_c,
    sycl::atanh(input_elem),
    oneapi::mkl::vm::atanh(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_cbrt_c,
                     sycl::cbrt(input_elem),
                     oneapi::mkl::vm::cbrt(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_ceil_c,
                     sycl::ceil(input_elem),
                     oneapi::mkl::vm::ceil(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_copyto_c, input_elem, q.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_cos_c,
                     sycl::cos(input_elem),
                     oneapi::mkl::vm::cos(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_cosh_c,
                     sycl::cosh(input_elem),
                     oneapi::mkl::vm::cosh(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_degrees_c,
                     sycl::degrees(input_elem),
                     q.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_exp2_c,
                     sycl::exp2(input_elem),
                     oneapi::mkl::vm::exp2(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_exp_c,
                     sycl::exp(input_elem),
                     oneapi::mkl::vm::exp(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_expm1_c,
    sycl::expm1(input_elem),
    oneapi::mkl::vm::expm1(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_fabs_c,
                     sycl::fabs(input_elem),
                     oneapi::mkl::vm::abs(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_floor_c,
    sycl::floor(input_elem),
    oneapi::mkl::vm::floor(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_log10_c,
    sycl::log10(input_elem),
    oneapi::mkl::vm::log10(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_log1p_c,
    sycl::log1p(input_elem),
    oneapi::mkl::vm::log1p(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_log2_c,
                     sycl::log2(input_elem),
                     oneapi::mkl::vm::log2(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_log_c,
                     sycl::log(input_elem),
                     oneapi::mkl::vm::ln(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_radians_c,
                     sycl::radians(input_elem),
                     q.submit(kernel_func))
MACRO_1ARG_2TYPES_OP(dpnp_sin_c,
                     sycl::sin(input_elem),
                     oneapi::mkl::vm::sin(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_sinh_c,
                     sycl::sinh(input_elem),
                     oneapi::mkl::vm::sinh(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_sqrt_c,
                     sycl::sqrt(input_elem),
                     oneapi::mkl::vm::sqrt(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_tan_c,
                     sycl::tan(input_elem),
                     oneapi::mkl::vm::tan(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(dpnp_tanh_c,
                     sycl::tanh(input_elem),
                     oneapi::mkl::vm::tanh(q, input1_size, input1_data, result))
MACRO_1ARG_2TYPES_OP(
    dpnp_trunc_c,
    sycl::trunc(input_elem),
    oneapi::mkl::vm::trunc(q, input1_size, input1_data, result))

#undef MACRO_1ARG_2TYPES_OP
