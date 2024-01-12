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
 * Macro `MACRO_2ARG_3TYPES_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 * - vector operation over SYCL group used to calculate the result
 * - list of types vector operation accepts
 * - mkl operation used to calculate the result
 * - list of types OneMKL operation accepts
 *
 */

#ifndef MACRO_2ARG_3TYPES_OP
#error "MACRO_2ARG_3TYPES_OP is not defined"
#endif

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_2ARG_3TYPES_OP(__name__, __operation__, __vec_operation__,       \
                             __vec_types__, __mkl_operation__, __mkl_types__)  \
    /** @ingroup BACKEND_API */                                                \
    /** @brief Per element operation function __name__ */                      \
    /** */                                                                     \
    /** Function "__name__" executes operator "__operation__" over             \
     * corresponding elements of input arrays            */                    \
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
    /** @param[in]  input2_in          Input array 2. */                       \
    /** @param[in]  input2_size        Input array 2 size. */                  \
    /** @param[in]  input2_ndim        Number of input array 2 dimensions.     \
     */                                                                        \
    /** @param[in]  input2_shape       Input array 2 shape. */                 \
    /** @param[in]  input2_strides     Input array 2 strides. */               \
    /** @param[in]  where              Where condition. */                     \
    /** @param[in]  dep_event_vec_ref  Reference to vector of SYCL events.     \
     */                                                                        \
    template <typename _DataType_input1, typename _DataType_input2,            \
              typename _DataType_output>                                       \
    DPCTLSyclEventRef __name__(                                                \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType_input1, typename _DataType_input2,            \
              typename _DataType_output>                                       \
    void __name__(                                                             \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where)

#endif

MACRO_2ARG_3TYPES_OP(dpnp_add_c,
                     input1_elem + input2_elem,
                     x1 + x2,
                     MACRO_UNPACK_TYPES(bool, std::int32_t, std::int64_t),
                     oneapi::mkl::vm::add,
                     MACRO_UNPACK_TYPES(float,
                                        double,
                                        std::complex<float>,
                                        std::complex<double>))

MACRO_2ARG_3TYPES_OP(dpnp_arctan2_c,
                     sycl::atan2(input1_elem, input2_elem),
                     sycl::atan2(x1, x2),
                     MACRO_UNPACK_TYPES(float, double),
                     oneapi::mkl::vm::atan2,
                     MACRO_UNPACK_TYPES(float, double))

MACRO_2ARG_3TYPES_OP(dpnp_copysign_c,
                     sycl::copysign(input1_elem, input2_elem),
                     sycl::copysign(x1, x2),
                     MACRO_UNPACK_TYPES(float, double),
                     oneapi::mkl::vm::copysign,
                     MACRO_UNPACK_TYPES(float, double))

MACRO_2ARG_3TYPES_OP(dpnp_divide_c,
                     input1_elem / input2_elem,
                     x1 / x2,
                     MACRO_UNPACK_TYPES(bool, std::int32_t, std::int64_t),
                     oneapi::mkl::vm::div,
                     MACRO_UNPACK_TYPES(float,
                                        double,
                                        std::complex<float>,
                                        std::complex<double>))

MACRO_2ARG_3TYPES_OP(
    dpnp_fmod_c,
    dispatch_fmod_op(input1_elem, input2_elem),
    dispatch_fmod_op(x1, x2),
    MACRO_UNPACK_TYPES(std::int32_t, std::int64_t, float, double),
    oneapi::mkl::vm::fmod,
    MACRO_UNPACK_TYPES(float, double))

MACRO_2ARG_3TYPES_OP(dpnp_hypot_c,
                     sycl::hypot(input1_elem, input2_elem),
                     sycl::hypot(x1, x2),
                     MACRO_UNPACK_TYPES(float, double),
                     oneapi::mkl::vm::hypot,
                     MACRO_UNPACK_TYPES(float, double))

MACRO_2ARG_3TYPES_OP(dpnp_maximum_c,
                     sycl::max(input1_elem, input2_elem),
                     nullptr,
                     std::false_type,
                     oneapi::mkl::vm::fmax,
                     MACRO_UNPACK_TYPES(float, double))

MACRO_2ARG_3TYPES_OP(dpnp_minimum_c,
                     sycl::min(input1_elem, input2_elem),
                     nullptr,
                     std::false_type,
                     oneapi::mkl::vm::fmin,
                     MACRO_UNPACK_TYPES(float, double))

// "multiply" needs to be standalone kernel (not autogenerated) due to complex
// algorithm. This is not an element wise. pytest
// "tests/third_party/cupy/creation_tests/test_ranges.py::TestMgrid::test_mgrid3"
// requires multiplication shape1[10] with shape2[10,1] and result expected as
// shape[10,10]
MACRO_2ARG_3TYPES_OP(dpnp_multiply_c,
                     input1_elem *input2_elem,
                     x1 *x2,
                     MACRO_UNPACK_TYPES(bool, std::int32_t, std::int64_t),
                     oneapi::mkl::vm::mul,
                     MACRO_UNPACK_TYPES(float,
                                        double,
                                        std::complex<float>,
                                        std::complex<double>))

MACRO_2ARG_3TYPES_OP(dpnp_power_c,
                     static_cast<_DataType_output>(std::pow(input1_elem,
                                                            input2_elem)),
                     sycl::pow(x1, x2),
                     MACRO_UNPACK_TYPES(float, double),
                     oneapi::mkl::vm::pow,
                     MACRO_UNPACK_TYPES(float,
                                        double,
                                        std::complex<float>,
                                        std::complex<double>))

MACRO_2ARG_3TYPES_OP(dpnp_subtract_c,
                     input1_elem - input2_elem,
                     x1 - x2,
                     MACRO_UNPACK_TYPES(bool, std::int32_t, std::int64_t),
                     oneapi::mkl::vm::sub,
                     MACRO_UNPACK_TYPES(float,
                                        double,
                                        std::complex<float>,
                                        std::complex<double>))

#undef MACRO_2ARG_3TYPES_OP
