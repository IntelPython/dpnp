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

#if defined(MACRO_1ARG_1TYPE_OP)

/*
 * This header file contains single argument element wise functions definitions
 *
 * Macro `MACRO_1ARG_1TYPE_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 * - mkl operation used to calculate the result
 *
 */

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)          \
    /** @ingroup BACKEND_API */                                                \
    /** @brief Per element operation function __name__ */                      \
    /** */                                                                     \
    /** Function "__name__" executes operator "__operation1__" over            \
     * each element of the array                        */                     \
    /** */                                                                     \
    /** @param[in]  q_ref              Reference to SYCL queue. */             \
    /** @param[out] result_out         Output array. */                        \
    /** @param[in]  result_size        Output array size. */                   \
    /** @param[in]  result_ndim        Number of output array                  \
     * dimensions. */                                                          \
    /** @param[in]  result_shape       Output array shape. */                  \
    /** @param[in]  result_strides     Output array strides. */                \
    /** @param[in]  input1_in          Input array 1. */                       \
    /** @param[in]  input1_size        Input array 1 size. */                  \
    /** @param[in]  input1_ndim        Number of input array 1                 \
     * dimensions. */                                                          \
    /** @param[in]  input1_shape       Input array 1 shape. */                 \
    /** @param[in]  input1_strides     Input array 1 strides. */               \
    /** @param[in]  where              Where condition. */                     \
    /** @param[in]  dep_event_vec_ref  Reference to vector of SYCL             \
     * events. */                                                              \
    template <typename _DataType>                                              \
    DPCTLSyclEventRef __name__(                                                \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType>                                              \
    void __name__(                                                             \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where);

#endif // _SECTION_DOCUMENTATION_GENERATION_

MACRO_1ARG_1TYPE_OP(dpnp_conjugate_c,
                    std::conj(input_elem),
                    q.submit(kernel_func))
MACRO_1ARG_1TYPE_OP(dpnp_copy_c, input_elem, q.submit(kernel_func))
MACRO_1ARG_1TYPE_OP(dpnp_erf_c,
                    dispatch_erf_op(input_elem),
                    oneapi::mkl::vm::erf(q, input1_size, input1_data, result))
MACRO_1ARG_1TYPE_OP(dpnp_negative_c, -input_elem, q.submit(kernel_func))
MACRO_1ARG_1TYPE_OP(
    dpnp_recip_c,
    _DataType(1) / input_elem,
    q.submit(kernel_func)) // error: no member named 'recip' in namespace 'sycl'
MACRO_1ARG_1TYPE_OP(dpnp_sign_c,
                    dispatch_sign_op(input_elem),
                    q.submit(kernel_func)) // no sycl::sign for int and long
MACRO_1ARG_1TYPE_OP(dpnp_square_c,
                    input_elem *input_elem,
                    oneapi::mkl::vm::sqr(q, input1_size, input1_data, result))

#undef MACRO_1ARG_1TYPE_OP

#else
#error "MACRO_1ARG_1TYPE_OP is not defined"
#endif // MACRO_1ARG_1TYPE_OP
