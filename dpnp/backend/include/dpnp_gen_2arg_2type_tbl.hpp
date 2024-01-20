//*****************************************************************************
// Copyright (c) 2023-2024, Intel Corporation
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
 * Macro `MACRO_2ARG_2TYPES_LOGIC_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 *
 */

#ifndef MACRO_2ARG_2TYPES_LOGIC_OP
#error "MACRO_2ARG_2TYPES_LOGIC_OP is not defined"
#endif

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_2ARG_2TYPES_LOGIC_OP(__name__, __operation__)                    \
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
    template <typename _DataType_input1, typename _DataType_input2>            \
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
        const DPCTLEventVectorRef dep_event_vec_ref);

#endif

MACRO_2ARG_2TYPES_LOGIC_OP(dpnp_equal_c, input1_elem == input2_elem)
MACRO_2ARG_2TYPES_LOGIC_OP(dpnp_greater_c, input1_elem > input2_elem)
MACRO_2ARG_2TYPES_LOGIC_OP(dpnp_greater_equal_c, input1_elem >= input2_elem)
MACRO_2ARG_2TYPES_LOGIC_OP(dpnp_less_c, input1_elem < input2_elem)
MACRO_2ARG_2TYPES_LOGIC_OP(dpnp_less_equal_c, input1_elem <= input2_elem)
MACRO_2ARG_2TYPES_LOGIC_OP(dpnp_not_equal_c, input1_elem != input2_elem)

#undef MACRO_2ARG_2TYPES_LOGIC_OP
