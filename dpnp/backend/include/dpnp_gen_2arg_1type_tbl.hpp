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
 * This header file contains single argument bitwise functions definitions
 *
 * Macro `MACRO_CUSTOM_2ARG_1TYPE_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 *
 */

#ifndef MACRO_CUSTOM_2ARG_1TYPE_OP
#error "MACRO_CUSTOM_2ARG_1TYPE_OP is not defined"
#endif

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_CUSTOM_2ARG_1TYPE_OP(__name__, __operation__)                                                             \
    /** @ingroup BACKEND_API                                                                                         */ \
    /** @brief Element wise operation function __name__                                                              */ \
    /**                                                                                                              */ \
    /** Function "__name__" executes operator "__operation__" over corresponding elements of input arrays            */ \
    /**                                                                                                              */ \
    /** @param[in]  array1   Input array 1.                                                                          */ \
    /** @param[in]  array2   Input array 2.                                                                          */ \
    /** @param[out] result1  Output array.                                                                           */ \
    /** @param[in]  size     Number of elements in the output array.                                                 */ \
    template <typename _DataType>                                                                                       \
    void __name__(void* array1, void* array2, void* result1, size_t size);

#endif

MACRO_CUSTOM_2ARG_1TYPE_OP(dpnp_bitwise_and_c, input_elem1& input_elem2)
MACRO_CUSTOM_2ARG_1TYPE_OP(dpnp_bitwise_or_c, input_elem1 | input_elem2)
MACRO_CUSTOM_2ARG_1TYPE_OP(dpnp_bitwise_xor_c, input_elem1 ^ input_elem2)
MACRO_CUSTOM_2ARG_1TYPE_OP(dpnp_left_shift_c, input_elem1 << input_elem2)
MACRO_CUSTOM_2ARG_1TYPE_OP(dpnp_right_shift_c, input_elem1 >> input_elem2)

#undef MACRO_CUSTOM_2ARG_1TYPE_OP
