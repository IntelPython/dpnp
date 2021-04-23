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
 * Macro `MACRO_1ARG_1TYPE_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 * - mkl operation used to calculate the result
 *
 */

#ifndef MACRO_1ARG_1TYPE_OP
#error "MACRO_1ARG_1TYPE_OP is not defined"
#endif

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)                                                   \
    /** @ingroup BACKEND_API                                                                                         */ \
    /** @brief Per element operation function __name__                                                               */ \
    /**                                                                                                              */ \
    /** Function "__name__" executes operator "__operation1__" over each element of the array                        */ \
    /**                                                                                                              */ \
    /** @param[in]  array1   Input array.                                                                            */ \
    /** @param[out] result1  Output array.                                                                           */ \
    /** @param[in]  size     Number of elements in the input array.                                                  */ \
    template <typename _DataType>                                                                                       \
    void __name__(void* array1, void* result1, size_t size);

#endif

MACRO_1ARG_1TYPE_OP(dpnp_conjugate_c, std::conj(input_elem), DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_1TYPE_OP(dpnp_copy_c, input_elem, DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_1TYPE_OP(dpnp_erf_c,
                    cl::sycl::erf((double)input_elem),
                    oneapi::mkl::vm::erf(DPNP_QUEUE, size, array1, result)) // no sycl::erf for int and long
MACRO_1ARG_1TYPE_OP(dpnp_negative_c, -input_elem, DPNP_QUEUE.submit(kernel_func))
MACRO_1ARG_1TYPE_OP(dpnp_recip_c,
                    _DataType(1) / input_elem,
                    DPNP_QUEUE.submit(kernel_func)) // error: no member named 'recip' in namespace 'cl::sycl'
MACRO_1ARG_1TYPE_OP(dpnp_sign_c,
                    cl::sycl::sign((double)input_elem),
                    DPNP_QUEUE.submit(kernel_func)) // no sycl::sign for int and long
MACRO_1ARG_1TYPE_OP(dpnp_square_c, input_elem* input_elem, oneapi::mkl::vm::sqr(DPNP_QUEUE, size, array1, result))

#undef MACRO_1ARG_1TYPE_OP
