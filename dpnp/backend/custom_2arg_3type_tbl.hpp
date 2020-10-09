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
 * Macro `MACRO_CUSTOM_2ARG_3TYPES_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 *
 */

#ifndef MACRO_CUSTOM_2ARG_3TYPES_OP
#error "MACRO_CUSTOM_2ARG_3TYPES_OP is not defined"
#endif

MACRO_CUSTOM_2ARG_3TYPES_OP(add, input_elem1 + input_elem2, oneapi::mkl::vm::add)
MACRO_CUSTOM_2ARG_3TYPES_OP(arctan2, cl::sycl::atan2((double)input_elem1, (double)input_elem2), oneapi::mkl::vm::atan2)
MACRO_CUSTOM_2ARG_3TYPES_OP(divide, input_elem1 / input_elem2, oneapi::mkl::vm::div)
MACRO_CUSTOM_2ARG_3TYPES_OP(fmod, cl::sycl::fmod((double)input_elem1, (double)input_elem2), oneapi::mkl::vm::fmod)
MACRO_CUSTOM_2ARG_3TYPES_OP(hypot, cl::sycl::hypot((double)input_elem1, (double)input_elem2), oneapi::mkl::vm::hypot)
MACRO_CUSTOM_2ARG_3TYPES_OP(maximum, cl::sycl::max(input_elem1, input_elem2), oneapi::mkl::vm::fmax)
MACRO_CUSTOM_2ARG_3TYPES_OP(minimum, cl::sycl::min(input_elem1, input_elem2), oneapi::mkl::vm::fmin)
MACRO_CUSTOM_2ARG_3TYPES_OP(multiply, input_elem1* input_elem2, oneapi::mkl::vm::mul)
MACRO_CUSTOM_2ARG_3TYPES_OP(power, cl::sycl::powr((double)input_elem1, (double)input_elem2), oneapi::mkl::vm::pow)
MACRO_CUSTOM_2ARG_3TYPES_OP(subtract, input_elem1 - input_elem2, oneapi::mkl::vm::sub)

#undef MACRO_CUSTOM_2ARG_3TYPES_OP
