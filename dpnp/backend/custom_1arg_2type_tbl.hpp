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
 * Macro `MACRO_CUSTOM_1ARG_2TYPES_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 *
 */

#ifndef MACRO_CUSTOM_1ARG_2TYPES_OP
#error "MACRO_CUSTOM_1ARG_2TYPES_OP is not defined"
#endif

MACRO_CUSTOM_1ARG_2TYPES_OP(acos, cl::sycl::acos(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(acosh, cl::sycl::acosh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(asin, cl::sycl::asin(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(asinh, cl::sycl::asinh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(atan, cl::sycl::atan(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(atanh, cl::sycl::atanh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(cbrt, cl::sycl::cbrt(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(cos, cl::sycl::cos(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(cosh, cl::sycl::cosh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(degrees, cl::sycl::degrees(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(exp, cl::sycl::exp(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(exp2, cl::sycl::exp2(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(expm1, cl::sycl::expm1(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(log, cl::sycl::log(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(log10, cl::sycl::log10(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(log1p, cl::sycl::log1p(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(log2, cl::sycl::log2(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(radians, cl::sycl::radians(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(sin, cl::sycl::sin(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(sinh, cl::sycl::sinh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(sqrt, cl::sycl::sqrt(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(tan, cl::sycl::tan(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(tanh, cl::sycl::tanh(input_elem))

#undef MACRO_CUSTOM_1ARG_2TYPES_OP
