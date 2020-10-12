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
 * Macros `MACRO_CUSTOM_1ARG_2TYPES_OP` and `MACRO_CUSTOM_1ARG_2TYPES_2OPS` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 * - mkl operation used to calculate the result
 *
 */

#ifndef MACRO_CUSTOM_1ARG_2TYPES_OP
#error "MACRO_CUSTOM_1ARG_2TYPES_OP is not defined"
#endif

MACRO_CUSTOM_1ARG_2TYPES_OP(degrees, cl::sycl::degrees(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(radians, cl::sycl::radians(input_elem))

#undef MACRO_CUSTOM_1ARG_2TYPES_OP


#ifndef MACRO_CUSTOM_1ARG_2TYPES_2OPS
#error "MACRO_CUSTOM_1ARG_2TYPES_2OPS is not defined"
#endif

MACRO_CUSTOM_1ARG_2TYPES_2OPS(acos, cl::sycl::acos(input_elem), oneapi::mkl::vm::acos)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(acosh, cl::sycl::acosh(input_elem), oneapi::mkl::vm::acosh)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(asin, cl::sycl::asin(input_elem), oneapi::mkl::vm::asin)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(asinh, cl::sycl::asinh(input_elem), oneapi::mkl::vm::asinh)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(atan, cl::sycl::atan(input_elem), oneapi::mkl::vm::atan)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(atanh, cl::sycl::atanh(input_elem), oneapi::mkl::vm::atanh)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(cbrt, cl::sycl::cbrt(input_elem), oneapi::mkl::vm::cbrt)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(ceil, cl::sycl::ceil(input_elem), oneapi::mkl::vm::ceil)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(cos, cl::sycl::cos(input_elem), oneapi::mkl::vm::cos)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(cosh, cl::sycl::cosh(input_elem), oneapi::mkl::vm::cosh)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(exp2, cl::sycl::exp2(input_elem), oneapi::mkl::vm::exp2)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(exp, cl::sycl::exp(input_elem), oneapi::mkl::vm::exp)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(expm1, cl::sycl::expm1(input_elem), oneapi::mkl::vm::expm1)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(fabs, cl::sycl::fabs(input_elem), oneapi::mkl::vm::abs)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(floor, cl::sycl::floor(input_elem), oneapi::mkl::vm::floor)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(log10, cl::sycl::log10(input_elem), oneapi::mkl::vm::log10)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(log1p, cl::sycl::log1p(input_elem), oneapi::mkl::vm::log1p)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(log2, cl::sycl::log2(input_elem), oneapi::mkl::vm::log2)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(log, cl::sycl::log(input_elem), oneapi::mkl::vm::ln)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(sin, cl::sycl::sin(input_elem), oneapi::mkl::vm::sin)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(sinh, cl::sycl::sinh(input_elem), oneapi::mkl::vm::sinh)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(sqrt, cl::sycl::sqrt(input_elem), oneapi::mkl::vm::sqrt)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(tan, cl::sycl::tan(input_elem), oneapi::mkl::vm::tan)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(tanh, cl::sycl::tanh(input_elem), oneapi::mkl::vm::tanh)
MACRO_CUSTOM_1ARG_2TYPES_2OPS(trunc, cl::sycl::trunc(input_elem), oneapi::mkl::vm::trunc)

#undef MACRO_CUSTOM_1ARG_2TYPES_2OPS
