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
 * Macro `MACRO_CUSTOM_1ARG_2TYPES_MKL_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 *
 */

#ifndef MACRO_CUSTOM_1ARG_2TYPES_MKL_OP
#error "MACRO_CUSTOM_1ARG_2TYPES_MKL_OP is not defined"
#endif

MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(acos, cl::sycl::acos, oneapi::mkl::vm::acos)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(acosh, cl::sycl::acosh, oneapi::mkl::vm::acosh)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(asin, cl::sycl::asin, oneapi::mkl::vm::asin)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(asinh, cl::sycl::asinh, oneapi::mkl::vm::asinh)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(atan, cl::sycl::atan, oneapi::mkl::vm::atan)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(atanh, cl::sycl::atanh, oneapi::mkl::vm::atanh)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(cbrt, cl::sycl::cbrt, oneapi::mkl::vm::cbrt)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(ceil, cl::sycl::ceil, oneapi::mkl::vm::ceil)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(cos, cl::sycl::cos, oneapi::mkl::vm::cos)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(cosh, cl::sycl::cosh, oneapi::mkl::vm::cosh)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(exp2, cl::sycl::exp2, oneapi::mkl::vm::exp2)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(exp, cl::sycl::exp, oneapi::mkl::vm::exp)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(expm1, cl::sycl::expm1, oneapi::mkl::vm::expm1)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(fabs, cl::sycl::fabs, oneapi::mkl::vm::abs)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(floor, cl::sycl::floor, oneapi::mkl::vm::floor)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(log10, cl::sycl::log10, oneapi::mkl::vm::log10)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(log1p, cl::sycl::log1p, oneapi::mkl::vm::log1p)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(log2, cl::sycl::log2, oneapi::mkl::vm::log2)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(log, cl::sycl::log, oneapi::mkl::vm::ln)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(sin, cl::sycl::sin, oneapi::mkl::vm::sin)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(sinh, cl::sycl::sinh, oneapi::mkl::vm::sinh)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(sqrt, cl::sycl::sqrt, oneapi::mkl::vm::sqrt)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(tan, cl::sycl::tan, oneapi::mkl::vm::tan)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(tanh, cl::sycl::tanh, oneapi::mkl::vm::tanh)
MACRO_CUSTOM_1ARG_2TYPES_MKL_OP(trunc, cl::sycl::trunc, oneapi::mkl::vm::trunc)

#undef MACRO_CUSTOM_1ARG_2TYPES_MKL_OP
