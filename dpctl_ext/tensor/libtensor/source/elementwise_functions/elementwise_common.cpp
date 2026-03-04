//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_elementwise_impl
/// extension, specifically functions for elementwise operations.
//===---------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "abs.hpp"
#include "acos.hpp"
#include "acosh.hpp"
// #include "add.hpp"
#include "angle.hpp"
#include "asin.hpp"
#include "asinh.hpp"
#include "atan.hpp"
// #include "atan2.hpp"
#include "atanh.hpp"
// #include "bitwise_and.hpp"
#include "bitwise_invert.hpp"
// #include "bitwise_left_shift.hpp"
// #include "bitwise_or.hpp"
// #include "bitwise_right_shift.hpp"
// #include "bitwise_xor.hpp"
// #include "cbrt.hpp"
#include "ceil.hpp"
#include "conj.hpp"
// #include "copysign.hpp"
#include "cos.hpp"
#include "cosh.hpp"
// #include "equal.hpp"
#include "exp.hpp"
// #include "exp2.hpp"
#include "expm1.hpp"
#include "floor.hpp"
// #include "floor_divide.hpp"
// #include "greater.hpp"
// #include "greater_equal.hpp"
// #include "hypot.hpp"
#include "imag.hpp"
#include "isfinite.hpp"
#include "isinf.hpp"
#include "isnan.hpp"
// #include "less.hpp"
// #include "less_equal.hpp"
#include "log.hpp"
#include "log10.hpp"
#include "log1p.hpp"
#include "log2.hpp"
// #include "logaddexp.hpp"
// #include "logical_and.hpp"
// #include "logical_not.hpp"
// #include "logical_or.hpp"
// #include "logical_xor.hpp"
// #include "maximum.hpp"
// #include "minimum.hpp"
// #include "multiply.hpp"
// #include "negative.hpp"
// #include "nextafter.hpp"
// #include "not_equal.hpp"
// #include "positive.hpp"
// #include "pow.hpp"
// #include "proj.hpp"
// #include "real.hpp"
// #include "reciprocal.hpp"
// #include "remainder.hpp"
// #include "round.hpp"
// #include "rsqrt.hpp"
// #include "sign.hpp"
// #include "signbit.hpp"
// #include "sin.hpp"
// #include "sinh.hpp"
// #include "sqrt.hpp"
// #include "square.hpp"
// #include "subtract.hpp"
// #include "tan.hpp"
// #include "tanh.hpp"
// #include "true_divide.hpp"
// #include "trunc.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;

/*! @brief Add elementwise functions to Python module */
void init_elementwise_functions(py::module_ m)
{
    init_abs(m);
    init_acos(m);
    init_acosh(m);
    // init_add(m);
    init_angle(m);
    init_asin(m);
    init_asinh(m);
    init_atan(m);
    // init_atan2(m);
    init_atanh(m);
    // init_bitwise_and(m);
    init_bitwise_invert(m);
    // init_bitwise_left_shift(m);
    // init_bitwise_or(m);
    // init_bitwise_right_shift(m);
    // init_bitwise_xor(m);
    // init_cbrt(m);
    init_ceil(m);
    init_conj(m);
    // init_copysign(m);
    init_cos(m);
    init_cosh(m);
    // init_divide(m);
    // init_equal(m);
    init_exp(m);
    // init_exp2(m);
    init_expm1(m);
    init_floor(m);
    // init_floor_divide(m);
    // init_greater(m);
    // init_greater_equal(m);
    // init_hypot(m);
    init_imag(m);
    init_isfinite(m);
    init_isinf(m);
    init_isnan(m);
    // init_less(m);
    // init_less_equal(m);
    init_log(m);
    init_log10(m);
    init_log1p(m);
    init_log2(m);
    // init_logaddexp(m);
    // init_logical_and(m);
    // init_logical_not(m);
    // init_logical_or(m);
    // init_logical_xor(m);
    // init_maximum(m);
    // init_minimum(m);
    // init_multiply(m);
    // init_nextafter(m);
    // init_negative(m);
    // init_not_equal(m);
    // init_positive(m);
    // init_pow(m);
    // init_proj(m);
    // init_real(m);
    // init_reciprocal(m);
    // init_remainder(m);
    // init_round(m);
    // init_rsqrt(m);
    // init_sign(m);
    // init_signbit(m);
    // init_sin(m);
    // init_sinh(m);
    // init_sqrt(m);
    // init_square(m);
    // init_subtract(m);
    // init_tan(m);
    // init_tanh(m);
    // init_trunc(m);
}

} // namespace dpctl::tensor::py_internal
