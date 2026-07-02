# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import pytest

import dpnp.tensor as dpt
from dpnp.tests.tensor.helper import get_queue_or_skip

from .utils import _usm_types


@pytest.mark.parametrize("usm_type", _usm_types)
class TestUnaryUSMType:
    def unary_elementwise(self, fn, usm_type, dtype="f4"):
        q = get_queue_or_skip()
        x = dpt.asarray(
            [1, 2, 3, 4], dtype=dtype, usm_type=usm_type, sycl_queue=q
        )
        return getattr(dpt, fn)(x)

    def test_abs(self, usm_type):
        self.unary_elementwise("abs", usm_type)

    def test_acos(self, usm_type):
        self.unary_elementwise("acos", usm_type)

    def test_acosh(self, usm_type):
        self.unary_elementwise("acosh", usm_type)

    def test_angle(self, usm_type):
        self.unary_elementwise("angle", usm_type, dtype="c8")

    def test_asin(self, usm_type):
        self.unary_elementwise("asin", usm_type)

    def test_asinh(self, usm_type):
        self.unary_elementwise("asinh", usm_type)

    def test_atan(self, usm_type):
        self.unary_elementwise("atan", usm_type)

    def test_atanh(self, usm_type):
        self.unary_elementwise("atanh", usm_type)

    def test_bitwise_invert(self, usm_type):
        self.unary_elementwise("bitwise_invert", usm_type, dtype="i4")

    def test_cbrt(self, usm_type):
        self.unary_elementwise("cbrt", usm_type)

    def test_ceil(self, usm_type):
        self.unary_elementwise("ceil", usm_type)

    def test_conj(self, usm_type):
        self.unary_elementwise("conj", usm_type)

    def test_cos(self, usm_type):
        self.unary_elementwise("cos", usm_type)

    def test_cosh(self, usm_type):
        self.unary_elementwise("cosh", usm_type)

    def test_exp(self, usm_type):
        self.unary_elementwise("exp", usm_type)

    def test_exp2(self, usm_type):
        self.unary_elementwise("exp2", usm_type)

    def test_expm1(self, usm_type):
        self.unary_elementwise("expm1", usm_type)

    def test_floor(self, usm_type):
        self.unary_elementwise("floor", usm_type)

    def test_imag(self, usm_type):
        self.unary_elementwise("imag", usm_type)

    def test_isfinite(self, usm_type):
        self.unary_elementwise("isfinite", usm_type)

    def test_isinf(self, usm_type):
        self.unary_elementwise("isinf", usm_type)

    def test_isnan(self, usm_type):
        self.unary_elementwise("isnan", usm_type)

    def test_log(self, usm_type):
        self.unary_elementwise("log", usm_type)

    def test_log1p(self, usm_type):
        self.unary_elementwise("log1p", usm_type)

    def test_log2(self, usm_type):
        self.unary_elementwise("log2", usm_type)

    def test_log10(self, usm_type):
        self.unary_elementwise("log10", usm_type)

    def test_logical_not(self, usm_type):
        self.unary_elementwise("logical_not", usm_type, dtype="i4")

    def test_negative(self, usm_type):
        self.unary_elementwise("negative", usm_type)

    def test_positive(self, usm_type):
        self.unary_elementwise("positive", usm_type)

    def test_proj(self, usm_type):
        self.unary_elementwise("proj", usm_type, dtype="c8")

    def test_real(self, usm_type):
        self.unary_elementwise("real", usm_type, dtype="c8")

    def test_reciprocal(self, usm_type):
        self.unary_elementwise("reciprocal", usm_type)

    def test_round(self, usm_type):
        self.unary_elementwise("round", usm_type)

    def test_rsqrt(self, usm_type):
        self.unary_elementwise("rsqrt", usm_type)

    def test_sign(self, usm_type):
        self.unary_elementwise("sign", usm_type)

    def test_signbit(self, usm_type):
        self.unary_elementwise("signbit", usm_type)

    def test_sin(self, usm_type):
        self.unary_elementwise("sin", usm_type)

    def test_sinh(self, usm_type):
        self.unary_elementwise("sinh", usm_type)

    def test_square(self, usm_type):
        self.unary_elementwise("square", usm_type)

    def test_sqrt(self, usm_type):
        self.unary_elementwise("sqrt", usm_type)

    def test_tan(self, usm_type):
        self.unary_elementwise("tan", usm_type)

    def test_tanh(self, usm_type):
        self.unary_elementwise("tanh", usm_type)

    def test_trunc(self, usm_type):
        self.unary_elementwise("trunc", usm_type)

    def test_usm_basic(self, usm_type):
        q = get_queue_or_skip()

        sz = 128
        dt = dpt.int32
        x = dpt.ones(sz, dtype=dt, usm_type=usm_type, sycl_queue=q)

        r = dpt.abs(x)
        assert isinstance(r, dpt.usm_ndarray)
        assert r.usm_type == x.usm_type


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
class TestBinaryUSMType:
    def binary_elementwise(self, fn, op1_usm_type, op2_usm_type, dtype="f4"):
        q = get_queue_or_skip()
        x = dpt.asarray(
            [1, 2, 3, 4, 5, 6], dtype=dtype, usm_type=op1_usm_type, sycl_queue=q
        )
        y = dpt.asarray(
            [1, 2, 3, 4, 5, 6], dtype=dtype, usm_type=op2_usm_type, sycl_queue=q
        )
        return getattr(dpt, fn)(x, y)

    def test_add(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("add", op1_usm_type, op2_usm_type)

    def test_atan2(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("atan2", op1_usm_type, op2_usm_type)

    def test_bitwise_and(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise(
            "bitwise_and", op1_usm_type, op2_usm_type, dtype="i4"
        )

    def test_bitwise_left_shift(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise(
            "bitwise_left_shift", op1_usm_type, op2_usm_type, dtype="i4"
        )

    def test_bitwise_or(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise(
            "bitwise_or", op1_usm_type, op2_usm_type, dtype="i4"
        )

    def test_bitwise_right_shift(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise(
            "bitwise_right_shift", op1_usm_type, op2_usm_type, dtype="i4"
        )

    def test_bitwise_xor(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise(
            "bitwise_xor", op1_usm_type, op2_usm_type, dtype="i4"
        )

    def test_copysign(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("copysign", op1_usm_type, op2_usm_type)

    def test_divide(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("divide", op1_usm_type, op2_usm_type)

    def test_equal(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("equal", op1_usm_type, op2_usm_type)

    def test_floor_divide(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("floor_divide", op1_usm_type, op2_usm_type)

    def test_hypot(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("hypot", op1_usm_type, op2_usm_type)

    def test_greater(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("greater", op1_usm_type, op2_usm_type)

    def test_greater_equal(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("greater_equal", op1_usm_type, op2_usm_type)

    def test_less(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("less", op1_usm_type, op2_usm_type)

    def test_less_equal(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("less_equal", op1_usm_type, op2_usm_type)

    def test_logaddexp(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("logaddexp", op1_usm_type, op2_usm_type)

    def test_logical_and(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("logical_and", op1_usm_type, op2_usm_type)

    def test_logical_or(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("logical_or", op1_usm_type, op2_usm_type)

    def test_logical_xor(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("logical_xor", op1_usm_type, op2_usm_type)

    def test_maximum(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("maximum", op1_usm_type, op2_usm_type)

    def test_minimum(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("minimum", op1_usm_type, op2_usm_type)

    def test_multiply(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("multiply", op1_usm_type, op2_usm_type)

    def test_nextafter(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("nextafter", op1_usm_type, op2_usm_type)

    def test_not_equal(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("not_equal", op1_usm_type, op2_usm_type)

    def test_pow(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("pow", op1_usm_type, op2_usm_type)

    def test_remainder(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("remainder", op1_usm_type, op2_usm_type)

    def test_subtract(self, op1_usm_type, op2_usm_type):
        self.binary_elementwise("subtract", op1_usm_type, op2_usm_type)

    def test_binary_usm_type_coercion(self, op1_usm_type, op2_usm_type):
        q = get_queue_or_skip()

        sz = 128
        dt = dpt.int32
        ar1 = dpt.ones(sz, dtype=dt, usm_type=op1_usm_type, sycl_queue=q)
        ar2 = dpt.ones_like(ar1, dtype=dt, usm_type=op2_usm_type, sycl_queue=q)

        r = dpt.add(ar1, ar2)
        assert isinstance(r, dpt.usm_ndarray)
        expected_usm_type = dpt.get_coerced_usm_type(
            (op1_usm_type, op2_usm_type)
        )
        assert r.usm_type == expected_usm_type
