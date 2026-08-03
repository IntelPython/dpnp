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

import numpy as np
import pytest

import dpnp.tensor as dpt
from dpnp.tensor._type_utils import _can_cast

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _compare_dtypes,
    _no_complex_dtypes,
)


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes)
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes)
def test_remainder_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.remainder(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.remainder(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.remainder(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.remainder(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("dt", _no_complex_dtypes[1:8:2])
def test_remainder_negative_integers(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.arange(-5, -1, 1, dtype=dt, sycl_queue=q)
    x_np = np.arange(-5, -1, 1, dtype=dt)
    val = 3

    r1 = dpt.remainder(x, val)
    expected = np.remainder(x_np, val)
    assert (dpt.asnumpy(r1) == expected).all()

    r2 = dpt.remainder(val, x)
    expected = np.remainder(val, x_np)
    assert (dpt.asnumpy(r2) == expected).all()


def test_remainder_integer_zero():
    get_queue_or_skip()

    for dt in ["i4", "u4"]:
        x = dpt.ones(1, dtype=dt)
        y = dpt.zeros_like(x)

        assert (dpt.asnumpy(dpt.remainder(x, y)) == np.zeros(1, dtype=dt)).all()

        x = dpt.astype(x, dt)
        y = dpt.zeros_like(x)

        assert (dpt.asnumpy(dpt.remainder(x, y)) == np.zeros(1, dtype=dt)).all()


@pytest.mark.parametrize("dt", _no_complex_dtypes[9:])
def test_remainder_negative_floats(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.linspace(-5, 5, 20, dtype=dt, sycl_queue=q)
    x_np = np.linspace(-5, 5, 20, dtype=dt)
    val = 3

    tol = 8 * dpt.finfo(dt).resolution

    r1 = dpt.remainder(x, val)
    expected = np.remainder(x_np, val)
    with np.errstate(invalid="ignore"):
        np.allclose(
            dpt.asnumpy(r1), expected, rtol=tol, atol=tol, equal_nan=True
        )

    r2 = dpt.remainder(val, x)
    expected = np.remainder(val, x_np)
    with np.errstate(invalid="ignore"):
        np.allclose(
            dpt.asnumpy(r2), expected, rtol=tol, atol=tol, equal_nan=True
        )


def test_remainder_special_cases():
    get_queue_or_skip()

    lhs = [dpt.nan, dpt.inf, 0.0, -0.0, -0.0, 1.0, dpt.inf, -dpt.inf]
    rhs = [dpt.nan, dpt.inf, -0.0, 1.0, 1.0, 0.0, 1.0, -1.0]

    x, y = dpt.asarray(lhs, dtype="f4"), dpt.asarray(rhs, dtype="f4")

    x_np, y_np = np.asarray(lhs, dtype="f4"), np.asarray(rhs, dtype="f4")

    res = dpt.remainder(x, y)

    with np.errstate(invalid="ignore"):
        np.allclose(dpt.asnumpy(res), np.remainder(x_np, y_np))


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_remainder_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64, casting="same_kind"):
        ar1 %= ar2
        assert dpt.all(ar1 == dpt.zeros(ar1.shape, dtype=ar1.dtype))

        ar3 = dpt.ones(sz, dtype=op1_dtype)
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

        ar3[::-1] %= ar4[::2]
        assert dpt.all(ar3 == dpt.zeros(ar3.shape, dtype=ar3.dtype))

    else:
        with pytest.raises(ValueError):
            ar1 %= ar2


def test_remainder_inplace_basic():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    expected = x & 1
    x %= 2

    assert dpt.all(x == expected)
