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

import ctypes

import dpctl
import numpy as np
import pytest

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
from dpctl_ext.tensor._type_utils import _can_cast

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _compare_dtypes,
    _integral_dtypes,
    _no_complex_dtypes,
    _usm_types,
)


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_floor_divide_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.floor_divide(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.floor_divide(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.floor_divide(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.floor_divide(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_floor_divide_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.floor_divide(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_floor_divide_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    ar2 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.floor_divide(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.floor_divide(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.floor_divide(ar1, ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.floor_divide(ar1, ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.ones((20, 20), dtype="i4", order="F")
    ar2 = dpt.ones((20, 20), dtype="i4", order="F")
    r1 = dpt.floor_divide(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.floor_divide(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.floor_divide(ar1, ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.floor_divide(ar1, ar2, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.floor_divide(ar1, ar2, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.floor_divide(ar1, ar2, order="K")
    assert r4.strides == (-1, 20)


def test_floor_divide_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(1, 6, dtype="i4")

    r = dpt.floor_divide(m, v)

    expected = np.floor_divide(
        np.ones((100, 5), dtype="i4"), np.arange(1, 6, dtype="i4")
    )
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()

    r2 = dpt.floor_divide(v, m)
    expected2 = np.floor_divide(
        np.arange(1, 6, dtype="i4"), np.ones((100, 5), dtype="i4")
    )
    assert (dpt.asnumpy(r2) == expected2.astype(r2.dtype)).all()


@pytest.mark.parametrize("arr_dt", _no_complex_dtypes[1:])
def test_floor_divide_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.ones((10, 10), dtype=arr_dt, sycl_queue=q)
    py_ones = (
        bool(1),
        int(1),
        float(1),
        np.float32(1),
        ctypes.c_int(1),
    )
    for sc in py_ones:
        R = dpt.floor_divide(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.floor_divide(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


class MockArray:
    def __init__(self, arr):
        self.data_ = arr

    @property
    def __sycl_usm_array_interface__(self):
        return self.data_.__sycl_usm_array_interface__


def test_floor_divide_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)
    b = dpt.ones(10)
    c = MockArray(b)
    r = dpt.floor_divide(a, c)
    assert isinstance(r, dpt.usm_ndarray)


def test_floor_divide_canary_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)

    class Canary:
        def __init__(self):
            pass

        @property
        def __sycl_usm_array_interface__(self):
            return None

    c = Canary()
    with pytest.raises(ValueError):
        dpt.floor_divide(a, c)


def test_floor_divide_gh_1247():
    get_queue_or_skip()

    x = dpt.ones(1, dtype="i4")
    res = dpt.floor_divide(x, -2)
    np.testing.assert_array_equal(
        dpt.asnumpy(res), np.full(res.shape, -1, dtype=res.dtype)
    )

    x = dpt.full(1, -1, dtype="i4")
    res = dpt.floor_divide(x, 2)
    np.testing.assert_array_equal(
        dpt.asnumpy(res), np.full(res.shape, -1, dtype=res.dtype)
    )


@pytest.mark.parametrize("dtype", _integral_dtypes)
def test_floor_divide_integer_zero(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.arange(10, dtype=dtype, sycl_queue=q)
    y = dpt.zeros_like(x, sycl_queue=q)
    res = dpt.floor_divide(x, y)
    np.testing.assert_array_equal(
        dpt.asnumpy(res), np.zeros(x.shape, dtype=res.dtype)
    )


def test_floor_divide_special_cases():
    q = get_queue_or_skip()

    x = dpt.empty(1, dtype="f4", sycl_queue=q)
    y = dpt.empty_like(x)
    x[0], y[0] = dpt.inf, dpt.inf
    res = dpt.floor_divide(x, y)
    with np.errstate(all="ignore"):
        res_np = np.floor_divide(dpt.asnumpy(x), dpt.asnumpy(y))
        np.testing.assert_array_equal(dpt.asnumpy(res), res_np)

    x[0], y[0] = 0.0, -1.0
    res = dpt.floor_divide(x, y)
    x_np = dpt.asnumpy(x)
    y_np = dpt.asnumpy(y)
    res_np = np.floor_divide(x_np, y_np)
    np.testing.assert_array_equal(dpt.asnumpy(res), res_np)

    res = dpt.floor_divide(y, x)
    with np.errstate(all="ignore"):
        res_np = np.floor_divide(y_np, x_np)
        np.testing.assert_array_equal(dpt.asnumpy(res), res_np)

    x[0], y[0] = -1.0, dpt.inf
    res = dpt.floor_divide(x, y)
    np.testing.assert_array_equal(
        dpt.asnumpy(res), np.asarray([-0.0], dtype="f4")
    )

    res = dpt.floor_divide(y, x)
    np.testing.assert_array_equal(
        dpt.asnumpy(res), np.asarray([-dpt.inf], dtype="f4")
    )

    x[0], y[0] = 1.0, dpt.nan
    res = dpt.floor_divide(x, y)
    res_np = np.floor_divide(dpt.asnumpy(x), dpt.asnumpy(y))
    np.testing.assert_array_equal(dpt.asnumpy(res), res_np)


@pytest.mark.parametrize("dtype", _no_complex_dtypes[1:])
def test_divide_inplace_python_scalar(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.zeros((10, 10), dtype=dtype, sycl_queue=q)
    dt_kind = X.dtype.kind
    if dt_kind in "ui":
        X //= int(1)
    elif dt_kind == "f":
        X //= float(1)


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes[1:])
def test_floor_divide_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype, sycl_queue=q)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    # out array only valid if it is inexact
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64, casting="same_kind"):
        ar1 //= ar2
        assert dpt.all(ar1 == 1)

        ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)[::2]
        ar3 //= ar4
        assert dpt.all(ar3 == 1)
    else:
        with pytest.raises(ValueError):
            ar1 //= ar2
            dpt.floor_divide(ar1, ar2, out=ar1)
