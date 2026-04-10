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

import dpnp.tensor as dpt

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _compare_dtypes,
    _usm_types,
)


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_greater_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.zeros(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.greater(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.greater(
        np.zeros(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.zeros(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.greater(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.greater(
        np.zeros(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("op_dtype", ["c8", "c16"])
def test_greater_complex_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 127
    ar1_np_real = np.random.randint(0, 10, sz)
    ar1_np_imag = np.random.randint(0, 10, sz)
    ar1_np = ar1_np_real + 1j * ar1_np_imag
    ar1 = dpt.asarray(ar1_np, dtype=op_dtype)

    ar2_np_real = np.random.randint(0, 10, sz)
    ar2_np_imag = np.random.randint(0, 10, sz)
    ar2_np = ar2_np_real + 1j * ar2_np_imag
    ar2 = dpt.asarray(ar2_np, dtype=op_dtype)

    r = dpt.greater(ar1, ar2)
    expected = np.greater(ar1_np, ar2_np)
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == expected.shape
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.greater(ar1[::-2], ar2[::2])
    expected1 = np.greater(ar1_np[::-2], ar2_np[::2])
    assert _compare_dtypes(r.dtype, expected1.dtype, sycl_queue=q)
    assert r1.shape == expected1.shape
    assert (dpt.asnumpy(r1) == expected1).all()

    ar3 = dpt.asarray([1.0 + 9j, 2.0 + 0j, 2.0 + 1j, 2.0 + 2j], dtype=op_dtype)
    ar4 = dpt.asarray([2.0 + 0j, dpt.nan, dpt.inf, -dpt.inf], dtype=op_dtype)

    ar3_np = dpt.asnumpy(ar3)
    ar4_np = dpt.asnumpy(ar4)

    r2 = dpt.greater(ar3, ar4)
    with np.errstate(invalid="ignore"):
        expected2 = np.greater(ar3_np, ar4_np)
    assert (dpt.asnumpy(r2) == expected2).all()

    r3 = dpt.greater(ar4, ar4)
    with np.errstate(invalid="ignore"):
        expected3 = np.greater(ar4_np, ar4_np)
    assert (dpt.asnumpy(r3) == expected3).all()


def test_greater_complex_float():
    get_queue_or_skip()

    ar1 = dpt.asarray([1.0 + 9j, 2.0 + 0j, 2.0 + 1j, 2.0 + 2j], dtype="c8")
    ar2 = dpt.full((4,), 2, dtype="f4")

    ar1_np = dpt.asnumpy(ar1)
    ar2_np = dpt.asnumpy(ar2)

    r = dpt.greater(ar1, ar2)
    expected = np.greater(ar1_np, ar2_np)
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.greater(ar2, ar1)
    expected1 = np.greater(ar2_np, ar1_np)
    assert (dpt.asnumpy(r1) == expected1).all()
    with np.errstate(invalid="ignore"):
        for tp in [dpt.nan, dpt.inf, -dpt.inf]:

            ar3 = dpt.full((4,), tp)
            ar3_np = dpt.asnumpy(ar3)

            r2 = dpt.greater(ar1, ar3)
            expected2 = np.greater(ar1_np, ar3_np)
            assert (dpt.asnumpy(r2) == expected2).all()

            r3 = dpt.greater(ar3, ar1)
            expected3 = np.greater(ar3_np, ar1_np)
            assert (dpt.asnumpy(r3) == expected3).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_greater_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.greater(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpt.get_coerced_usm_type((op1_usm_type, op2_usm_type))
    assert r.usm_type == expected_usm_type


def test_greater_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    ar2 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.greater(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.greater(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.greater(ar1, ar2, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.greater(ar1, ar2, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.ones((20, 20), dtype="i4", order="F")
    ar2 = dpt.ones((20, 20), dtype="i4", order="F")
    r1 = dpt.greater(ar1, ar2, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.greater(ar1, ar2, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.greater(ar1, ar2, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.greater(ar1, ar2, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.greater(ar1, ar2, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    ar2 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.greater(ar1, ar2, order="K")
    assert r4.strides == (-1, 20)


def test_greater_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(1, 6, dtype="i4")

    r = dpt.greater(m, v)

    expected = np.greater(
        np.ones((100, 5), dtype="i4"), np.arange(1, 6, dtype="i4")
    )
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()

    r2 = dpt.greater(v, m)
    expected2 = np.greater(
        np.arange(1, 6, dtype="i4"), np.ones((100, 5), dtype="i4")
    )
    assert (dpt.asnumpy(r2) == expected2.astype(r2.dtype)).all()


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_greater_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.ones((10, 10), dtype=arr_dt, sycl_queue=q)
    py_ones = (
        bool(1),
        int(1),
        float(1),
        complex(1),
        np.float32(1),
        ctypes.c_int(1),
    )
    for sc in py_ones:
        R = dpt.greater(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.greater(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


class MockArray:
    def __init__(self, arr):
        self.data_ = arr

    @property
    def __sycl_usm_array_interface__(self):
        return self.data_.__sycl_usm_array_interface__


def test_greater_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)
    b = dpt.ones(10)
    c = MockArray(b)
    r = dpt.greater(a, c)
    assert isinstance(r, dpt.usm_ndarray)


def test_greater_canary_mock_array():
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
        dpt.greater(a, c)


def test_greater_mixed_integer_kinds():
    get_queue_or_skip()

    x1 = dpt.flip(dpt.arange(-9, 1, dtype="i8"))
    x2 = dpt.arange(10, dtype="u8")

    # u8 - i8
    res = dpt.greater(x2, x1)
    assert dpt.all(res[1:])
    assert not res[0]
    # i8 - u8
    assert not dpt.any(dpt.greater(x1, x2))

    # Python scalar
    assert dpt.all(dpt.greater(x2, -1))
    assert not dpt.any(dpt.greater(-1, x2))


def test_greater_very_large_py_int():
    get_queue_or_skip()

    py_int = dpt.iinfo(dpt.int64).max + 10

    x = dpt.asarray(3, dtype="u8")
    assert py_int > x
    assert not dpt.greater(x, py_int)

    x = dpt.asarray(py_int, dtype="u8")
    assert x > -1
    assert not dpt.greater(-1, x)
