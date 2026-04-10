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

import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError

import dpnp.tensor as dpt

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

_numeric_dtypes = [
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]

_all_dtypes = ["?"] + _numeric_dtypes


@pytest.mark.parametrize("dtype", _numeric_dtypes)
def test_isin_basic(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 100
    x = dpt.arange(n, dtype=dtype, sycl_queue=q)
    test = dpt.arange(n - 1, dtype=dtype, sycl_queue=q)
    r1 = dpt.isin(x, test)
    assert dpt.all(r1[:-1])
    assert not r1[-1]
    assert r1.shape == x.shape

    # test with invert keyword
    r2 = dpt.isin(x, test, invert=True)
    assert not dpt.any(r2[:-1])
    assert r2[-1]
    assert r2.shape == x.shape


def test_isin_basic_bool():
    dt = dpt.bool
    n = 100
    x = dpt.zeros(n, dtype=dt)
    x[-1] = True
    test = dpt.zeros((), dtype=dt)
    r1 = dpt.isin(x, test)
    assert dpt.all(r1[:-1])
    assert not r1[-1]
    assert r1.shape == x.shape

    r2 = dpt.isin(x, test, invert=True)
    assert not dpt.any(r2[:-1])
    assert r2[-1]
    assert r2.shape == x.shape


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_isin_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, m = 100, 20
    x = dpt.zeros((n, m), dtype=dtype, order="F", sycl_queue=q)
    x[:, ::2] = dpt.arange(1, (m / 2) + 1, dtype=dtype, sycl_queue=q)
    x_s = x[:, ::2]
    test = dpt.arange(1, (m / 2), dtype=dtype, sycl_queue=q)
    r1 = dpt.isin(x_s, test)
    assert dpt.all(r1[:, :-1])
    assert not dpt.any(r1[:, -1])
    assert not dpt.any(x[:, 1::2])
    assert r1.shape == x_s.shape
    assert r1.flags.c_contiguous

    # test with invert keyword
    r2 = dpt.isin(x_s, test, invert=True)
    assert not dpt.any(r2[:, :-1])
    assert dpt.all(r2[:, -1])
    assert not dpt.any(x[:, 1:2])
    assert r2.shape == x_s.shape
    assert r2.flags.c_contiguous


def test_isin_strided_bool():
    dt = dpt.bool

    n, m = 100, 20
    x = dpt.zeros((n, m), dtype=dt, order="F")
    x[:, :-2:2] = True
    x_s = x[:, ::2]
    test = dpt.ones((), dtype=dt)
    r1 = dpt.isin(x_s, test)
    assert dpt.all(r1[:, :-1])
    assert not dpt.any(r1[:, -1])
    assert not dpt.any(x[:, 1::2])
    assert r1.shape == x_s.shape
    assert r1.flags.c_contiguous

    # test with invert keyword
    r2 = dpt.isin(x_s, test, invert=True)
    assert not dpt.any(r2[:, :-1])
    assert dpt.all(r2[:, -1])
    assert not dpt.any(x[:, 1:2])
    assert r2.shape == x_s.shape
    assert r2.flags.c_contiguous


@pytest.mark.parametrize("dt1", _numeric_dtypes)
@pytest.mark.parametrize("dt2", _numeric_dtypes)
def test_isin_dtype_matrix(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    sz = 10
    x = dpt.asarray([0, 1, 11], dtype=dt1, sycl_queue=q)
    test1 = dpt.arange(sz, dtype=dt2, sycl_queue=q)

    r1 = dpt.isin(x, test1)
    assert isinstance(r1, dpt.usm_ndarray)
    assert r1.dtype == dpt.bool
    assert r1.shape == x.shape
    assert not r1[-1]
    assert dpt.all(r1[0:-1])
    assert r1.sycl_queue == x.sycl_queue

    test2 = dpt.tile(dpt.asarray([[0, 1]], dtype=dt2, sycl_queue=q).mT, 2)
    r2 = dpt.isin(x, test2)
    assert isinstance(r2, dpt.usm_ndarray)
    assert r2.dtype == dpt.bool
    assert r2.shape == x.shape
    assert not r2[-1]
    assert dpt.all(r1[0:-1])
    assert r2.sycl_queue == x.sycl_queue


def test_isin_empty_inputs():
    get_queue_or_skip()

    x = dpt.ones((10, 0, 1), dtype="i4")
    test = dpt.ones((), dtype="i4")
    res1 = dpt.isin(x, test)
    assert isinstance(res1, dpt.usm_ndarray)
    assert res1.size == 0
    assert res1.shape == x.shape
    assert res1.dtype == dpt.bool

    res2 = dpt.isin(x, test, invert=True)
    assert isinstance(res2, dpt.usm_ndarray)
    assert res2.size == 0
    assert res2.shape == x.shape
    assert res2.dtype == dpt.bool

    x = dpt.ones((3, 3), dtype="i4")
    test = dpt.ones(0, dtype="i4")
    res3 = dpt.isin(x, test)
    assert isinstance(res3, dpt.usm_ndarray)
    assert res3.shape == x.shape
    assert res3.dtype == dpt.bool
    assert not dpt.all(res3)

    res4 = dpt.isin(x, test, invert=True)
    assert isinstance(res4, dpt.usm_ndarray)
    assert res4.shape == x.shape
    assert res4.dtype == dpt.bool
    assert dpt.all(res4)


def test_isin_validation():
    get_queue_or_skip()
    with pytest.raises(ExecutionPlacementError):
        dpt.isin(1, 1)
    not_bool = {}
    with pytest.raises(TypeError):
        dpt.isin(dpt.ones([1]), dpt.ones([1]), invert=not_bool)


def test_isin_special_floating_point_vals():
    get_queue_or_skip()

    # real and complex nans compare false
    x = dpt.asarray(dpt.nan, dtype="f4")
    test = dpt.asarray(dpt.nan, dtype="f4")
    assert not dpt.isin(x, test)

    x = dpt.asarray(dpt.nan, dtype="c8")
    test = dpt.asarray(dpt.nan, dtype="c8")
    assert not dpt.isin(x, test)

    # -0.0 compares equal to +0.0
    x = dpt.asarray(-0.0, dtype="f4")
    test = dpt.asarray(0.0, dtype="f4")
    assert dpt.isin(x, test)
    assert dpt.isin(test, x)


@pytest.mark.parametrize("dt", _all_dtypes)
def test_isin_py_scalars(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    x = dpt.zeros((10, 10), dtype=dt, sycl_queue=q)
    py_zeros = (
        bool(0),
        int(0),
        float(0),
        complex(0),
        np.float32(0),
        ctypes.c_int(0),
    )
    for sc in py_zeros:
        r1 = dpt.isin(x, sc)
        assert isinstance(r1, dpt.usm_ndarray)
        r2 = dpt.isin(sc, x)
        assert isinstance(r2, dpt.usm_ndarray)


def test_isin_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    x = dpt.ones(10, sycl_queue=q1)
    test = dpt.ones_like(x, sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.isin(x, test)
