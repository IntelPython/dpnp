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
from dpctl.utils import SequentialOrderManager

import dpnp.tensor as dpt
from dpnp.tensor._tensor_elementwise_impl import _divide_by_scalar
from dpnp.tensor._type_utils import _can_cast

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _compare_dtypes,
)


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_divide_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.divide(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.divide(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.divide(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.divide(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    )
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_divide_python_scalar(arr_dt):
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
        R = dpt.divide(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.divide(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_divide_inplace_dtype_matrix(op1_dtype, op2_dtype):
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
    if (
        _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64, casting="same_kind")
        and dpt.dtype(op1_dtype).kind in "fc"
    ):
        ar1 /= ar2
        assert dpt.all(ar1 == 1)

        ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)[::2]
        ar3 /= ar4
        assert dpt.all(ar3 == 1)
    else:
        with pytest.raises(ValueError):
            ar1 /= ar2
            dpt.divide(ar1, ar2, out=ar1)

    # out is second arg
    ar1 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype, sycl_queue=q)
    if (
        _can_cast(ar1.dtype, ar2.dtype, _fp16, _fp64)
        and dpt.dtype(op2_dtype).kind in "fc"
    ):
        dpt.divide(ar1, ar2, out=ar2)
        assert dpt.all(ar2 == 1)

        ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)[::2]
        dpt.divide(ar3, ar4, out=ar4)
        dpt.all(ar4 == 1)
    else:
        with pytest.raises(ValueError):
            dpt.divide(ar1, ar2, out=ar2)


def test_divide_gh_1711():
    "See https://github.com/IntelPython/dpctl/issues/1711"
    get_queue_or_skip()

    res = dpt.divide(-4, dpt.asarray(1, dtype="u4"))
    assert isinstance(res, dpt.usm_ndarray)
    assert res.dtype.kind == "f"
    assert dpt.allclose(res, -4 / dpt.asarray(1, dtype="i4"))

    res = dpt.divide(dpt.asarray(3, dtype="u4"), -2)
    assert isinstance(res, dpt.usm_ndarray)
    assert res.dtype.kind == "f"
    assert dpt.allclose(res, dpt.asarray(3, dtype="i4") / -2)


# don't test for overflowing double as Python won't cast
# a Python integer of that size to a Python float
@pytest.mark.parametrize("fp_dt", [dpt.float16, dpt.float32])
def test_divide_by_scalar_overflow(fp_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(fp_dt, q)

    x = dpt.ones(10, dtype=fp_dt, sycl_queue=q)
    out = dpt.empty_like(x)

    max_exp = np.finfo(fp_dt).maxexp
    sca = 2**max_exp

    _manager = SequentialOrderManager[q]
    dep_evs = _manager.submitted_events
    _, ev = _divide_by_scalar(
        src=x, scalar=sca, dst=out, sycl_queue=q, depends=dep_evs
    )
    ev.wait()

    assert dpt.all(out == 0)
