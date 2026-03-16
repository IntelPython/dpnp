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
import re

import dpctl
import numpy as np
import pytest
from numpy.testing import assert_allclose

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _compare_dtypes,
    _no_complex_dtypes,
    _usm_types,
)


@pytest.mark.parametrize("op1_dtype", _no_complex_dtypes)
@pytest.mark.parametrize("op2_dtype", _no_complex_dtypes)
def test_logaddexp_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.logaddexp(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.logaddexp(dpt.asnumpy(ar1), dpt.asnumpy(ar2))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    tol = 8 * max(
        np.finfo(r.dtype).resolution, np.finfo(expected.dtype).resolution
    )
    assert_allclose(
        dpt.asnumpy(r), expected.astype(r.dtype), atol=tol, rtol=tol
    )
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.logaddexp(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected = np.logaddexp(dpt.asnumpy(ar3)[::-1], dpt.asnumpy(ar4)[::2])
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert_allclose(
        dpt.asnumpy(r), expected.astype(r.dtype), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_logaddexp_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.logaddexp(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_logaddexp_order():
    get_queue_or_skip()

    test_shape = (
        20,
        20,
    )
    test_shape2 = tuple(2 * dim for dim in test_shape)
    n = test_shape[-1]

    for dt1, dt2 in zip(["i4", "i4", "f4"], ["i4", "f4", "i4"]):
        ar1 = dpt.ones(test_shape, dtype=dt1, order="C")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="C")
        r1 = dpt.logaddexp(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.logaddexp(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.logaddexp(ar1, ar2, order="A")
        assert r3.flags.c_contiguous
        r4 = dpt.logaddexp(ar1, ar2, order="K")
        assert r4.flags.c_contiguous

        ar1 = dpt.ones(test_shape, dtype=dt1, order="F")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="F")
        r1 = dpt.logaddexp(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.logaddexp(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.logaddexp(ar1, ar2, order="A")
        assert r3.flags.f_contiguous
        r4 = dpt.logaddexp(ar1, ar2, order="K")
        assert r4.flags.f_contiguous

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2]
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2]
        r4 = dpt.logaddexp(ar1, ar2, order="K")
        assert r4.strides == (n, -1)
        r5 = dpt.logaddexp(ar1, ar2, order="C")
        assert r5.strides == (n, 1)

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2].mT
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2].mT
        r4 = dpt.logaddexp(ar1, ar2, order="K")
        assert r4.strides == (-1, n)
        r5 = dpt.logaddexp(ar1, ar2, order="C")
        assert r5.strides == (n, 1)


def test_logaddexp_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(1, 6, dtype="i4")

    r = dpt.logaddexp(m, v)

    expected = np.logaddexp(
        np.ones((100, 5), dtype="i4"), np.arange(1, 6, dtype="i4")
    )
    assert (dpt.asnumpy(r) == expected.astype(r.dtype)).all()

    r2 = dpt.logaddexp(v, m)
    expected2 = np.logaddexp(
        np.arange(1, 6, dtype="i4"), np.ones((100, 5), dtype="i4")
    )
    assert (dpt.asnumpy(r2) == expected2.astype(r2.dtype)).all()


def test_logaddexp_broadcasting_error():
    get_queue_or_skip()
    m = dpt.ones((10, 10), dtype="i4")
    v = dpt.ones((3,), dtype="i4")
    with pytest.raises(ValueError):
        dpt.logaddexp(m, v)


@pytest.mark.parametrize("arr_dt", _no_complex_dtypes)
def test_logaddexp_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.zeros((10, 10), dtype=arr_dt, sycl_queue=q)
    py_zeros = (
        bool(0),
        int(0),
        float(0),
        np.float32(0),
        ctypes.c_int(0),
    )
    for sc in py_zeros:
        R = dpt.logaddexp(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.logaddexp(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_logaddexp_dtype_error(
    dtype,
):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    ar1 = dpt.ones(5, dtype=dtype)
    ar2 = dpt.ones_like(ar1, dtype="f4")

    y = dpt.zeros_like(ar1, dtype="int8")
    with pytest.raises(ValueError) as excinfo:
        dpt.logaddexp(ar1, ar2, out=y)
    assert re.match("Output array of type.*is needed", str(excinfo.value))
