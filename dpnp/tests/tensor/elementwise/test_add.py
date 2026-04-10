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
from dpctl.utils import ExecutionPlacementError

import dpnp.tensor as dpt
from dpnp.tensor._type_utils import _can_cast

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
def test_add_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.add(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.add(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, 2, dtype=r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    r2 = dpt.empty_like(ar1, dtype=r.dtype)
    dpt.add(ar1, ar2, out=r2)
    assert (dpt.asnumpy(r2) == np.full(r2.shape, 2, dtype=r2.dtype)).all()

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.add(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.add(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, 2, dtype=r.dtype)).all()

    r2 = dpt.empty_like(ar1, dtype=r.dtype)
    dpt.add(ar3[::-1], ar4[::2], out=r2)
    assert (dpt.asnumpy(r2) == np.full(r2.shape, 2, dtype=r2.dtype)).all()


@pytest.mark.parametrize("op1_usm_type", _usm_types)
@pytest.mark.parametrize("op2_usm_type", _usm_types)
def test_add_usm_type_matrix(op1_usm_type, op2_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.ones(sz, dtype="i4", usm_type=op1_usm_type)
    ar2 = dpt.ones_like(ar1, dtype="i4", usm_type=op2_usm_type)

    r = dpt.add(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_usm_type = dpctl.utils.get_coerced_usm_type(
        (op1_usm_type, op2_usm_type)
    )
    assert r.usm_type == expected_usm_type


def test_add_order():
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
        r1 = dpt.add(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.add(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.add(ar1, ar2, order="A")
        assert r3.flags.c_contiguous
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.flags.c_contiguous

        ar1 = dpt.ones(test_shape, dtype=dt1, order="F")
        ar2 = dpt.ones(test_shape, dtype=dt2, order="F")
        r1 = dpt.add(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.add(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.add(ar1, ar2, order="A")
        assert r3.flags.f_contiguous
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.flags.f_contiguous

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2]
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2]
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.strides == (n, -1)
        r5 = dpt.add(ar1, ar2, order="C")
        assert r5.strides == (n, 1)

        ar1 = dpt.ones(test_shape2, dtype=dt1, order="C")[:20, ::-2].mT
        ar2 = dpt.ones(test_shape2, dtype=dt2, order="C")[:20, ::-2].mT
        r4 = dpt.add(ar1, ar2, order="K")
        assert r4.strides == (-1, n)
        r5 = dpt.add(ar1, ar2, order="C")
        assert r5.strides == (n, 1)


def test_add_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    r = dpt.add(m, v)
    assert (dpt.asnumpy(r) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()

    r2 = dpt.add(v, m)
    assert (dpt.asnumpy(r2) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()

    r3 = dpt.empty_like(m)
    dpt.add(m, v, out=r3)
    assert (dpt.asnumpy(r3) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()

    r4 = dpt.empty_like(m)
    dpt.add(v, m, out=r4)
    assert (dpt.asnumpy(r4) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()


def test_add_broadcasting_new_shape():
    get_queue_or_skip()

    ar1 = dpt.ones((6, 1), dtype="i4")
    ar2 = dpt.arange(6, dtype="i4")

    r = dpt.add(ar1, ar2)
    assert (dpt.asnumpy(r) == np.arange(1, 7, dtype="i4")[np.newaxis, :]).all()

    r1 = dpt.add(ar2, ar1)
    assert (dpt.asnumpy(r1) == np.arange(1, 7, dtype="i4")[np.newaxis, :]).all()

    r2 = dpt.add(ar1[::2], ar2[::2])
    assert (
        dpt.asnumpy(r2) == np.arange(1, 7, dtype="i4")[::2][np.newaxis, :]
    ).all()

    r3 = dpt.empty_like(ar1)
    with pytest.raises(ValueError):
        dpt.add(ar1, ar2, out=r3)

    ar3 = dpt.ones((6, 1), dtype="i4")
    ar4 = dpt.ones((1, 6), dtype="i4")

    r4 = dpt.add(ar3, ar4)
    assert (dpt.asnumpy(r4) == np.full((6, 6), 2, dtype="i4")).all()

    r5 = dpt.add(ar4, ar3)
    assert (dpt.asnumpy(r5) == np.full((6, 6), 2, dtype="i4")).all()

    r6 = dpt.add(ar3[::2], ar4[:, ::2])
    assert (dpt.asnumpy(r6) == np.full((3, 3), 2, dtype="i4")).all()

    r7 = dpt.add(ar3[::2], ar4)
    assert (dpt.asnumpy(r7) == np.full((3, 6), 2, dtype="i4")).all()


def test_add_broadcasting_error():
    get_queue_or_skip()
    m = dpt.ones((10, 10), dtype="i4")
    v = dpt.ones((3,), dtype="i4")
    with pytest.raises(ValueError):
        dpt.add(m, v)


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_add_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    X = dpt.zeros((10, 10), dtype=arr_dt, sycl_queue=q)
    py_zeros = (
        bool(0),
        int(0),
        float(0),
        complex(0),
        np.float32(0),
        ctypes.c_int(0),
    )
    for sc in py_zeros:
        R = dpt.add(X, sc)
        assert isinstance(R, dpt.usm_ndarray)
        R = dpt.add(sc, X)
        assert isinstance(R, dpt.usm_ndarray)


class MockArray:
    def __init__(self, arr):
        self.data_ = arr

    @property
    def __sycl_usm_array_interface__(self):
        return self.data_.__sycl_usm_array_interface__


def test_add_mock_array():
    get_queue_or_skip()
    a = dpt.arange(10)
    b = dpt.ones(10)
    c = MockArray(b)
    r = dpt.add(a, c)
    assert isinstance(r, dpt.usm_ndarray)


def test_add_canary_mock_array():
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
        dpt.add(a, c)


def test_add_types_property():
    get_queue_or_skip()
    types = dpt.add.types
    assert isinstance(types, list)
    assert len(types) > 0
    assert types == dpt.add.types_


def test_add_errors():
    get_queue_or_skip()
    try:
        gpu_queue = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('gpu') failed, skipping")
    try:
        cpu_queue = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('cpu') failed, skipping")

    ar1 = dpt.ones(2, dtype="float32", sycl_queue=gpu_queue)
    ar2 = dpt.ones_like(ar1, sycl_queue=gpu_queue)
    y = dpt.empty_like(ar1, sycl_queue=cpu_queue)
    with pytest.raises(ExecutionPlacementError) as excinfo:
        dpt.add(ar1, ar2, out=y)
    assert "Input and output allocation queues are not compatible" in str(
        excinfo.value
    )

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones_like(ar1, dtype="int32")
    y = dpt.empty(3)
    with pytest.raises(ValueError) as excinfo:
        dpt.add(ar1, ar2, out=y)
    assert "The shape of input and output arrays are inconsistent" in str(
        excinfo.value
    )

    ar1 = np.ones(2, dtype="float32")
    ar2 = np.ones_like(ar1, dtype="int32")
    with pytest.raises(ExecutionPlacementError) as excinfo:
        dpt.add(ar1, ar2)
    assert re.match(
        "Execution placement can not be unambiguously inferred.*",
        str(excinfo.value),
    )

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones_like(ar1, dtype="int32")
    y = np.empty(ar1.shape, dtype=ar1.dtype)
    with pytest.raises(TypeError) as excinfo:
        dpt.add(ar1, ar2, out=y)
    assert "output array must be of usm_ndarray type" in str(excinfo.value)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_add_dtype_error(
    dtype,
):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    ar1 = dpt.ones(5, dtype=dtype)
    ar2 = dpt.ones_like(ar1, dtype="f4")

    y = dpt.zeros_like(ar1, dtype="int8")
    with pytest.raises(ValueError) as excinfo:
        dpt.add(ar1, ar2, out=y)
    assert re.match("Output array of type.*is needed", str(excinfo.value))


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_add_inplace_python_scalar(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.zeros((10, 10), dtype=dtype, sycl_queue=q)
    dt_kind = X.dtype.kind
    if dt_kind in "ui":
        X += int(0)
    elif dt_kind == "f":
        X += float(0)
    elif dt_kind == "c":
        X += complex(0)
    elif dt_kind == "b":
        X += bool(0)


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_add_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    # operators use a different Python implementation which permits
    # same kind style casting
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64, casting="same_kind"):
        ar1 += ar2
        assert (
            dpt.asnumpy(ar1) == np.full(ar1.shape, 2, dtype=ar1.dtype)
        ).all()

        ar3 = dpt.ones(sz, dtype=op1_dtype)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)[::2]
        ar3 += ar4
        assert (
            dpt.asnumpy(ar3) == np.full(ar3.shape, 2, dtype=ar3.dtype)
        ).all()
    else:
        with pytest.raises(ValueError):
            ar1 += ar2

    # here, test the special case where out is the first argument
    # so an in-place kernel is used for efficiency
    # this covers a specific branch in the BinaryElementwiseFunc logic
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64):
        dpt.add(ar1, ar2, out=ar1)
        assert (
            dpt.asnumpy(ar1) == np.full(ar1.shape, 2, dtype=ar1.dtype)
        ).all()

        ar3 = dpt.ones(sz, dtype=op1_dtype)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)[::2]
        dpt.add(ar3, ar4, out=ar3)
        assert (
            dpt.asnumpy(ar3) == np.full(ar3.shape, 2, dtype=ar3.dtype)
        ).all()
    else:
        with pytest.raises(ValueError):
            dpt.add(ar1, ar2, out=ar1)

    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)
    if _can_cast(ar1.dtype, ar2.dtype, _fp16, _fp64):
        dpt.add(ar1, ar2, out=ar2)
        assert (
            dpt.asnumpy(ar2) == np.full(ar2.shape, 2, dtype=ar2.dtype)
        ).all()

        ar3 = dpt.ones(sz, dtype=op1_dtype)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype)[::2]
        dpt.add(ar3, ar4, out=ar4)
        assert (
            dpt.asnumpy(ar4) == np.full(ar4.shape, 2, dtype=ar4.dtype)
        ).all()
    else:
        with pytest.raises(ValueError):
            dpt.add(ar1, ar2, out=ar2)


def test_add_inplace_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    dpt.add(m, v, out=m)
    assert (dpt.asnumpy(m) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()

    # check case where second arg is out
    dpt.add(v, m, out=m)
    assert (
        dpt.asnumpy(m) == np.arange(10, dtype="i4")[np.newaxis, 1:10:2]
    ).all()


def test_add_inplace_operator_broadcasting():
    get_queue_or_skip()

    m = dpt.ones((100, 5), dtype="i4")
    v = dpt.arange(5, dtype="i4")

    m += v
    assert (dpt.asnumpy(m) == np.arange(1, 6, dtype="i4")[np.newaxis, :]).all()


def test_add_inplace_operator_mutual_broadcast():
    get_queue_or_skip()

    x1 = dpt.ones((1, 10), dtype="i4")
    x2 = dpt.ones((10, 1), dtype="i4")

    with pytest.raises(ValueError):
        dpt.add._inplace_op(x1, x2)


def test_add_inplace_errors():
    get_queue_or_skip()
    try:
        gpu_queue = dpctl.SyclQueue("gpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('gpu') failed, skipping")
    try:
        cpu_queue = dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("SyclQueue('cpu') failed, skipping")

    ar1 = dpt.ones(2, dtype="float32", sycl_queue=gpu_queue)
    ar2 = dpt.ones_like(ar1, sycl_queue=cpu_queue)
    with pytest.raises(ExecutionPlacementError):
        dpt.add(ar1, ar2, out=ar1)

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = dpt.ones(3, dtype="float32")
    with pytest.raises(ValueError):
        dpt.add(ar1, ar2, out=ar1)

    ar1 = np.ones(2, dtype="float32")
    ar2 = dpt.ones(2, dtype="float32")
    with pytest.raises(TypeError):
        dpt.add(ar1, ar2, out=ar1)

    ar1 = dpt.ones(2, dtype="float32")
    ar2 = {}
    with pytest.raises(ValueError):
        dpt.add(ar1, ar2, out=ar1)

    ar1 = dpt.ones((2, 1), dtype="float32")
    ar2 = dpt.ones((1, 2), dtype="float32")
    with pytest.raises(ValueError):
        dpt.add(ar1, ar2, out=ar1)


def test_add_inplace_operator_errors():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    x = dpt.ones(10, dtype="i4", sycl_queue=q1)
    with pytest.raises(TypeError):
        dpt.add._inplace_op(dict(), x)

    x.flags["W"] = False
    with pytest.raises(ValueError):
        dpt.add._inplace_op(x, 2)

    x_q1 = dpt.ones(10, dtype="i4", sycl_queue=q1)
    x_q2 = dpt.ones(10, dtype="i4", sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.add._inplace_op(x_q1, x_q2)


def test_add_inplace_same_tensors():
    get_queue_or_skip()

    ar1 = dpt.ones(10, dtype="i4")
    ar1 += ar1
    assert (dpt.asnumpy(ar1) == np.full(ar1.shape, 2, dtype="i4")).all()

    ar1 = dpt.ones(10, dtype="i4")
    ar2 = dpt.ones(10, dtype="i4")
    dpt.add(ar1, ar2, out=ar1)
    # all ar1 vals should be 2
    assert (dpt.asnumpy(ar1) == np.full(ar1.shape, 2, dtype="i4")).all()

    dpt.add(ar2, ar1, out=ar2)
    # all ar2 vals should be 3
    assert (dpt.asnumpy(ar2) == np.full(ar2.shape, 3, dtype="i4")).all()

    dpt.add(ar1, ar2, out=ar2)
    # all ar2 vals should be 5
    assert (dpt.asnumpy(ar2) == np.full(ar2.shape, 5, dtype="i4")).all()


def test_add_str_repr():
    add_s = str(dpt.add)
    assert isinstance(add_s, str)
    assert "add" in add_s

    add_r = repr(dpt.add)
    assert isinstance(add_r, str)
    assert "add" in add_r


def test_add_cfd():
    q1 = get_queue_or_skip()
    q2 = dpctl.SyclQueue(q1.sycl_device)

    x1 = dpt.ones(10, sycl_queue=q1)
    x2 = dpt.ones(10, sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.add(x1, x2)

    with pytest.raises(ExecutionPlacementError):
        dpt.add(x1, x1, out=x2)


def test_add_out_type_check():
    get_queue_or_skip()

    x1 = dpt.ones(10)
    x2 = dpt.ones(10)

    out = range(10)

    with pytest.raises(TypeError):
        dpt.add(x1, x2, out=out)


def test_add_out_need_temporary():
    get_queue_or_skip()

    x = dpt.ones(10, dtype="u4")

    dpt.add(x[:6], 1, out=x[-6:])

    assert dpt.all(x[:-6] == 1) and dpt.all(x[-6:] == 2)
