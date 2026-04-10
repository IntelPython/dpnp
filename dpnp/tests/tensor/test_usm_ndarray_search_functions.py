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
import itertools

import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_array_equal

import dpnp.tensor as dpt
from dpnp.tensor._search_functions import _where_result_type
from dpnp.tensor._type_utils import _all_data_types

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

_all_dtypes = [
    "?",
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "e",
    "f",
    "d",
    "F",
    "D",
]


class mock_device:
    def __init__(self, fp16, fp64):
        self.has_aspect_fp16 = fp16
        self.has_aspect_fp64 = fp64


def test_where_basic():
    get_queue_or_skip()

    cond = dpt.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
            [True, True, True],
        ]
    )
    out = dpt.where(cond, dpt.asarray(1), dpt.asarray(0))
    out_expected = dpt.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]]
    )
    assert (dpt.asnumpy(out) == dpt.asnumpy(out_expected)).all()

    out = dpt.where(cond, dpt.ones(cond.shape), dpt.zeros(cond.shape))
    assert (dpt.asnumpy(out) == dpt.asnumpy(out_expected)).all()

    out = dpt.where(
        cond,
        dpt.ones(cond.shape[0], dtype="i4")[:, dpt.newaxis],
        dpt.zeros(cond.shape[0], dtype="i4")[:, dpt.newaxis],
    )
    assert (dpt.asnumpy(out) == dpt.asnumpy(out_expected)).all()


def _dtype_all_close(x1, x2):
    if np.issubdtype(x2.dtype, np.floating) or np.issubdtype(
        x2.dtype, np.complexfloating
    ):
        x2_dtype = x2.dtype
        return np.allclose(
            x1, x2, atol=np.finfo(x2_dtype).eps, rtol=np.finfo(x2_dtype).eps
        )
    else:
        return np.allclose(x1, x2)


@pytest.mark.parametrize("dt1", _all_dtypes)
@pytest.mark.parametrize("dt2", _all_dtypes)
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("fp64", [True, False])
def test_where_result_types(dt1, dt2, fp16, fp64):
    dev = mock_device(fp16, fp64)

    dt1 = dpt.dtype(dt1)
    dt2 = dpt.dtype(dt2)
    res_t = _where_result_type(dt1, dt2, dev)

    if fp16 and fp64:
        assert res_t == dpt.result_type(dt1, dt2)
    else:
        if res_t:
            assert res_t.kind == dpt.result_type(dt1, dt2).kind
        else:
            # some illegal cases are covered above, but
            # this guarantees that _where_result_type
            # produces None only when one of the dtypes
            # is illegal given fp aspects of device
            all_dts = _all_data_types(fp16, fp64)
            assert dt1 not in all_dts or dt2 not in all_dts


@pytest.mark.parametrize("dt", _all_dtypes)
def test_where_mask_dtypes(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    # mask dtype changes
    cond = dpt.asarray([0, 1, 3, 0, 10], dtype=dt, sycl_queue=q)
    x1 = dpt.asarray(0, dtype="f4", sycl_queue=q)
    x2 = dpt.asarray(1, dtype="f4", sycl_queue=q)
    res = dpt.where(cond, x1, x2)

    res_check = np.asarray([1, 0, 0, 1, 0], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # contiguous cases
    x1 = dpt.full(cond.shape, 0, dtype="f4", sycl_queue=q)
    x2 = dpt.full(cond.shape, 1, dtype="f4", sycl_queue=q)
    res = dpt.where(cond, x1, x2)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # input array dtype changes
    cond = dpt.asarray([False, True, True, False, True], sycl_queue=q)
    x1 = dpt.asarray(0, dtype=dt, sycl_queue=q)
    x2 = dpt.asarray(1, dtype=dt, sycl_queue=q)
    res = dpt.where(cond, x1, x2)

    res_check = np.asarray([1, 0, 0, 1, 0], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # contiguous cases
    x1 = dpt.full(cond.shape, 0, dtype=dt, sycl_queue=q)
    x2 = dpt.full(cond.shape, 1, dtype=dt, sycl_queue=q)
    res = dpt.where(cond, x1, x2)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)


def test_where_asymmetric_dtypes():
    q = get_queue_or_skip()

    cond = dpt.asarray([0, 1, 3, 0, 10], dtype="?", sycl_queue=q)
    x1 = dpt.asarray(2, dtype="i4", sycl_queue=q)
    x2 = dpt.asarray(3, dtype="i8", sycl_queue=q)

    res = dpt.where(cond, x1, x2)
    res_check = np.asarray([3, 2, 2, 3, 2], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)

    # flip order

    res = dpt.where(cond, x2, x1)
    res_check = np.asarray([2, 3, 3, 2, 3], dtype=res.dtype)
    assert _dtype_all_close(dpt.asnumpy(res), res_check)


def test_where_nan_inf():
    get_queue_or_skip()

    cond = dpt.asarray([True, False, True, False], dtype="?")
    x1 = dpt.asarray([np.nan, 2.0, np.inf, 3.0], dtype="f4")
    x2 = dpt.asarray([2.0, np.nan, 3.0, np.inf], dtype="f4")

    cond_np = dpt.asnumpy(cond)
    x1_np = dpt.asnumpy(x1)
    x2_np = dpt.asnumpy(x2)

    res = dpt.where(cond, x1, x2)
    res_np = np.where(cond_np, x1_np, x2_np)

    assert np.allclose(dpt.asnumpy(res), res_np, equal_nan=True)

    res = dpt.where(x1, cond, x2)
    res_np = np.where(x1_np, cond_np, x2_np)
    assert _dtype_all_close(dpt.asnumpy(res), res_np)


def test_where_empty():
    # check that numpy returns same results when
    # handling empty arrays
    get_queue_or_skip()

    empty = dpt.empty(0, dtype="i2")
    m = dpt.asarray(True)
    x1 = dpt.asarray(1, dtype="i2")
    x2 = dpt.asarray(2, dtype="i2")
    res = dpt.where(empty, x1, x2)

    empty_np = np.empty(0, dtype="i2")
    m_np = dpt.asnumpy(m)
    x1_np = dpt.asnumpy(x1)
    x2_np = dpt.asnumpy(x2)
    res_np = np.where(empty_np, x1_np, x2_np)

    assert_array_equal(dpt.asnumpy(res), res_np)

    res = dpt.where(m, empty, x2)
    res_np = np.where(m_np, empty_np, x2_np)

    assert_array_equal(dpt.asnumpy(res), res_np)

    # check that broadcasting is performed
    with pytest.raises(ValueError):
        dpt.where(empty, x1, dpt.empty((1, 2)))


@pytest.mark.parametrize("order", ["C", "F"])
def test_where_contiguous(order):
    get_queue_or_skip()

    cond = dpt.asarray(
        [
            [[True, False, False], [False, True, True]],
            [[False, True, False], [True, False, True]],
            [[False, False, True], [False, False, True]],
            [[False, False, False], [True, False, True]],
            [[True, True, True], [True, False, True]],
        ],
        order=order,
    )

    x1 = dpt.full(cond.shape, 2, dtype="i4", order=order)
    x2 = dpt.full(cond.shape, 3, dtype="i4", order=order)
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)

    assert _dtype_all_close(dpt.asnumpy(res), expected)


def test_where_contiguous1D():
    get_queue_or_skip()

    cond = dpt.asarray([True, False, True, False, False, True])

    x1 = dpt.full(cond.shape, 2, dtype="i4")
    x2 = dpt.full(cond.shape, 3, dtype="i4")
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)
    assert_array_equal(dpt.asnumpy(res), expected)

    # test with complex dtype (branch in kernel)
    x1 = dpt.astype(x1, dpt.complex64)
    x2 = dpt.astype(x2, dpt.complex64)
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)
    assert _dtype_all_close(dpt.asnumpy(res), expected)


def test_where_gh_1170():
    get_queue_or_skip()

    cond = dpt.asarray([False, True, True, False], dtype="?")
    x1 = dpt.ones((3, 4), dtype="i4")
    x2 = dpt.zeros((3, 4), dtype="i4")

    res = dpt.where(cond, x1, x2)
    expected = np.broadcast_to(dpt.asnumpy(cond).astype(x1.dtype), x1.shape)

    assert_array_equal(dpt.asnumpy(res), expected)


def test_where_strided():
    get_queue_or_skip()

    s0, s1 = 4, 9
    cond = dpt.reshape(
        dpt.asarray(
            [True, False, False, False, True, True, False, True, False] * s0
        ),
        (s0, s1),
    )[:, ::3]

    x1 = dpt.reshape(
        dpt.arange(cond.shape[0] * cond.shape[1] * 2, dtype="i4"),
        (cond.shape[0], cond.shape[1] * 2),
    )[:, ::2]
    x2 = dpt.reshape(
        dpt.arange(cond.shape[0] * cond.shape[1] * 3, dtype="i4"),
        (cond.shape[0], cond.shape[1] * 3),
    )[:, ::3]
    expected = np.where(dpt.asnumpy(cond), dpt.asnumpy(x1), dpt.asnumpy(x2))
    res = dpt.where(cond, x1, x2)

    assert_array_equal(dpt.asnumpy(res), expected)

    # negative strides
    res = dpt.where(cond, dpt.flip(x1), x2)
    expected = np.where(
        dpt.asnumpy(cond), np.flip(dpt.asnumpy(x1)), dpt.asnumpy(x2)
    )
    assert_array_equal(dpt.asnumpy(res), expected)

    res = dpt.where(dpt.flip(cond), x1, x2)
    expected = np.where(
        np.flip(dpt.asnumpy(cond)), dpt.asnumpy(x1), dpt.asnumpy(x2)
    )
    assert_array_equal(dpt.asnumpy(res), expected)


def test_where_invariants():
    get_queue_or_skip()

    test_sh = (
        6,
        8,
    )
    mask = dpt.asarray(np.random.choice([True, False], size=test_sh))
    p = dpt.ones(test_sh, dtype=dpt.int16)
    m = dpt.full(test_sh, -1, dtype=dpt.int16)
    inds_list = [
        (
            np.s_[:3],
            np.s_[::2],
        ),
        (
            np.s_[::2],
            np.s_[::2],
        ),
        (
            np.s_[::-1],
            np.s_[:],
        ),
    ]
    for ind in inds_list:
        r1 = dpt.where(mask, p, m)[ind]
        r2 = dpt.where(mask[ind], p[ind], m[ind])
        assert (dpt.asnumpy(r1) == dpt.asnumpy(r2)).all()


def test_where_arg_validation():
    get_queue_or_skip()

    check = {}
    x1 = dpt.empty((1,), dtype="i4")
    x2 = dpt.empty((1,), dtype="i4")

    with pytest.raises(TypeError):
        dpt.where(check, x1, x2)
    with pytest.raises(ValueError):
        dpt.where(x1, check, x2)
    with pytest.raises(ValueError):
        dpt.where(x1, x2, check)


def test_where_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()
    q3 = get_queue_or_skip()

    x1 = dpt.empty((1,), dtype="i4", sycl_queue=q1)
    x2 = dpt.empty((1,), dtype="i4", sycl_queue=q2)

    with pytest.raises(ExecutionPlacementError):
        dpt.where(dpt.empty((1,), dtype="i4", sycl_queue=q1), x1, x2)
    with pytest.raises(ExecutionPlacementError):
        dpt.where(dpt.empty((1,), dtype="i4", sycl_queue=q3), x1, x2)
    with pytest.raises(ExecutionPlacementError):
        dpt.where(x1, x1, x2)


def test_where_order():
    get_queue_or_skip()

    test_sh = (
        20,
        20,
    )
    test_sh2 = tuple(2 * dim for dim in test_sh)
    n = test_sh[-1]

    for dt1, dt2 in zip(["i4", "i4", "f4"], ["i4", "f4", "i4"]):
        ar1 = dpt.zeros(test_sh, dtype=dt1, order="C")
        ar2 = dpt.ones(test_sh, dtype=dt2, order="C")
        condition = dpt.zeros(test_sh, dtype="?", order="C")
        res1 = dpt.where(condition, ar1, ar2, order="C")
        assert res1.flags.c_contiguous
        res2 = dpt.where(condition, ar1, ar2, order="F")
        assert res2.flags.f_contiguous
        res3 = dpt.where(condition, ar1, ar2, order="A")
        assert res3.flags.c_contiguous
        res4 = dpt.where(condition, ar1, ar2, order="K")
        assert res4.flags.c_contiguous

        ar1 = dpt.ones(test_sh, dtype=dt1, order="F")
        ar2 = dpt.ones(test_sh, dtype=dt2, order="F")
        condition = dpt.zeros(test_sh, dtype="?", order="F")
        res1 = dpt.where(condition, ar1, ar2, order="C")
        assert res1.flags.c_contiguous
        res2 = dpt.where(condition, ar1, ar2, order="F")
        assert res2.flags.f_contiguous
        res3 = dpt.where(condition, ar1, ar2, order="A")
        assert res2.flags.f_contiguous
        res4 = dpt.where(condition, ar1, ar2, order="K")
        assert res4.flags.f_contiguous

        ar1 = dpt.ones(test_sh2, dtype=dt1, order="C")[:20, ::-2]
        ar2 = dpt.ones(test_sh2, dtype=dt2, order="C")[:20, ::-2]
        condition = dpt.zeros(test_sh2, dtype="?", order="C")[:20, ::-2]
        res1 = dpt.where(condition, ar1, ar2, order="K")
        assert res1.strides == (n, -1)
        res2 = dpt.where(condition, ar1, ar2, order="C")
        assert res2.strides == (n, 1)

        ar1 = dpt.ones(test_sh2, dtype=dt1, order="C")[:20, ::-2].mT
        ar2 = dpt.ones(test_sh2, dtype=dt2, order="C")[:20, ::-2].mT
        condition = dpt.zeros(test_sh2, dtype="?", order="C")[:20, ::-2].mT
        res1 = dpt.where(condition, ar1, ar2, order="K")
        assert res1.strides == (-1, n)
        res2 = dpt.where(condition, ar1, ar2, order="C")
        assert res2.strides == (n, 1)

        ar1 = dpt.ones(n, dtype=dt1, order="C")
        ar2 = dpt.broadcast_to(dpt.ones(n, dtype=dt2, order="C"), test_sh)
        condition = dpt.zeros(n, dtype="?", order="C")
        res = dpt.where(condition, ar1, ar2, order="K")
        assert res.strides == (20, 1)


def test_where_unaligned():
    get_queue_or_skip()

    x = dpt.ones(513, dtype="i4")
    a = dpt.full(512, 2, dtype="i4")
    b = dpt.zeros(512, dtype="i4")

    expected = dpt.full(512, 2, dtype="i4")
    assert dpt.all(dpt.where(x[1:], a, b) == expected)


def test_where_out():
    get_queue_or_skip()

    n1, n2, n3 = 3, 4, 5
    ar1 = dpt.reshape(dpt.arange(n1 * n2 * n3, dtype="i4"), (n1, n2, n3))
    ar2 = dpt.full_like(ar1, -5)
    condition = dpt.tile(
        dpt.reshape(
            dpt.asarray([True, False, False, True], dtype="?"), (1, n2, 1)
        ),
        (n1, 1, n3),
    )

    out = dpt.zeros((2 * n1, 3 * n2, n3), dtype="i4")
    res = dpt.where(condition, ar1, ar2, out=out[::-2, 1::3, :])

    assert dpt.all(res == out[::-2, 1::3, :])
    assert dpt.all(out[::-2, 0::3, :] == 0)
    assert dpt.all(out[::-2, 2::3, :] == 0)

    assert dpt.all(res[:, 1:3, :] == -5)
    assert dpt.all(res[:, 0, :] == ar1[:, 0, :])
    assert dpt.all(res[:, 3, :] == ar1[:, 3, :])

    condition = dpt.tile(
        dpt.reshape(dpt.asarray([1, 0], dtype="i4"), (1, 2, 1)),
        (n1, 2, n3),
    )
    res = dpt.where(
        condition[:, ::-1, :], condition[:, ::-1, :], condition, out=condition
    )
    assert dpt.all(res == condition)
    assert dpt.all(condition == 1)

    condition = dpt.tile(
        dpt.reshape(dpt.asarray([True, False], dtype="?"), (1, 2, 1)),
        (n1, 2, n3),
    )
    ar1 = dpt.full((n1, n2, n3), 7, dtype="i4")
    ar2 = dpt.full_like(ar1, -5)
    res = dpt.where(condition, ar1, ar2, out=ar2[:, ::-1, :])
    assert dpt.all(ar2[:, ::-1, :] == res)
    assert dpt.all(ar2[:, ::2, :] == -5)
    assert dpt.all(ar2[:, 1::2, :] == 7)

    condition = dpt.tile(
        dpt.reshape(dpt.asarray([True, False], dtype="?"), (1, 2, 1)),
        (n1, 2, n3),
    )
    ar1 = dpt.full((n1, n2, n3), 7, dtype="i4")
    ar2 = dpt.full_like(ar1, -5)
    res = dpt.where(condition, ar1, ar2, out=ar1[:, ::-1, :])
    assert dpt.all(ar1[:, ::-1, :] == res)
    assert dpt.all(ar1[:, ::2, :] == -5)
    assert dpt.all(ar1[:, 1::2, :] == 7)


def test_where_out_arg_validation():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    condition = dpt.ones(5, dtype="i4", sycl_queue=q1)
    x1 = dpt.ones(5, dtype="i4", sycl_queue=q1)
    x2 = dpt.ones(5, dtype="i4", sycl_queue=q1)

    out_wrong_queue = dpt.empty_like(condition, sycl_queue=q2)
    out_wrong_dtype = dpt.empty_like(condition, dtype="f4")
    out_wrong_shape = dpt.empty(6, dtype="i4", sycl_queue=q1)
    out_not_writable = dpt.empty_like(condition)
    out_not_writable.flags["W"] = False

    with pytest.raises(TypeError):
        dpt.where(condition, x1, x2, out=dict())
    with pytest.raises(ExecutionPlacementError):
        dpt.where(condition, x1, x2, out=out_wrong_queue)
    with pytest.raises(ValueError):
        dpt.where(condition, x1, x2, out=out_wrong_dtype)
    with pytest.raises(ValueError):
        dpt.where(condition, x1, x2, out=out_wrong_shape)
    with pytest.raises(ValueError):
        dpt.where(condition, x1, x2, out=out_not_writable)


@pytest.mark.parametrize("arr_dt", _all_dtypes)
def test_where_python_scalar(arr_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arr_dt, q)

    n1, n2 = 10, 10
    condition = dpt.tile(
        dpt.reshape(
            dpt.asarray([True, False], dtype="?", sycl_queue=q), (1, 2)
        ),
        (n1, n2 // 2),
    )
    x = dpt.zeros((n1, n2), dtype=arr_dt, sycl_queue=q)
    py_scalars = (
        bool(0),
        int(0),
        float(0),
        complex(0),
        np.float32(0),
        ctypes.c_int(0),
    )
    for sc in py_scalars:
        r = dpt.where(condition, x, sc)
        assert isinstance(r, dpt.usm_ndarray)
        r = dpt.where(condition, sc, x)
        assert isinstance(r, dpt.usm_ndarray)


def test_where_two_python_scalars():
    get_queue_or_skip()

    n1, n2 = 10, 10
    condition = dpt.tile(
        dpt.reshape(dpt.asarray([True, False], dtype="?"), (1, 2)),
        (n1, n2 // 2),
    )

    py_scalars = [
        bool(0),
        int(0),
        float(0),
        complex(0),
        np.float32(0),
        ctypes.c_int(0),
    ]

    for sc1, sc2 in itertools.product(py_scalars, repeat=2):
        r = dpt.where(condition, sc1, sc2)
        assert isinstance(r, dpt.usm_ndarray)
