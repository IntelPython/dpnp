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

import dpctl
import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_array_equal

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import dpctl_ext.tensor._tensor_impl as ti
from dpctl_ext.tensor._copy_utils import _take_multi_index

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

_all_dtypes = [
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

_all_int_dtypes = ["u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8"]


def test_basic_slice1():
    q = get_queue_or_skip()
    x = dpt.empty(10, dtype="u2", sycl_queue=q)
    y = x[0]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == 0
    assert y.shape == ()
    assert y.strides == ()


def test_basic_slice2():
    q = get_queue_or_skip()
    x = dpt.empty(10, dtype="i2", sycl_queue=q)
    y = x[(0,)]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == 0
    assert y.shape == ()
    assert y.strides == ()


def test_basic_slice3():
    q = get_queue_or_skip()
    x = dpt.empty(10, dtype="i2", sycl_queue=q)
    y = x[:]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == x.ndim
    assert y.shape == x.shape
    assert y.strides == x.strides
    y = x[(slice(None, None, None),)]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.ndim == x.ndim
    assert y.shape == x.shape
    assert y.strides == x.strides


def test_basic_slice4():
    q = get_queue_or_skip()
    n0, n1 = 5, 3
    x = dpt.empty((n0, n1), dtype="f4", sycl_queue=q)
    y = x[::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == x.shape
    assert y.strides == (-x.strides[0], x.strides[1])
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    assert actual_offset == (n0 - 1) * n1


def test_basic_slice5():
    q = get_queue_or_skip()
    n0, n1 = 5, 3
    x = dpt.empty((n0, n1), dtype="c8", sycl_queue=q)
    y = x[:, ::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == x.shape
    assert y.strides == (x.strides[0], -x.strides[1])
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    assert actual_offset == (n1 - 1)


def test_basic_slice6():
    q = get_queue_or_skip()
    i0, n0, n1 = 2, 4, 3
    x = dpt.empty((n0, n1), dtype="c8", sycl_queue=q)
    y = x[i0, ::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (x.shape[1],)
    assert y.strides == (-x.strides[1],)
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    expected_offset = i0 * x.strides[0] + (n1 - 1) * x.strides[1]
    assert actual_offset == expected_offset


def test_basic_slice7():
    q = get_queue_or_skip()
    n0, n1, n2 = 5, 3, 2
    x = dpt.empty((n0, n1, n2), dtype="?", sycl_queue=q)
    y = x[..., ::-1]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == x.shape
    assert y.strides == (
        x.strides[0],
        x.strides[1],
        -x.strides[2],
    )
    actual_offset = y.__sycl_usm_array_interface__["offset"]
    expected_offset = (n2 - 1) * x.strides[2]
    assert actual_offset == expected_offset


def test_basic_slice8():
    q = get_queue_or_skip()
    n0, n1 = 3, 7
    x = dpt.empty((n0, n1), dtype="u1", sycl_queue=q)
    y = x[..., dpt.newaxis]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (n0, n1, 1)
    assert y.strides == (n1, 1, 0)


def test_basic_slice9():
    q = get_queue_or_skip()
    n0, n1 = 3, 7
    x = dpt.empty((n0, n1), dtype="u8", sycl_queue=q)
    y = x[dpt.newaxis, ...]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (1, n0, n1)
    assert y.strides == (0, n1, 1)


def test_basic_slice10():
    q = get_queue_or_skip()
    n0, n1, n2 = 3, 7, 5
    x = dpt.empty((n0, n1, n2), dtype="u1", sycl_queue=q)
    y = x[dpt.newaxis, ..., :]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (1, n0, n1, n2)
    assert y.strides == (0, n1 * n2, n2, 1)


def _all_equal(it1, it2):
    return all(bool(x == y) for x, y in zip(it1, it2))


def test_advanced_slice1():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert y.strides == (1,)
    assert _all_equal(
        (x[ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )
    y = x[(ii,)]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert y.strides == (1,)
    assert _all_equal(
        (x[ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )


def test_advanced_slice1_negative_strides():
    q = get_queue_or_skip()
    ii = dpt.asarray([0, 1], sycl_queue=q)
    x = dpt.flip(dpt.arange(5, dtype="i4", sycl_queue=q))
    y = x[ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert y.strides == (1,)
    assert _all_equal(
        (x[ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )


def test_advanced_slice2():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[ii, dpt.newaxis]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape + (1,)
    assert y.flags["C"]


def test_advanced_slice3():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[dpt.newaxis, ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (1,) + ii.shape
    assert y.flags["C"]


def _make_3d(dt, q):
    return dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype=dt, sycl_queue=q),
        (
            3,
            3,
            3,
        ),
    )


def test_advanced_slice4():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = _make_3d("i4", q)
    y = x[ii, ii, ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert _all_equal(
        (x[ii[k], ii[k], ii[k]] for k in range(ii.shape[0])),
        (y[k] for k in range(ii.shape[0])),
    )


def test_advanced_slice5():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = _make_3d("i4", q)
    y = x[ii, 0, ii]
    assert isinstance(y, dpt.usm_ndarray)
    # 0 broadcast to [0, 0] per array API
    assert y.shape == ii.shape
    assert _all_equal(
        (x[ii[i], 0, ii[i]] for i in range(ii.shape[0])),
        (y[i] for i in range(ii.shape[0])),
    )


def test_advanced_slice6():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = _make_3d("i4", q)
    y = x[:, ii, ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (
        x.shape[0],
        ii.shape[0],
    )
    assert _all_equal(
        (
            x[i, ii[k], ii[k]]
            for i in range(x.shape[0])
            for k in range(ii.shape[0])
        ),
        (y[i, k] for i in range(x.shape[0]) for k in range(ii.shape[0])),
    )


def test_advanced_slice7():
    q = get_queue_or_skip()
    mask = dpt.asarray(
        [
            [[True, True, False], [False, True, True], [True, False, True]],
            [[True, False, False], [False, False, True], [False, True, False]],
            [[True, True, True], [False, False, False], [False, False, True]],
        ],
        sycl_queue=q,
    )
    x = _make_3d("i2", q)
    y = x[mask]
    expected = [0, 1, 4, 5, 6, 8, 9, 14, 16, 18, 19, 20, 26]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (len(expected),)
    assert all(dpt.asnumpy(y[k]) == expected[k] for k in range(len(expected)))


def test_advanced_slice8():
    q = get_queue_or_skip()
    mask = dpt.asarray(
        [[True, False, False], [False, True, False], [False, True, False]],
        sycl_queue=q,
    )
    x = _make_3d("u2", q)
    y = x[mask]
    expected = dpt.asarray(
        [[0, 1, 2], [12, 13, 14], [21, 22, 23]], sycl_queue=q
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(expected)).all()


def test_advanced_slice9():
    q = get_queue_or_skip()
    mask = dpt.asarray(
        [[True, False, False], [False, True, False], [False, True, False]],
        sycl_queue=q,
    )
    x = _make_3d("u4", q)
    y = x[:, mask]
    expected = dpt.asarray([[0, 4, 7], [9, 13, 16], [18, 22, 25]], sycl_queue=q)
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(expected)).all()


def lin_id(i, j, k):
    """global_linear_id for (3,3,3) range traversed in C-contiguous order"""
    return 9 * i + 3 * j + k


def test_advanced_slice10():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i0 = dpt.asarray([0, 1, 1], device=x.device)
    i1 = dpt.asarray([1, 1, 2], device=x.device)
    i2 = dpt.asarray([2, 0, 1], device=x.device)
    y = x[i0, i1, i2]
    res_expected = dpt.asarray(
        [
            lin_id(0, 1, 2),
            lin_id(1, 1, 0),
            lin_id(1, 2, 1),
        ],
        sycl_queue=q,
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == res_expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(res_expected)).all()


def test_advanced_slice11():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i0 = dpt.asarray([0, 1, 1], device=x.device)
    i2 = dpt.asarray([2, 0, 1], device=x.device)
    with pytest.raises(IndexError):
        x[i0, :, i2]


def test_advanced_slice12():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i1 = dpt.asarray([1, 1, 2], device=x.device)
    i2 = dpt.asarray([2, 0, 1], device=x.device)
    y = x[:, dpt.newaxis, i1, i2, dpt.newaxis]
    res_expected = dpt.asarray(
        [
            [[[lin_id(0, 1, 2)], [lin_id(0, 1, 0)], [lin_id(0, 2, 1)]]],
            [[[lin_id(1, 1, 2)], [lin_id(1, 1, 0)], [lin_id(1, 2, 1)]]],
            [[[lin_id(2, 1, 2)], [lin_id(2, 1, 0)], [lin_id(2, 2, 1)]]],
        ],
        sycl_queue=q,
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == res_expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(res_expected)).all()


def test_advanced_slice13():
    q = get_queue_or_skip()
    x = _make_3d("u8", q)
    i1 = dpt.asarray([[1], [2]], device=x.device)
    i2 = dpt.asarray([[0, 1]], device=x.device)
    y = x[i1, i2, 0]
    expected = dpt.asarray(
        [
            [lin_id(1, 0, 0), lin_id(1, 1, 0)],
            [lin_id(2, 0, 0), lin_id(2, 1, 0)],
        ],
        device=x.device,
    )
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == expected.shape
    assert (dpt.asnumpy(y) == dpt.asnumpy(expected)).all()


def test_advanced_slice14():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.reshape(dpt.arange(3**5, dtype="i4", sycl_queue=q), (3,) * 5)
    y = x[ii, 0, ii, 1, :]
    assert isinstance(y, dpt.usm_ndarray)
    # integers broadcast to ii.shape per array API
    assert y.shape == ii.shape + x.shape[-1:]
    assert _all_equal(
        (
            x[ii[i], 0, ii[i], 1, k]
            for i in range(ii.shape[0])
            for k in range(x.shape[-1])
        ),
        (y[i, k] for i in range(ii.shape[0]) for k in range(x.shape[-1])),
    )


def test_advanced_slice15():
    q = get_queue_or_skip()
    ii = dpt.asarray([1, 2], sycl_queue=q)
    x = dpt.reshape(dpt.arange(3**5, dtype="i4", sycl_queue=q), (3,) * 5)
    # : cannot appear between two integral arrays
    with pytest.raises(IndexError):
        x[ii, 0, ii, :, ii]


def test_advanced_slice16():
    q = get_queue_or_skip()
    ii = dpt.asarray(1, sycl_queue=q)
    i0 = dpt.asarray(False, sycl_queue=q)
    i1 = dpt.asarray(True, sycl_queue=q)
    x = dpt.reshape(dpt.arange(3**5, dtype="i4", sycl_queue=q), (3,) * 5)
    y = x[ii, i0, ii, i1, :]
    # TODO: add a shape check here when discrepancy with NumPy is investigated
    assert isinstance(y, dpt.usm_ndarray)


def test_integer_indexing_numpy_array():
    q = get_queue_or_skip()
    ii = np.asarray([1, 2])
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == ii.shape
    assert dpt.all(x[1:3] == y)


def test_boolean_indexing_numpy_array():
    q = get_queue_or_skip()
    ii = np.asarray(
        [False, True, True, False, False, False, False, False, False, False]
    )
    x = dpt.arange(10, dtype="i4", sycl_queue=q)
    y = x[ii]
    assert isinstance(y, dpt.usm_ndarray)
    assert y.shape == (2,)
    assert dpt.all(x[1:3] == y)


def test_boolean_indexing_validation():
    get_queue_or_skip()
    x = dpt.zeros(10, dtype="i4")
    ii = dpt.ones((2, 5), dtype="?")
    with pytest.raises(IndexError):
        x[ii]
    with pytest.raises(IndexError):
        x[ii[0, :]]


def test_boolean_indexing_getitem_empty_mask():
    get_queue_or_skip()
    x = dpt.ones((2, 3, 4), dtype="i4")
    ii = dpt.ones((0,), dtype="?")
    assert x[ii].size == 0
    ii1 = dpt.ones((0, 3), dtype="?")
    assert x[ii1].size == 0
    ii2 = dpt.ones((0, 3, 4), dtype="?")
    assert x[ii2].size == 0


def test_boolean_indexing_setitem_empty_mask():
    get_queue_or_skip()
    x = dpt.ones((2, 3, 4), dtype="i4")
    ii = dpt.ones((0,), dtype="?")
    x[ii] = 0
    assert dpt.all(x == 1)
    ii1 = dpt.ones((0, 3), dtype="?")
    x[ii1] = 0
    assert dpt.all(x == 1)
    ii2 = dpt.ones((0, 3, 4), dtype="?")
    x[ii2] = 0
    assert dpt.all(x == 1)


def test_integer_indexing_1d():
    get_queue_or_skip()
    x = dpt.arange(10, dtype="i4")
    ind_1d = dpt.asarray([7, 3, 1], dtype="u2")
    ind_2d = dpt.asarray([[2, 3, 4], [3, 4, 5], [5, 6, 7]], dtype="i4")

    y1 = x[ind_1d]
    assert y1.shape == ind_1d.shape
    y2 = x[ind_2d]
    assert y2.shape == ind_2d.shape
    assert (dpt.asnumpy(y1) == np.array([7, 3, 1], dtype="i4")).all()
    assert (
        dpt.asnumpy(y2)
        == np.array([[2, 3, 4], [3, 4, 5], [5, 6, 7]], dtype="i4")
    ).all()


def test_integer_indexing_2d():
    get_queue_or_skip()
    n0, n1 = 5, 7
    x = dpt.reshape(
        dpt.arange(n0 * n1, dtype="i4"),
        (
            n0,
            n1,
        ),
    )
    ind0 = dpt.arange(n0)
    ind1 = dpt.arange(n1)

    y = x[ind0[:2, dpt.newaxis], ind1[dpt.newaxis, -2:]]
    assert y.dtype == x.dtype
    assert (dpt.asnumpy(y) == np.array([[5, 6], [12, 13]])).all()


def test_integer_strided_indexing():
    get_queue_or_skip()
    n0, n1 = 5, 7
    x = dpt.reshape(
        dpt.arange(2 * n0 * n1, dtype="i4"),
        (
            2 * n0,
            n1,
        ),
    )
    ind0 = dpt.arange(n0)
    ind1 = dpt.arange(n1)

    z = x[::-2, :]
    y = z[ind0[:2, dpt.newaxis], ind1[dpt.newaxis, -2:]]
    assert y.dtype == x.dtype
    zc = dpt.copy(z, order="C")
    yc = zc[ind0[:2, dpt.newaxis], ind1[dpt.newaxis, -2:]]
    assert (dpt.asnumpy(y) == dpt.asnumpy(yc)).all()


def test_TrueFalse_indexing():
    get_queue_or_skip()
    n0, n1 = 2, 3
    x = dpt.ones((n0, n1))
    for ind in [True, dpt.asarray(True)]:
        y1 = x[ind]
        assert y1.shape == (1, n0, n1)
        assert y1._pointer == x._pointer
        y2 = x[:, ind]
        assert y2.shape == (n0, 1, n1)
        assert y2._pointer == x._pointer
        y3 = x[..., ind]
        assert y3.shape == (n0, n1, 1)
        assert y3._pointer == x._pointer
    for ind in [False, dpt.asarray(False)]:
        y1 = x[ind]
        assert y1.shape == (0, n0, n1)
        assert y1._pointer == x._pointer
        y2 = x[:, ind]
        assert y2.shape == (n0, 0, n1)
        assert y2._pointer == x._pointer
        y3 = x[..., ind]
        assert y3.shape == (n0, n1, 0)
        assert y3._pointer == x._pointer


def test_mixed_index_getitem():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(1000, dtype="i4"), (10, 10, 10))
    i1b = dpt.ones(10, dtype="?")
    info = x.__array_namespace__().__array_namespace_info__()
    ind_dt = info.default_dtypes(device=x.device)["indexing"]
    i0 = dpt.asarray([0, 2, 3], dtype=ind_dt)[:, dpt.newaxis]
    i2 = dpt.asarray([3, 4, 7], dtype=ind_dt)[:, dpt.newaxis]
    y = x[i0, i1b, i2]
    assert y.shape == (3, dpt.sum(i1b, dtype="i8"))


def test_mixed_index_setitem():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(1000, dtype="i4"), (10, 10, 10))
    i1b = dpt.ones(10, dtype="?")
    info = x.__array_namespace__().__array_namespace_info__()
    ind_dt = info.default_dtypes(device=x.device)["indexing"]
    i0 = dpt.asarray([0, 2, 3], dtype=ind_dt)[:, dpt.newaxis]
    i2 = dpt.asarray([3, 4, 7], dtype=ind_dt)[:, dpt.newaxis]
    v_shape = (3, int(dpt.sum(i1b, dtype="i8")))
    canary = 7
    x[i0, i1b, i2] = dpt.full(v_shape, canary, dtype=x.dtype)
    assert x[0, 0, 3] == canary


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
def test_take_basic(data_dt, ind_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.arange(10, dtype=data_dt)
    ind = dpt.arange(2, 5, dtype=ind_dt)
    y = dpt.take(x, ind)
    assert y.dtype == x.dtype
    assert (dpt.asnumpy(y) == np.arange(2, 5, dtype=data_dt)).all()


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
def test_put_basic(data_dt, ind_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.arange(10, dtype=data_dt)
    ind = dpt.arange(2, 5, dtype=ind_dt)
    val = dpt.ones(3, dtype=data_dt)
    dpt.put(x, ind, val)
    assert (
        dpt.asnumpy(x)
        == np.array([0, 1, 1, 1, 1, 5, 6, 7, 8, 9], dtype=data_dt)
    ).all()


def test_take_basic_axis():
    get_queue_or_skip()

    n0, n1 = 5, 7
    x = dpt.reshape(
        dpt.arange(n0 * n1, dtype="i4"),
        (
            n0,
            n1,
        ),
    )
    ind = dpt.arange(2, 4)
    y0 = dpt.take(x, ind, axis=0)
    y1 = dpt.take(x, ind, axis=1)
    assert y0.shape == (2, n1)
    assert y1.shape == (n0, 2)


def test_put_basic_axis():
    get_queue_or_skip()

    n0, n1 = 5, 7
    x = dpt.reshape(
        dpt.arange(n0 * n1, dtype="i4"),
        (
            n0,
            n1,
        ),
    )
    ind = dpt.arange(2, 4)
    v0 = dpt.zeros((2, n1), dtype=x.dtype)
    v1 = dpt.zeros((n0, 2), dtype=x.dtype)
    dpt.put(x, ind, v0, axis=0)
    dpt.put(x, ind, v1, axis=1)
    expected = np.arange(n0 * n1, dtype="i4").reshape((n0, n1))
    expected[[2, 3], :] = 0
    expected[:, [2, 3]] = 0
    assert (expected == dpt.asnumpy(x)).all()


@pytest.mark.parametrize("data_dt", _all_dtypes)
def test_put_0d_val(data_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.arange(5, dtype=data_dt, sycl_queue=q)
    ind = dpt.asarray([0], dtype="i8", sycl_queue=q)
    val = dpt.asarray(2, dtype=x.dtype, sycl_queue=q)
    x[ind] = val
    assert_array_equal(np.asarray(2, dtype=data_dt), dpt.asnumpy(x[0]))

    x = dpt.asarray(5, dtype=data_dt, sycl_queue=q)
    dpt.put(x, ind, val)
    assert_array_equal(np.asarray(2, dtype=data_dt), dpt.asnumpy(x))


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
def test_take_0d_data(data_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.asarray(0, dtype=data_dt, sycl_queue=q)
    ind = dpt.arange(5, dtype="i8", sycl_queue=q)

    y = dpt.take(x, ind)
    assert (
        dpt.asnumpy(y)
        == np.broadcast_to(np.asarray(0, dtype=data_dt), ind.shape)
    ).all()


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
def test_put_0d_data(data_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.asarray(0, dtype=data_dt, sycl_queue=q)
    ind = dpt.arange(5, dtype="i8", sycl_queue=q)
    val = dpt.asarray(2, dtype=data_dt, sycl_queue=q)

    dpt.put(x, ind, val, axis=0)
    assert (
        dpt.asnumpy(x)
        == np.broadcast_to(np.asarray(2, dtype=data_dt), ind.shape)
    ).all()


@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
def test_indexing_0d_ind(ind_dt):
    q = get_queue_or_skip()

    x = dpt.arange(5, dtype="i4", sycl_queue=q)
    ind = dpt.asarray(3, dtype=ind_dt, sycl_queue=q)

    y = x[ind]
    assert dpt.asnumpy(x[3]) == dpt.asnumpy(y)


@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
def test_put_0d_ind(ind_dt):
    q = get_queue_or_skip()

    x = dpt.arange(5, dtype="i4", sycl_queue=q)
    ind = dpt.asarray(3, dtype=ind_dt, sycl_queue=q)
    val = dpt.asarray(5, dtype=x.dtype, sycl_queue=q)

    x[ind] = val
    assert dpt.asnumpy(x[3]) == dpt.asnumpy(val)


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
def test_take_strided_1d_source(data_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.arange(27, dtype=data_dt, sycl_queue=q)
    ind = dpt.arange(4, 9, dtype="i8", sycl_queue=q)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        assert_array_equal(
            np.take(x_np[s], ind_np, axis=0),
            dpt.asnumpy(dpt.take(x[s], ind, axis=0)),
        )

    # 0-strided
    x = dpt.usm_ndarray(
        (27,),
        dtype=data_dt,
        strides=(0,),
        buffer_ctor_kwargs={"queue": q},
    )
    x[0] = x_np[0]
    assert_array_equal(
        np.broadcast_to(x_np[0], ind.shape),
        dpt.asnumpy(dpt.take(x, ind, axis=0)),
    )


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_take_strided(data_dt, order):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.reshape(_make_3d(data_dt, q), (9, 3), order=order)
    ind = dpt.arange(2, dtype="i8", sycl_queue=q)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        for sgn in (-1, 1):
            xs = x[s, ::sgn]
            xs_np = x_np[s, ::sgn]
            assert_array_equal(
                np.take(xs_np, ind_np, axis=0),
                dpt.asnumpy(dpt.take(xs, ind, axis=0)),
            )
            assert_array_equal(
                np.take(xs_np, ind_np, axis=1),
                dpt.asnumpy(dpt.take(xs, ind, axis=1)),
            )


@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
def test_take_strided_1d_indices(ind_dt):
    q = get_queue_or_skip()

    x = dpt.arange(27, dtype="i4", sycl_queue=q)
    ind = dpt.arange(12, 24, dtype=ind_dt, sycl_queue=q)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind).astype("i8")

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        assert_array_equal(
            np.take(x_np, ind_np[s], axis=0),
            dpt.asnumpy(dpt.take(x, ind[s], axis=0)),
        )

    # 0-strided
    ind = dpt.usm_ndarray(
        (12,),
        dtype=ind_dt,
        strides=(0,),
        buffer_ctor_kwargs={"queue": q},
    )
    ind[0] = ind_np[0]
    assert_array_equal(
        np.broadcast_to(x_np[ind_np[0]], ind.shape),
        dpt.asnumpy(dpt.take(x, ind, axis=0)),
    )


@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_take_strided_indices(ind_dt, order):
    q = get_queue_or_skip()

    x = dpt.arange(27, dtype="i4", sycl_queue=q)
    ind = dpt.reshape(
        dpt.arange(12, 24, dtype=ind_dt, sycl_queue=q), (4, 3), order=order
    )

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind).astype("i8")

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        for sgn in [-1, 1]:
            inds = ind[s, ::sgn]
            inds_np = ind_np[s, ::sgn]
            assert_array_equal(
                np.take(x_np, inds_np, axis=0),
                dpt.asnumpy(x[inds]),
            )


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_put_strided_1d_destination(data_dt, order):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.arange(27, dtype=data_dt, sycl_queue=q)
    ind = dpt.arange(4, 9, dtype="i8", sycl_queue=q)
    val = dpt.asarray(9, dtype=x.dtype, sycl_queue=q)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind)
    val_np = dpt.asnumpy(val)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        x_np1 = x_np.copy()
        x_np1[s][ind_np] = val_np

        x1 = dpt.copy(x)
        dpt.put(x1[s], ind, val, axis=0)

        assert_array_equal(x_np1, dpt.asnumpy(x1))


@pytest.mark.parametrize(
    "data_dt",
    _all_dtypes,
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_put_strided_destination(data_dt, order):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    x = dpt.reshape(_make_3d(data_dt, q), (9, 3), order=order)
    ind = dpt.arange(2, dtype="i8", sycl_queue=q)
    val = dpt.asarray(9, dtype=x.dtype, sycl_queue=q)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind)
    val_np = dpt.asnumpy(val)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        for sgn in [-1, 1]:
            xs = x[s, ::sgn]
            xs_np = x_np[s, ::sgn]

            x_np1 = xs_np.copy()
            x_np1[ind_np] = val_np

            x1 = dpt.copy(xs)
            dpt.put(x1, ind, val, axis=0)
            assert_array_equal(x_np1, dpt.asnumpy(x1))

            x_np1 = xs_np.copy()
            x_np1[:, ind_np] = val_np

            x1 = dpt.copy(xs)
            dpt.put(x1, ind, val, axis=1)
            assert_array_equal(x_np1, dpt.asnumpy(x1))

            x_np1 = xs_np.copy()
            x_np1[ind_np, ind_np] = val_np

            x1 = dpt.copy(xs)
            x1[ind, ind] = val
            assert_array_equal(x_np1, dpt.asnumpy(x1))


@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
def test_put_strided_1d_indices(ind_dt):
    q = get_queue_or_skip()

    x = dpt.arange(27, dtype="i4", sycl_queue=q)
    ind = dpt.arange(12, 24, dtype=ind_dt, sycl_queue=q)
    val = dpt.asarray(-1, dtype=x.dtype, sycl_queue=q)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind).astype("i8")
    val_np = dpt.asnumpy(val)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        x_copy = dpt.copy(x)
        dpt.put(x_copy, ind[s], val, axis=0)

        x_np_copy = x_np.copy()
        x_np_copy[ind_np[s]] = val_np

        assert_array_equal(x_np_copy, dpt.asnumpy(x_copy))


@pytest.mark.parametrize(
    "ind_dt",
    _all_int_dtypes,
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_put_strided_indices(ind_dt, order):
    q = get_queue_or_skip()

    x = dpt.arange(27, dtype="i4", sycl_queue=q)
    ind = dpt.reshape(
        dpt.arange(12, 24, dtype=ind_dt, sycl_queue=q), (4, 3), order=order
    )
    val = dpt.asarray(-1, sycl_queue=q, dtype=x.dtype)

    x_np = dpt.asnumpy(x)
    ind_np = dpt.asnumpy(ind).astype("i8")
    val_np = dpt.asnumpy(val)

    for s in (
        slice(None, None, 2),
        slice(None, None, -2),
    ):
        for sgn in [-1, 1]:
            inds = ind[s, ::sgn]
            inds_np = ind_np[s, ::sgn]

            x_copy = dpt.copy(x)
            x_copy[inds] = val

            x_np_copy = x_np.copy()
            x_np_copy[inds_np] = val_np

            assert_array_equal(x_np_copy, dpt.asnumpy(x_copy))


def test_integer_indexing_modes():
    q = get_queue_or_skip()

    x = dpt.arange(5, sycl_queue=q)
    x_np = dpt.asnumpy(x)

    # wrapping negative indices
    ind = dpt.asarray([-4, -3, 0, 2, 4], dtype="i8", sycl_queue=q)

    res = dpt.take(x, ind, mode="wrap")
    expected_arr = np.take(x_np, dpt.asnumpy(ind), mode="raise")

    assert (dpt.asnumpy(res) == expected_arr).all()

    # clipping to 0 (disabling negative indices)
    ind = dpt.asarray([-6, -3, 0, 2, 6], dtype="i8", sycl_queue=q)

    res = dpt.take(x, ind, mode="clip")
    expected_arr = np.take(x_np, dpt.asnumpy(ind), mode="clip")

    assert (dpt.asnumpy(res) == expected_arr).all()


def test_take_arg_validation():
    q = get_queue_or_skip()

    x = dpt.arange(4, dtype="i4", sycl_queue=q)
    ind0 = dpt.arange(4, dtype="i8", sycl_queue=q)
    ind1 = dpt.arange(2.0, dtype="f", sycl_queue=q)

    with pytest.raises(TypeError):
        dpt.take(dict(), ind0, axis=0)
    with pytest.raises(TypeError):
        dpt.take(x, dict(), axis=0)
    with pytest.raises(IndexError):
        x[[]]
    with pytest.raises(IndexError):
        dpt.take(x, ind1, axis=0)
    with pytest.raises(IndexError):
        x[ind1]

    with pytest.raises(ValueError):
        dpt.take(dpt.reshape(x, (2, 2)), ind0)
    with pytest.raises(ValueError):
        dpt.take(x, ind0, mode=0)
    with pytest.raises(ValueError):
        dpt.take(dpt.reshape(x, (2, 2)), ind0, axis=None)
    with pytest.raises(ValueError):
        dpt.take(x, dpt.reshape(ind0, (2, 2)))
    with pytest.raises(ValueError):
        dpt.take(x[0], ind0, axis=2)
    with pytest.raises(ValueError):
        dpt.take(x[:, dpt.newaxis, dpt.newaxis], ind0, axis=None)


def test_put_arg_validation():
    q = get_queue_or_skip()

    x = dpt.arange(4, dtype="i4", sycl_queue=q)
    ind0 = dpt.arange(4, dtype="i8", sycl_queue=q)
    ind1 = dpt.arange(2.0, dtype="f", sycl_queue=q)
    val = dpt.asarray(2, dtype=x.dtype, sycl_queue=q)

    with pytest.raises(TypeError):
        dpt.put(dict(), ind0, val, axis=0)
    with pytest.raises(TypeError):
        dpt.put(x, dict(), val, axis=0)
    with pytest.raises(IndexError):
        x[[]] = val
    with pytest.raises(IndexError):
        dpt.put(x, ind1, val, axis=0)
    with pytest.raises(IndexError):
        x[ind1] = val
    with pytest.raises(TypeError):
        dpt.put(x, ind0, {}, axis=0)
    with pytest.raises(TypeError):
        x[ind0] = {}

    with pytest.raises(ValueError):
        dpt.put(x, ind0, val, mode=0)
    with pytest.raises(ValueError):
        dpt.put(x, dpt.reshape(ind0, (2, 2)), val)
    with pytest.raises(ValueError):
        dpt.put(x[0], ind0, val, axis=2)
    with pytest.raises(ValueError):
        dpt.put(x[:, dpt.newaxis, dpt.newaxis], ind0, val, axis=None)


def test_advanced_indexing_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    x = dpt.arange(4, sycl_queue=q1)
    ind0 = dpt.asarray([0], sycl_queue=q1)
    ind1 = dpt.asarray([0], sycl_queue=q2)
    val0 = dpt.asarray(2, dtype=x.dtype, sycl_queue=q1)
    val1 = dpt.asarray(2, dtype=x.dtype, sycl_queue=q2)

    with pytest.raises(ExecutionPlacementError):
        dpt.take(x, ind1, axis=0)
    with pytest.raises(ExecutionPlacementError):
        x[ind1]
    with pytest.raises(ExecutionPlacementError):
        dpt.put(x, ind1, val0, axis=0)
    with pytest.raises(ExecutionPlacementError):
        x[ind1] = val0
    with pytest.raises(ExecutionPlacementError):
        dpt.put(x, ind0, val1, axis=0)
    with pytest.raises(ExecutionPlacementError):
        x[ind0] = val1


def test_extract_all_1d():
    get_queue_or_skip()
    x = dpt.arange(30, dtype="i4")
    sel = dpt.ones(30, dtype="?")
    sel[::2] = False

    res = x[sel]
    expected_res = dpt.asnumpy(x)[dpt.asnumpy(sel)]
    assert (dpt.asnumpy(res) == expected_res).all()

    res2 = dpt.extract(sel, x)
    assert (dpt.asnumpy(res2) == expected_res).all()

    # test strided case
    x = dpt.arange(15, dtype="i4")
    sel_np = np.zeros(15, dtype="?")
    np.put(sel_np, np.random.choice(sel_np.size, size=7), True)
    sel = dpt.asarray(sel_np)

    res = x[sel[::-1]]
    expected_res = dpt.asnumpy(x)[sel_np[::-1]]
    assert (dpt.asnumpy(res) == expected_res).all()

    res2 = dpt.extract(sel[::-1], x)
    assert (dpt.asnumpy(res2) == expected_res).all()


def test_extract_all_2d():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(30, dtype="i4"), (5, 6))
    sel = dpt.ones(30, dtype="?")
    sel[::2] = False
    sel = dpt.reshape(sel, x.shape)

    res = x[sel]
    expected_res = dpt.asnumpy(x)[dpt.asnumpy(sel)]
    assert (dpt.asnumpy(res) == expected_res).all()

    res2 = dpt.extract(sel, x)
    assert (dpt.asnumpy(res2) == expected_res).all()


def test_extract_2D_axis0():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(30, dtype="i4"), (5, 6))
    sel = dpt.ones(x.shape[0], dtype="?")
    sel[::2] = False

    res = x[sel]
    expected_res = dpt.asnumpy(x)[dpt.asnumpy(sel)]
    assert (dpt.asnumpy(res) == expected_res).all()


def test_extract_2D_axis1():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(30, dtype="i4"), (5, 6))
    sel = dpt.ones(x.shape[1], dtype="?")
    sel[::2] = False

    res = x[:, sel]
    expected = dpt.asnumpy(x)[:, dpt.asnumpy(sel)]
    assert (dpt.asnumpy(res) == expected).all()


def test_extract_begin():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((3, 3), dtype="?")
    sel[0, 0] = True
    sel[1, 1] = True
    z = y[sel]
    expected = dpt.asnumpy(y)[[0, 1], [0, 1]]
    assert (dpt.asnumpy(z) == expected).all()


def test_extract_end():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((4, 4), dtype="?")
    sel[0, 0] = True
    z = y[..., sel]
    expected = dpt.asnumpy(y)[..., [0], [0]]
    assert (dpt.asnumpy(z) == expected).all()


def test_extract_middle():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((3, 4), dtype="?")
    sel[0, 0] = True
    z = y[:, sel]
    expected = dpt.asnumpy(y)[:, [0], [0], :]
    assert (dpt.asnumpy(z) == expected).all()


def test_extract_empty_result():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((3, 4), dtype="?")
    z = y[:, sel]
    assert z.shape == (
        y.shape[0],
        0,
        y.shape[3],
    )


def test_place_all_1d():
    get_queue_or_skip()
    x = dpt.arange(10, dtype="i2")
    sel = dpt.zeros(10, dtype="?")
    sel[0::2] = True
    val = dpt.zeros(5, dtype=x.dtype)
    x[sel] = val
    assert (dpt.asnumpy(x) == np.array([0, 1, 0, 3, 0, 5, 0, 7, 0, 9])).all()
    dpt.place(x, sel, dpt.asarray([2]))
    assert (dpt.asnumpy(x) == np.array([2, 1, 2, 3, 2, 5, 2, 7, 2, 9])).all()


def test_place_2d_axis0():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(12, dtype="i2"), (3, 4))
    sel = dpt.asarray([True, False, True])
    val = dpt.zeros((2, 4), dtype=x.dtype)
    x[sel] = val
    expected_x = np.stack(
        (
            np.zeros(4, dtype="i2"),
            np.arange(4, 8, dtype="i2"),
            np.zeros(4, dtype="i2"),
        )
    )
    assert (dpt.asnumpy(x) == expected_x).all()


def test_place_2d_axis1():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(12, dtype="i2"), (3, 4))
    sel = dpt.asarray([True, False, True, False])
    val = dpt.zeros((3, 2), dtype=x.dtype)
    x[:, sel] = val
    expected_x = np.array(
        [[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11]], dtype="i2"
    )
    assert (dpt.asnumpy(x) == expected_x).all()


def test_place_2d_axis1_scalar():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(12, dtype="i2"), (3, 4))
    sel = dpt.asarray([True, False, True, False])
    val = dpt.zeros(tuple(), dtype=x.dtype)
    x[:, sel] = val
    expected_x = np.array(
        [[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11]], dtype="i2"
    )
    assert (dpt.asnumpy(x) == expected_x).all()


def test_place_all_slices():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(12, dtype="i2"), (3, 4))
    sel = dpt.asarray(
        [
            [False, True, True, False],
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype="?",
    )
    y = dpt.ones_like(x)
    y[sel] = x[sel]


def test_place_some_slices_begin():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((3, 3), dtype="?")
    sel[0, 0] = True
    sel[1, 1] = True
    z = y[sel]
    w = dpt.zeros_like(y)
    w[sel] = z


def test_place_some_slices_mid():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((3, 4), dtype="?")
    sel[0, 0] = True
    sel[1, 1] = True
    z = y[:, sel]
    w = dpt.zeros_like(y)
    w[:, sel] = z


def test_place_some_slices_end():
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(3 * 3 * 4 * 4, dtype="i2"), (3, 4, 3, 4))
    y = dpt.permute_dims(x, (2, 0, 3, 1))
    sel = dpt.zeros((4, 4), dtype="?")
    sel[0, 0] = True
    sel[1, 1] = True
    z = y[:, :, sel]
    w = dpt.zeros_like(y)
    w[:, :, sel] = z


def test_place_cycling():
    get_queue_or_skip()
    x = dpt.zeros(10, dtype="f4")
    y = dpt.asarray([2, 3])
    sel = dpt.ones(x.size, dtype="?")
    dpt.place(x, sel, y)
    expected = np.array(
        [
            2,
            3,
        ]
        * 5,
        dtype=x.dtype,
    )
    assert (dpt.asnumpy(x) == expected).all()


def test_place_subset():
    get_queue_or_skip()
    x = dpt.zeros(10, dtype="f4")
    y = dpt.ones_like(x)
    sel = dpt.ones(x.size, dtype="?")
    sel[::2] = False
    dpt.place(x, sel, y)
    expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=x.dtype)
    assert (dpt.asnumpy(x) == expected).all()


def test_place_empty_vals_error():
    get_queue_or_skip()
    x = dpt.zeros(10, dtype="f4")
    y = dpt.empty((0,), dtype=x.dtype)
    sel = dpt.ones(x.size, dtype="?")
    sel[::2] = False
    with pytest.raises(ValueError):
        dpt.place(x, sel, y)


def test_place_empty_vals_full_false_mask():
    get_queue_or_skip()
    x = dpt.ones(10, dtype="f4")
    y = dpt.empty((0,), dtype=x.dtype)
    sel = dpt.zeros(x.size, dtype="?")
    expected = np.ones(10, dtype=x.dtype)
    dpt.place(x, sel, y)
    assert (dpt.asnumpy(x) == expected).all()


def test_nonzero():
    get_queue_or_skip()
    x = dpt.concat((dpt.zeros(3), dpt.ones(4), dpt.zeros(3)))
    (i,) = dpt.nonzero(x)
    assert (dpt.asnumpy(i) == np.array([3, 4, 5, 6])).all()


def test_nonzero_f_contig():
    "See gh-1370"
    get_queue_or_skip()

    mask = dpt.zeros((5, 5), dtype="?", order="F")
    mask[2, 3] = True

    expected_res = np.nonzero(dpt.asnumpy(mask))
    result = dpt.nonzero(mask)

    for exp, res in zip(expected_res, result):
        assert_array_equal(dpt.asnumpy(res), exp)
    assert dpt.asnumpy(mask[result]).all()


def test_nonzero_compacting():
    """See gh-1370.
    Test with input where dimensionality
    of iteration space is compacted from 3d to 2d
    """
    get_queue_or_skip()

    mask = dpt.zeros((5, 5, 5), dtype="?", order="F")
    mask[3, 2, 1] = True
    mask_view = mask[..., :3]

    expected_res = np.nonzero(dpt.asnumpy(mask_view))
    result = dpt.nonzero(mask_view)

    for exp, res in zip(expected_res, result):
        assert_array_equal(dpt.asnumpy(res), exp)
    assert dpt.asnumpy(mask_view[result]).all()


def test_assign_scalar():
    get_queue_or_skip()
    x = dpt.arange(-5, 5, dtype="i8")
    cond = dpt.asarray(
        [True, True, True, True, True, False, False, False, False, False]
    )
    x[cond] = 0  # no error expected
    x[dpt.nonzero(cond)] = -1
    expected = np.array([-1, -1, -1, -1, -1, 0, 1, 2, 3, 4], dtype=x.dtype)
    assert (dpt.asnumpy(x) == expected).all()


def test_nonzero_large():
    get_queue_or_skip()
    m = dpt.full((60, 80), True)
    assert m[m].size == m.size

    m = dpt.full((30, 60, 80), True)
    assert m[m].size == m.size


def test_extract_arg_validation():
    get_queue_or_skip()
    with pytest.raises(TypeError):
        dpt.extract(None, None)
    cond = dpt.ones(10, dtype="?")
    with pytest.raises(TypeError):
        dpt.extract(cond, None)
    q1 = dpctl.SyclQueue()
    with pytest.raises(ExecutionPlacementError):
        dpt.extract(cond.to_device(q1), dpt.zeros_like(cond, dtype="u1"))
    with pytest.raises(ValueError):
        dpt.extract(dpt.ones((2, 3), dtype="?"), dpt.ones((3, 2), dtype="i1"))


def test_place_arg_validation():
    get_queue_or_skip()
    with pytest.raises(TypeError):
        dpt.place(None, None, None)
    arr = dpt.zeros(8, dtype="i1")
    with pytest.raises(TypeError):
        dpt.place(arr, None, None)
    cond = dpt.ones(8, dtype="?")
    with pytest.raises(TypeError):
        dpt.place(arr, cond, None)
    vals = dpt.ones_like(arr)
    q1 = dpctl.SyclQueue()
    with pytest.raises(ExecutionPlacementError):
        dpt.place(arr.to_device(q1), cond, vals)
    with pytest.raises(ValueError):
        dpt.place(dpt.reshape(arr, (2, 2, 2)), cond, vals)


def test_nonzero_arg_validation():
    get_queue_or_skip()
    with pytest.raises(TypeError):
        dpt.nonzero(list())
    with pytest.raises(ValueError):
        dpt.nonzero(dpt.asarray(1))


def test_nonzero_dtype():
    "See gh-1322"
    get_queue_or_skip()
    x = dpt.ones((3, 4))
    idx, idy = dpt.nonzero(x)
    # create array using device's
    # default index data type
    index_dt = dpt.dtype(ti.default_device_index_type(x.sycl_queue))
    assert idx.dtype == index_dt
    assert idy.dtype == index_dt


def test_take_empty_axes():
    get_queue_or_skip()

    x = dpt.ones((3, 0, 4, 5, 6), dtype="f4")
    inds = dpt.ones(1, dtype="i4")

    with pytest.raises(IndexError):
        dpt.take(x, inds, axis=1)

    inds = dpt.ones(0, dtype="i4")
    r = dpt.take(x, inds, axis=1)
    assert r.shape == x.shape


def test_put_empty_axes():
    get_queue_or_skip()

    x = dpt.ones((3, 0, 4, 5, 6), dtype="f4")
    inds = dpt.ones(1, dtype="i4")
    vals = dpt.zeros((3, 1, 4, 5, 6), dtype="f4")

    with pytest.raises(IndexError):
        dpt.put(x, inds, vals, axis=1)

    inds = dpt.ones(0, dtype="i4")
    vals = dpt.zeros_like(x)

    with pytest.raises(ValueError):
        dpt.put(x, inds, vals, axis=1)


def test_put_cast_vals():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    inds = dpt.arange(7, 10, dtype="i4")
    vals = dpt.zeros_like(inds, dtype="f4")

    dpt.put(x, inds, vals)
    assert dpt.all(x[7:10] == 0)


def test_advanced_integer_indexing_cast_vals():
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    inds = dpt.arange(7, 10, dtype="i4")
    vals = dpt.zeros_like(inds, dtype="f4")

    x[inds] = vals
    assert dpt.all(x[7:10] == 0)


def test_advanced_integer_indexing_empty_axis():
    get_queue_or_skip()

    # getting
    x = dpt.ones((3, 0, 4, 5, 6), dtype="f4")
    inds = dpt.ones(1, dtype="i4")
    with pytest.raises(IndexError):
        x[:, inds, ...]
    with pytest.raises(IndexError):
        x[inds, inds, inds, ...]

    # setting
    with pytest.raises(IndexError):
        x[:, inds, ...] = 2
    with pytest.raises(IndexError):
        x[inds, inds, inds, ...] = 2

    # empty inds
    inds = dpt.ones(0, dtype="i4")
    assert x[:, inds, ...].shape == x.shape
    assert x[inds, inds, inds, ...].shape == (0, 5, 6)

    vals = dpt.zeros_like(x)
    x[:, inds, ...] = vals
    vals = dpt.zeros((0, 5, 6), dtype="f4")
    x[inds, inds, inds, ...] = vals


def test_advanced_integer_indexing_cast_indices():
    get_queue_or_skip()

    inds0 = dpt.asarray([0, 1], dtype="i1")
    for ind_dts in (("i1", "i2", "i4"), ("i1", "u4", "i4"), ("u1", "u2", "u8")):
        x = dpt.ones((3, 4, 5, 6), dtype="i4")
        inds0 = dpt.asarray([0, 1], dtype=ind_dts[0])
        inds1 = dpt.astype(inds0, ind_dts[1])
        x[inds0, inds1, ...] = 2
        assert dpt.all(x[inds0, inds1, ...] == 2)
        inds2 = dpt.astype(inds0, ind_dts[2])
        x[inds0, inds1, ...] = 2
        assert dpt.all(x[inds0, inds1, inds2, ...] == 2)

    # fail when float would be required per type promotion
    inds0 = dpt.asarray([0, 1], dtype="i1")
    inds1 = dpt.astype(inds0, "u4")
    inds2 = dpt.astype(inds0, "u8")
    x = dpt.ones((3, 4, 5, 6), dtype="i4")
    # test getitem
    with pytest.raises(ValueError):
        x[inds0, inds1, inds2, ...]
    # test setitem
    with pytest.raises(ValueError):
        x[inds0, inds1, inds2, ...] = 1


def test_take_along_axis():
    get_queue_or_skip()

    n0, n1, n2 = 3, 5, 7
    x = dpt.reshape(dpt.arange(n0 * n1 * n2), (n0, n1, n2))
    ind_dt = dpt.__array_namespace_info__().default_dtypes(
        device=x.sycl_device
    )["indexing"]
    ind0 = dpt.ones((1, n1, n2), dtype=ind_dt)
    ind1 = dpt.ones((n0, 1, n2), dtype=ind_dt)
    ind2 = dpt.ones((n0, n1, 1), dtype=ind_dt)

    y0 = dpt.take_along_axis(x, ind0, axis=0)
    assert y0.shape == ind0.shape
    y1 = dpt.take_along_axis(x, ind1, axis=1)
    assert y1.shape == ind1.shape
    y2 = dpt.take_along_axis(x, ind2, axis=2)
    assert y2.shape == ind2.shape


def test_take_along_axis_validation():
    # validate first argument
    with pytest.raises(TypeError):
        dpt.take_along_axis(tuple(), list())
    get_queue_or_skip()
    n1, n2 = 2, 5
    x = dpt.ones(n1 * n2)
    # validate second argument
    with pytest.raises(TypeError):
        dpt.take_along_axis(x, list())
    x_dev = x.sycl_device
    info_ = dpt.__array_namespace_info__()
    def_dtypes = info_.default_dtypes(device=x_dev)
    ind_dt = def_dtypes["indexing"]
    ind = dpt.zeros(1, dtype=ind_dt)
    # axis validation
    with pytest.raises(ValueError):
        dpt.take_along_axis(x, ind, axis=1)
    # mode validation
    with pytest.raises(ValueError):
        dpt.take_along_axis(x, ind, axis=0, mode="invalid")
    # same array-ranks validation
    with pytest.raises(ValueError):
        dpt.take_along_axis(dpt.reshape(x, (n1, n2)), ind)
    # check compute-follows-data
    q2 = dpctl.SyclQueue(x_dev, property="enable_profiling")
    ind2 = dpt.zeros(1, dtype=ind_dt, sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.take_along_axis(x, ind2)


def test_put_along_axis():
    get_queue_or_skip()

    n0, n1, n2 = 3, 5, 7
    x = dpt.reshape(dpt.arange(n0 * n1 * n2), (n0, n1, n2))
    ind_dt = dpt.__array_namespace_info__().default_dtypes(
        device=x.sycl_device
    )["indexing"]
    ind0 = dpt.ones((1, n1, n2), dtype=ind_dt)
    ind1 = dpt.ones((n0, 1, n2), dtype=ind_dt)
    ind2 = dpt.ones((n0, n1, 1), dtype=ind_dt)

    xc = dpt.copy(x)
    vals = dpt.ones(ind0.shape, dtype=x.dtype)
    dpt.put_along_axis(xc, ind0, vals, axis=0)
    assert dpt.all(dpt.take_along_axis(xc, ind0, axis=0) == vals)

    xc = dpt.copy(x)
    vals = dpt.ones(ind1.shape, dtype=x.dtype)
    dpt.put_along_axis(xc, ind1, vals, axis=1)
    assert dpt.all(dpt.take_along_axis(xc, ind1, axis=1) == vals)

    xc = dpt.copy(x)
    vals = dpt.ones(ind2.shape, dtype=x.dtype)
    dpt.put_along_axis(xc, ind2, vals, axis=2)
    assert dpt.all(dpt.take_along_axis(xc, ind2, axis=2) == vals)

    xc = dpt.copy(x)
    vals = dpt.ones(ind2.shape, dtype=x.dtype)
    dpt.put_along_axis(xc, ind2, dpt.asnumpy(vals), axis=2)
    assert dpt.all(dpt.take_along_axis(xc, ind2, axis=2) == vals)


def test_put_along_axis_validation():
    # validate first argument
    with pytest.raises(TypeError):
        dpt.put_along_axis(tuple(), list(), list())
    get_queue_or_skip()
    n1, n2 = 2, 5
    x = dpt.ones(n1 * n2)
    # validate second argument
    with pytest.raises(TypeError):
        dpt.put_along_axis(x, list(), list())
    x_dev = x.sycl_device
    info_ = dpt.__array_namespace_info__()
    def_dtypes = info_.default_dtypes(device=x_dev)
    ind_dt = def_dtypes["indexing"]
    ind = dpt.zeros(1, dtype=ind_dt)
    vals = dpt.zeros(1, dtype=x.dtype)
    # axis validation
    with pytest.raises(ValueError):
        dpt.put_along_axis(x, ind, vals, axis=1)
    # mode validation
    with pytest.raises(ValueError):
        dpt.put_along_axis(x, ind, vals, axis=0, mode="invalid")
    # same array-ranks validation
    with pytest.raises(ValueError):
        dpt.put_along_axis(dpt.reshape(x, (n1, n2)), ind, vals)
    # check compute-follows-data
    q2 = dpctl.SyclQueue(x_dev, property="enable_profiling")
    ind2 = dpt.zeros(1, dtype=ind_dt, sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.put_along_axis(x, ind2, vals)


def test_put_along_axis_application():
    get_queue_or_skip()
    info_ = dpt.__array_namespace_info__()
    def_dtypes = info_.default_dtypes(device=None)
    ind_dt = def_dtypes["indexing"]
    all_perms = dpt.asarray(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [2, 0, 1, 3],
            [2, 1, 0, 3],
            [1, 0, 2, 3],
            [1, 2, 0, 3],
            [0, 1, 3, 2],
            [0, 2, 3, 1],
            [2, 0, 3, 1],
            [2, 1, 3, 0],
            [1, 0, 3, 2],
            [1, 2, 3, 0],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
        ],
        dtype=ind_dt,
    )
    p_mats = dpt.zeros((24, 4, 4), dtype=dpt.int64)
    vals = dpt.ones((24, 4, 1), dtype=p_mats.dtype)
    # form 24 permutation matrices
    dpt.put_along_axis(p_mats, all_perms[..., dpt.newaxis], vals, axis=2)
    p2 = p_mats @ p_mats
    p4 = p2 @ p2
    p8 = p4 @ p4
    expected = dpt.eye(4, dtype=p_mats.dtype)[dpt.newaxis, ...]
    assert dpt.all(p8 @ p4 == expected)


def check__extract_impl_validation(fn):
    x = dpt.ones(10)
    ind = dpt.ones(10, dtype="?")
    with pytest.raises(TypeError):
        fn(list(), ind)
    with pytest.raises(TypeError):
        fn(x, list())
    q2 = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
    ind2 = dpt.ones(10, dtype="?", sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        fn(x, ind2)
    with pytest.raises(ValueError):
        fn(x, ind, 1)


def check__nonzero_impl_validation(fn):
    with pytest.raises(TypeError):
        fn(list())


def check__take_multi_index(fn):
    x = dpt.ones(10)
    x_dev = x.sycl_device
    info_ = dpt.__array_namespace_info__()
    def_dtypes = info_.default_dtypes(device=x_dev)
    ind_dt = def_dtypes["indexing"]
    ind = dpt.arange(10, dtype=ind_dt)
    with pytest.raises(TypeError):
        fn(list(), tuple(), 1)
    with pytest.raises(ValueError):
        fn(x, (ind,), 0, mode=2)
    with pytest.raises(ValueError):
        fn(x, (None,), 1)
    with pytest.raises(IndexError):
        fn(x, (x,), 1)
    q2 = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
    ind2 = dpt.arange(10, dtype=ind_dt, sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        fn(x, (ind2,), 0)
    m = dpt.ones((10, 10))
    ind_1 = dpt.arange(10, dtype="i8")
    ind_2 = dpt.arange(10, dtype="u8")
    with pytest.raises(ValueError):
        fn(m, (ind_1, ind_2), 0)


def check__place_impl_validation(fn):
    with pytest.raises(TypeError):
        fn(list(), list(), list())
    x = dpt.ones(10)
    with pytest.raises(TypeError):
        fn(x, list(), list())
    q2 = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
    mask2 = dpt.ones(10, dtype="?", sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        fn(x, mask2, 1)
    x2 = dpt.ones((5, 5))
    mask2 = dpt.ones((5, 5), dtype="?")
    with pytest.raises(ValueError):
        fn(x2, mask2, x2, axis=1)


def check__put_multi_index_validation(fn):
    with pytest.raises(TypeError):
        fn(list(), list(), 0, list())
    x = dpt.ones(10)
    inds = dpt.arange(10, dtype="i8")
    vals = dpt.zeros(10)
    # test inds which is not a tuple/list
    fn(x, inds, 0, vals)
    x2 = dpt.ones((5, 5))
    ind1 = dpt.arange(5, dtype="i8")
    ind2 = dpt.arange(5, dtype="u8")
    with pytest.raises(ValueError):
        fn(x2, (ind1, ind2), 0, x2)
    with pytest.raises(TypeError):
        # invalid index type
        fn(x2, (ind1, list()), 0, x2)
    with pytest.raises(ValueError):
        # invalid mode keyword value
        fn(x, inds, 0, vals, mode=100)


def test__copy_utils():
    import dpctl_ext.tensor._copy_utils as cu

    get_queue_or_skip()

    check__extract_impl_validation(cu._extract_impl)
    check__nonzero_impl_validation(cu._nonzero_impl)
    check__take_multi_index(cu._take_multi_index)
    check__place_impl_validation(cu._place_impl)
    check__put_multi_index_validation(cu._put_multi_index)


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_take_indices_oob_py_ssize_t(mode):
    get_queue_or_skip()

    x = dpt.arange(10, dtype="i4")
    inds1 = dpt.full(5, dpt.iinfo(dpt.uint64).max, dtype=dpt.uint64)
    inds2 = dpt.full(5, dpt.iinfo(dpt.uint64).max, dtype=dpt.uint64)

    # sweep through a small range of indices
    # to check that OOB indices are well-behaved
    for i in range(1, 10):
        inds2 -= i
        r1 = dpt.take(x, inds1, mode=mode)
        r2 = dpt.take(x, inds2, mode=mode)

        assert dpt.all(r1 == r2)


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_put_indices_oob_py_ssize_t(mode):
    get_queue_or_skip()

    x = dpt.full(10, -1, dtype="i4")
    inds = dpt.full(1, dpt.iinfo(dpt.uint64).max, dtype=dpt.uint64)

    # OOB inds are positive, so always
    # clip to the top of range
    for i in range(1, 10):
        inds -= i
        dpt.put(x, inds, i, mode=mode)

        assert dpt.all(x[:-1] == -1)
        assert x[-1] == i


def test_take_along_axis_uint64_indices():
    get_queue_or_skip()

    inds = dpt.arange(1, 10, 2, dtype="u8")
    x = dpt.tile(dpt.asarray([0, -1], dtype="i4"), 5)
    res = dpt.take_along_axis(x, inds)
    assert dpt.all(res == -1)

    sh0 = 2
    inds = dpt.broadcast_to(inds, (sh0,) + inds.shape)
    x = dpt.broadcast_to(x, (sh0,) + x.shape)
    res = dpt.take_along_axis(x, inds, axis=1)
    assert dpt.all(res == -1)


def test_put_along_axis_uint64_indices():
    get_queue_or_skip()

    inds = dpt.arange(1, 10, 2, dtype="u8")
    x = dpt.zeros(10, dtype="i4")
    dpt.put_along_axis(x, inds, dpt.asarray(2, dtype=x.dtype))
    expected = dpt.tile(dpt.asarray([0, 2], dtype="i4"), 5)
    assert dpt.all(x == expected)

    sh0 = 2
    inds = dpt.broadcast_to(inds, (sh0,) + inds.shape)
    x = dpt.zeros((sh0,) + x.shape, dtype="i4")
    dpt.put_along_axis(x, inds, dpt.asarray(2, dtype=x.dtype), axis=1)
    expected = dpt.tile(dpt.asarray([0, 2], dtype="i4"), (2, 5))
    assert dpt.all(expected == x)


@pytest.mark.parametrize("data_dt", _all_dtypes)
@pytest.mark.parametrize("order", ["C", "F"])
def test_take_out(data_dt, order):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    axis = 0
    x = dpt.reshape(_make_3d(data_dt, q), (9, 3), order=order)
    ind = dpt.arange(2, dtype="i8", sycl_queue=q)
    out_sh = x.shape[:axis] + ind.shape + x.shape[axis + 1 :]
    out = dpt.empty(out_sh, dtype=data_dt, sycl_queue=q)

    expected = dpt.take(x, ind, axis=axis)

    dpt.take(x, ind, axis=axis, out=out)

    assert dpt.all(out == expected)


@pytest.mark.parametrize("data_dt", _all_dtypes)
@pytest.mark.parametrize("order", ["C", "F"])
def test_take_out_overlap(data_dt, order):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(data_dt, q)

    axis = 0
    x = dpt.reshape(_make_3d(data_dt, q), (9, 3), order=order)
    ind = dpt.arange(2, dtype="i8", sycl_queue=q)
    out = x[x.shape[axis] - ind.shape[axis] : x.shape[axis], :]

    expected = dpt.take(x, ind, axis=axis)

    dpt.take(x, ind, axis=axis, out=out)

    assert dpt.all(out == expected)
    assert dpt.all(x[x.shape[0] - ind.shape[0] : x.shape[0], :] == out)


def test_take_out_errors():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    x = dpt.arange(10, dtype="i4", sycl_queue=q1)
    ind = dpt.arange(2, dtype="i4", sycl_queue=q1)

    with pytest.raises(TypeError):
        dpt.take(x, ind, out=dict())

    out_read_only = dpt.empty(ind.shape, dtype=x.dtype, sycl_queue=q1)
    out_read_only.flags["W"] = False
    with pytest.raises(ValueError):
        dpt.take(x, ind, out=out_read_only)

    out_bad_shape = dpt.empty(0, dtype=x.dtype, sycl_queue=q1)
    with pytest.raises(ValueError):
        dpt.take(x, ind, out=out_bad_shape)

    out_bad_dt = dpt.empty(ind.shape, dtype="i8", sycl_queue=q1)
    with pytest.raises(ValueError):
        dpt.take(x, ind, out=out_bad_dt)

    out_bad_q = dpt.empty(ind.shape, dtype=x.dtype, sycl_queue=q2)
    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.take(x, ind, out=out_bad_q)


def test_getitem_impl_fn_invalid_inp():
    get_queue_or_skip()

    x = dpt.ones((10, 10), dtype="i4")

    bad_ind_type = (dpt.ones((), dtype="i4"), 2.0)
    with pytest.raises(TypeError):
        _take_multi_index(x, bad_ind_type, 0, 0)

    no_array_inds = (2, 3)
    with pytest.raises(TypeError):
        _take_multi_index(x, no_array_inds, 0, 0)
