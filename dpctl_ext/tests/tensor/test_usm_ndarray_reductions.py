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

from random import randrange

import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_allclose

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
from dpctl_ext.tensor._tensor_impl import default_device_index_type

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

_no_complex_dtypes = [
    "?",
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
]


_all_dtypes = _no_complex_dtypes + [
    "c8",
    "c16",
]


def test_max_min_axis():
    get_queue_or_skip()

    x = dpt.reshape(
        dpt.arange((3 * 4 * 5 * 6 * 7), dtype="i4"), (3, 4, 5, 6, 7)
    )

    m = dpt.max(x, axis=(1, 2, -1))
    assert m.shape == (3, 6)
    assert dpt.all(m == x[:, -1, -1, :, -1])

    m = dpt.min(x, axis=(1, 2, -1))
    assert m.shape == (3, 6)
    assert dpt.all(m == x[:, 0, 0, :, 0])


def test_max_axis1_axis0():
    """See gh-1455"""
    get_queue_or_skip()

    x = dpt.reshape(dpt.arange(3 * 4 * 5), (3, 4, 5))

    m = dpt.max(x, axis=0)
    assert dpt.all(m == x[-1, :, :])

    x = dpt.flip(x, axis=2)
    m = dpt.max(x, axis=2)
    assert dpt.all(m == x[:, :, 0])


def test_reduction_keepdims():
    get_queue_or_skip()

    n0, n1 = 3, 6
    x = dpt.ones((n0, 4, 5, n1, 7), dtype="i4")
    m = dpt.max(x, axis=(1, 2, -1), keepdims=True)

    xx = dpt.reshape(dpt.permute_dims(x, (0, 3, 1, 2, -1)), (n0, n1, -1))
    p = dpt.argmax(xx, axis=-1, keepdims=True)

    assert m.shape == (n0, 1, 1, n1, 1)
    assert dpt.all(m == dpt.reshape(x[:, 0, 0, :, 0], m.shape))
    assert dpt.all(p == 0)


def test_max_scalar():
    get_queue_or_skip()

    x = dpt.ones(())
    m = dpt.max(x)

    assert m.shape == ()
    assert x == m


@pytest.mark.parametrize("arg_dtype", ["i4", "f4", "c8"])
def test_reduction_kernels(arg_dtype):
    # i4 - always uses atomics w/ sycl group reduction
    # f4 - always uses atomics w/ custom group reduction
    # c8 - always uses temps w/ custom group reduction
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    x = dpt.ones((24, 1025), dtype=arg_dtype, sycl_queue=q)
    x[x.shape[0] // 2, :] = 3
    x[:, x.shape[1] // 2] = 3

    m = dpt.max(x)
    assert m == 3
    m = dpt.max(x, axis=0)
    assert dpt.all(m == 3)
    m = dpt.max(x, axis=1)
    assert dpt.all(m == 3)

    x = dpt.ones((24, 1025), dtype=arg_dtype, sycl_queue=q)
    x[x.shape[0] // 2, :] = 0
    x[:, x.shape[1] // 2] = 0

    m = dpt.min(x)
    assert m == 0
    m = dpt.min(x, axis=0)
    assert dpt.all(m == 0)
    m = dpt.min(x, axis=1)
    assert dpt.all(m == 0)


def test_max_min_nan_propagation():
    get_queue_or_skip()

    # float, finites
    x = dpt.arange(4, dtype="f4")
    x[0] = dpt.nan
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))

    # float, infinities
    x[1:] = dpt.inf
    assert dpt.isnan(dpt.max(x))
    x[1:] = -dpt.inf
    assert dpt.isnan(dpt.min(x))

    # complex
    x = dpt.arange(4, dtype="c8")
    x[0] = complex(dpt.nan, 0)
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))

    x[0] = complex(0, dpt.nan)
    assert dpt.isnan(dpt.max(x))
    assert dpt.isnan(dpt.min(x))


def test_argmax_scalar():
    get_queue_or_skip()

    x = dpt.ones(())
    m = dpt.argmax(x)

    assert m.shape == ()
    assert m == 0


@pytest.mark.parametrize("arg_dtype", ["i4", "f4", "c8"])
def test_search_reduction_kernels(arg_dtype):
    # i4 - always uses atomics w/ sycl group reduction
    # f4 - always uses atomics w/ custom group reduction
    # c8 - always uses temps w/ custom group reduction
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    x_shape = (24, 1024)
    x_size = np.prod(x_shape)
    x = dpt.ones(x_size, dtype=arg_dtype, sycl_queue=q)
    idx = randrange(x.size)
    idx_tup = np.unravel_index(idx, x_shape)
    x[idx] = 2

    m = dpt.argmax(x)
    assert m == idx

    # test case of strided input mapping to contig
    # implementation
    m = dpt.argmax(dpt.flip(x))
    assert m == x.size - 1 - idx

    # test case of strided implementation
    y = dpt.ones(2 * x.size, dtype=arg_dtype, sycl_queue=q)
    y[::2] = x
    m = dpt.argmax(y)
    assert m == 2 * idx

    x = dpt.reshape(x, x_shape)

    x[idx_tup[0], :] = 3
    m = dpt.argmax(x, axis=0)
    assert dpt.all(m == idx_tup[0])
    x[:, idx_tup[1]] = 4
    m = dpt.argmax(x, axis=1)
    assert dpt.all(m == idx_tup[1])

    x = x[:, ::-2]
    idx = randrange(x.shape[1])
    x[:, idx] = 5
    m = dpt.argmax(x, axis=1)
    assert dpt.all(m == idx)

    x = dpt.ones(x_size, dtype=arg_dtype, sycl_queue=q)
    idx = randrange(x.size)
    idx_tup = np.unravel_index(idx, x_shape)
    x[idx] = 0

    m = dpt.argmin(x)
    assert m == idx

    x = dpt.reshape(x, x_shape)

    x[idx_tup[0], :] = -1
    m = dpt.argmin(x, axis=0)
    assert dpt.all(m == idx_tup[0])
    x[:, idx_tup[1]] = -2
    m = dpt.argmin(x, axis=1)
    assert dpt.all(m == idx_tup[1])

    x = x[:, ::-2]
    idx = randrange(x.shape[1])
    x[:, idx] = -3
    m = dpt.argmin(x, axis=1)
    assert dpt.all(m == idx)


def test_argmax_argmin_nan_propagation():
    get_queue_or_skip()

    sz = 4
    idx = randrange(sz)
    # floats
    x = dpt.arange(sz, dtype="f4")
    x[idx] = dpt.nan
    assert dpt.argmax(x) == idx
    assert dpt.argmin(x) == idx

    # complex
    x = dpt.arange(sz, dtype="c8")
    x[idx] = complex(dpt.nan, 0)
    assert dpt.argmax(x) == idx
    assert dpt.argmin(x) == idx

    x[idx] = complex(0, dpt.nan)
    assert dpt.argmax(x) == idx
    assert dpt.argmin(x) == idx


def test_argmax_argmin_identities():
    # make sure that identity arrays work as expected
    get_queue_or_skip()

    x = dpt.full(3, dpt.iinfo(dpt.int32).min, dtype="i4")
    assert dpt.argmax(x) == 0
    x = dpt.full(3, dpt.iinfo(dpt.int32).max, dtype="i4")
    assert dpt.argmin(x) == 0


@pytest.mark.parametrize("order", ["C", "F"])
def test_argmax_axis0_axis1(order):
    get_queue_or_skip()

    x = dpt.asarray([[1, 2, 3], [6, 5, 4]], dtype="i4", order=order)
    assert dpt.argmax(x) == 3

    res = dpt.argmax(x, axis=0)
    expected = dpt.asarray([1, 1, 1], dtype=res.dtype)
    assert dpt.all(res == expected)

    res = dpt.argmax(x, axis=1)
    expected = dpt.asarray([2, 0], dtype=res.dtype)
    assert dpt.all(res == expected)


def test_reduction_arg_validation():
    get_queue_or_skip()

    x = {}
    with pytest.raises(TypeError):
        dpt.sum(x)
    with pytest.raises(TypeError):
        dpt.max(x)
    with pytest.raises(TypeError):
        dpt.argmax(x)

    x = dpt.zeros((0,), dtype="i4")
    with pytest.raises(ValueError):
        dpt.max(x)
    with pytest.raises(ValueError):
        dpt.argmax(x)


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
def test_logsumexp_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.logsumexp(m)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype.kind == "f"
    tol = dpt.finfo(r.dtype).resolution
    assert_allclose(
        dpt.asnumpy(r),
        np.logaddexp.reduce(dpt.asnumpy(m), dtype=r.dtype),
        rtol=tol,
        atol=tol,
    )


def test_logsumexp_empty():
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="f4")
    y = dpt.logsumexp(x)
    assert y.shape == ()
    assert y == -dpt.inf


def test_logsumexp_axis():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="f4")
    s = dpt.logsumexp(m, axis=(1, 2, -1))

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 6)
    tol = dpt.finfo(s.dtype).resolution
    assert_allclose(
        dpt.asnumpy(s),
        np.logaddexp.reduce(dpt.asnumpy(m), axis=(1, 2, -1), dtype=s.dtype),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_logsumexp_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.logsumexp(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)


def test_logsumexp_keepdims():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.logsumexp(m, axis=(1, 2, -1), keepdims=True)

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 1, 1, 6, 1)


def test_logsumexp_keepdims_zero_size():
    get_queue_or_skip()
    n = 10
    a = dpt.ones((n, 0, n))

    s1 = dpt.logsumexp(a, keepdims=True)
    assert s1.shape == (1, 1, 1)

    s2 = dpt.logsumexp(a, axis=(0, 1), keepdims=True)
    assert s2.shape == (1, 1, n)

    s3 = dpt.logsumexp(a, axis=(1, 2), keepdims=True)
    assert s3.shape == (n, 1, 1)

    s4 = dpt.logsumexp(a, axis=(0, 2), keepdims=True)
    assert s4.shape == (1, 0, 1)

    a0 = a[0]
    s5 = dpt.logsumexp(a0, keepdims=True)
    assert s5.shape == (1, 1)


def test_logsumexp_scalar():
    get_queue_or_skip()

    m = dpt.ones(())
    s = dpt.logsumexp(m)

    assert isinstance(s, dpt.usm_ndarray)
    assert m.sycl_queue == s.sycl_queue
    assert s.shape == ()


def test_logsumexp_complex():
    get_queue_or_skip()

    x = dpt.zeros(1, dtype="c8")
    with pytest.raises(ValueError):
        dpt.logsumexp(x)


def test_logsumexp_int_axis():
    get_queue_or_skip()

    x = dpt.zeros((8, 10), dtype="f4")
    res = dpt.logsumexp(x, axis=0)
    assert res.ndim == 1
    assert res.shape[0] == 10


def test_logsumexp_invalid_arr():
    x = {}
    with pytest.raises(TypeError):
        dpt.logsumexp(x)


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
def test_hypot_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.reduce_hypot(m)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype.kind == "f"
    tol = dpt.finfo(r.dtype).resolution
    assert_allclose(
        dpt.asnumpy(r),
        np.hypot.reduce(dpt.asnumpy(m), dtype=r.dtype),
        rtol=tol,
        atol=tol,
    )


def test_hypot_empty():
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="f4")
    y = dpt.reduce_hypot(x)
    assert y.shape == ()
    assert y == 0


@pytest.mark.parametrize("arg_dtype", _no_complex_dtypes[1:])
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_hypot_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.reduce_hypot(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)


def test_hypot_complex():
    get_queue_or_skip()

    x = dpt.zeros(1, dtype="c8")
    with pytest.raises(ValueError):
        dpt.reduce_hypot(x)


def test_tree_reduction_axis1_axis0():
    """See gh-1455"""
    get_queue_or_skip()

    x = dpt.reshape(dpt.arange(3 * 4 * 5, dtype="f4"), (3, 4, 5))

    m = dpt.logsumexp(x, axis=0)
    tol = dpt.finfo(m.dtype).resolution
    assert_allclose(
        dpt.asnumpy(m),
        np.logaddexp.reduce(dpt.asnumpy(x), axis=0, dtype=m.dtype),
        rtol=tol,
        atol=tol,
    )

    x = dpt.flip(x, axis=2)
    m = dpt.logsumexp(x, axis=2)
    assert_allclose(
        dpt.asnumpy(m),
        np.logaddexp.reduce(dpt.asnumpy(x), axis=2, dtype=m.dtype),
        rtol=tol,
        atol=tol,
    )


def test_numeric_reduction_out_kwarg():
    get_queue_or_skip()

    n1, n2, n3 = 3, 4, 5
    x = dpt.ones((n1, n2, n3), dtype="i8")
    out = dpt.zeros((2 * n1, 3 * n2), dtype="i8")
    res = dpt.sum(x, axis=-1, out=out[::-2, 1::3])
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == res)
    assert dpt.all(out[::-2, 1::3] == 5)

    out = dpt.zeros((2 * n1, 3 * n2, 1), dtype="i8")
    res = dpt.sum(x, axis=-1, keepdims=True, out=out[::-2, 1::3])
    assert res.shape == (n1, n2, 1)
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == res)
    assert dpt.all(out[::-2, 1::3] == 5)

    res = dpt.sum(x, axis=0, out=x[-1])
    assert dpt.all(x[-1] == res)
    assert dpt.all(x[-1] == 3)
    assert dpt.all(x[0:-1] == 1)

    # test no-op case
    x = dpt.ones((n1, n2, n3), dtype="i8")
    out = dpt.zeros((2 * n1, 3 * n2, n3), dtype="i8")
    res = dpt.sum(x, axis=(), out=out[::-2, 1::3])
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == x)

    # test with dtype kwarg
    x = dpt.ones((n1, n2, n3), dtype="i4")
    out = dpt.zeros((2 * n1, 3 * n2), dtype="f4")
    res = dpt.sum(x, axis=-1, dtype="f4", out=out[::-2, 1::3])
    zero_res = dpt.zeros_like(res)
    assert dpt.allclose(out[::-2, 0::3], zero_res)
    assert dpt.allclose(out[::-2, 2::3], zero_res)
    assert dpt.allclose(out[::-2, 1::3], res)
    assert dpt.allclose(out[::-2, 1::3], dpt.full_like(res, 5, dtype="f4"))


def test_comparison_reduction_out_kwarg():
    get_queue_or_skip()

    n1, n2, n3 = 3, 4, 5
    x = dpt.reshape(dpt.arange(n1 * n2 * n3, dtype="i4"), (n1, n2, n3))
    out = dpt.zeros((2 * n1, 3 * n2), dtype="i4")
    res = dpt.max(x, axis=-1, out=out[::-2, 1::3])
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == res)
    assert dpt.all(out[::-2, 1::3] == x[:, :, -1])

    out = dpt.zeros((2 * n1, 3 * n2, 1), dtype="i4")
    res = dpt.max(x, axis=-1, keepdims=True, out=out[::-2, 1::3])
    assert res.shape == (n1, n2, 1)
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == res)
    assert dpt.all(out[::-2, 1::3] == x[:, :, -1, dpt.newaxis])

    # test no-op case
    out = dpt.zeros((2 * n1, 3 * n2, n3), dtype="i4")
    res = dpt.max(x, axis=(), out=out[::-2, 1::3])
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == x)

    # test overlap
    res = dpt.max(x, axis=0, out=x[0])
    assert dpt.all(x[0] == res)
    assert dpt.all(x[0] == x[-1])


def test_search_reduction_out_kwarg():
    get_queue_or_skip()

    n1, n2, n3 = 3, 4, 5
    dt = dpt.__array_namespace_info__().default_dtypes()["indexing"]

    x = dpt.reshape(dpt.arange(n1 * n2 * n3, dtype=dt), (n1, n2, n3))
    out = dpt.zeros((2 * n1, 3 * n2), dtype=dt)
    res = dpt.argmax(x, axis=-1, out=out[::-2, 1::3])
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == res)
    assert dpt.all(out[::-2, 1::3] == n2)

    out = dpt.zeros((2 * n1, 3 * n2, 1), dtype=dt)
    res = dpt.argmax(x, axis=-1, keepdims=True, out=out[::-2, 1::3])
    assert res.shape == (n1, n2, 1)
    assert dpt.all(out[::-2, 0::3] == 0)
    assert dpt.all(out[::-2, 2::3] == 0)
    assert dpt.all(out[::-2, 1::3] == res)
    assert dpt.all(out[::-2, 1::3] == n3 - 1)

    # test no-op case
    x = dpt.ones((), dtype=dt)
    out = dpt.ones(2, dtype=dt)
    res = dpt.argmax(x, axis=None, out=out[1])
    assert dpt.all(out[0] == 1)
    assert dpt.all(out[1] == 0)

    # test overlap
    x = dpt.reshape(dpt.arange(n1 * n2, dtype=dt), (n1, n2))
    res = dpt.argmax(x, axis=0, out=x[0])
    assert dpt.all(x[0] == res)
    assert dpt.all(x[0] == n1 - 1)


def test_reduction_out_kwarg_arg_validation():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    ind_dt = dpt.__array_namespace_info__().default_dtypes()["indexing"]

    x = dpt.ones(10, dtype="f4")
    out_wrong_queue = dpt.empty((), dtype="f4", sycl_queue=q2)
    out_wrong_dtype = dpt.empty((), dtype="i4", sycl_queue=q1)
    out_wrong_shape = dpt.empty(1, dtype="f4", sycl_queue=q1)
    out_wrong_keepdims = dpt.empty((), dtype="f4", sycl_queue=q1)
    out_not_writable = dpt.empty((), dtype="f4", sycl_queue=q1)
    out_not_writable.flags["W"] = False

    with pytest.raises(TypeError):
        dpt.sum(x, out=dict())
    with pytest.raises(TypeError):
        dpt.max(x, out=dict())
    with pytest.raises(TypeError):
        dpt.argmax(x, out=dict())
    with pytest.raises(ExecutionPlacementError):
        dpt.sum(x, out=out_wrong_queue)
    with pytest.raises(ExecutionPlacementError):
        dpt.max(x, out=out_wrong_queue)
    with pytest.raises(ExecutionPlacementError):
        dpt.argmax(x, out=dpt.empty_like(out_wrong_queue, dtype=ind_dt))
    with pytest.raises(ValueError):
        dpt.sum(x, out=out_wrong_dtype)
    with pytest.raises(ValueError):
        dpt.max(x, out=out_wrong_dtype)
    with pytest.raises(ValueError):
        dpt.argmax(x, out=dpt.empty_like(out_wrong_dtype, dtype="f4"))
    with pytest.raises(ValueError):
        dpt.sum(x, out=out_wrong_shape)
    with pytest.raises(ValueError):
        dpt.max(x, out=out_wrong_shape)
    with pytest.raises(ValueError):
        dpt.argmax(x, out=dpt.empty_like(out_wrong_shape, dtype=ind_dt))
    with pytest.raises(ValueError):
        dpt.sum(x, out=out_not_writable)
    with pytest.raises(ValueError):
        dpt.max(x, out=out_not_writable)
    with pytest.raises(ValueError):
        search_not_writable = dpt.empty_like(out_not_writable, dtype=ind_dt)
        search_not_writable.flags["W"] = False
        dpt.argmax(x, out=search_not_writable)
    with pytest.raises(ValueError):
        dpt.sum(x, keepdims=True, out=out_wrong_keepdims)
    with pytest.raises(ValueError):
        dpt.max(x, keepdims=True, out=out_wrong_keepdims)
    with pytest.raises(ValueError):
        dpt.argmax(
            x,
            keepdims=True,
            out=dpt.empty_like(out_wrong_keepdims, dtype=ind_dt),
        )


@pytest.mark.parametrize("dt", _all_dtypes)
def test_count_nonzero(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    expected_dt = default_device_index_type(q.sycl_device)

    x = dpt.ones(10, dtype=dt, sycl_queue=q)
    res = dpt.count_nonzero(x)
    assert res == 10
    assert res.dtype == expected_dt

    x[3:6] = 0
    res = dpt.count_nonzero(x)
    assert res == 7
    assert res.dtype == expected_dt
