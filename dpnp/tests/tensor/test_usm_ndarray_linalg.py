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

import itertools

import dpctl
import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError

import dpnp.tensor as dpt

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

_numeric_types = [
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


def _map_int_to_type(n, dt):
    assert isinstance(n, int)
    assert n > 0
    if dt == dpt.int8:
        return ((n + 128) % 256) - 128
    elif dt == dpt.uint8:
        return n % 256
    elif dt == dpt.int16:
        return ((n + 32768) % 65536) - 32768
    elif dt == dpt.uint16:
        return n % 65536
    return n


def test_matrix_transpose():
    get_queue_or_skip()

    X = dpt.reshape(dpt.arange(2 * 3, dtype="i4"), (2, 3))
    res = dpt.matrix_transpose(X)
    expected_res = X.mT

    assert expected_res.shape == res.shape
    assert expected_res.flags["C"] == res.flags["C"]
    assert expected_res.flags["F"] == res.flags["F"]
    assert dpt.all(X.mT == res)


def test_matrix_transpose_arg_validation():
    get_queue_or_skip()

    X = dpt.empty(5, dtype="i4")
    with pytest.raises(ValueError):
        dpt.matrix_transpose(X)

    X = {}
    with pytest.raises(TypeError):
        dpt.matrix_transpose(X)

    X = dpt.empty((5, 5), dtype="i4")
    assert isinstance(dpt.matrix_transpose(X), dpt.usm_ndarray)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_simple(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, m = 235, 17
    m1 = dpt.zeros((m, n), dtype=dtype)
    m2 = dpt.zeros((n, m), dtype=dtype)

    dt = m1.dtype
    if dt.kind in "ui":
        n1 = min(n, dpt.iinfo(dt).max)
    else:
        n1 = n
    m1[:, :n1] = dpt.ones((m, n1), dtype=dt)
    m2[:n1, :] = dpt.ones((n1, m), dtype=dt)

    for k in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17]:
        r = dpt.matmul(m1[:k, :], m2[:, :k])
        assert dpt.all(r == dpt.full((k, k), fill_value=n1, dtype=dt))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_nilpotent1(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 77
    N_mat = dpt.eye(n, k=1, dtype=dtype)
    I_mat = dpt.eye(n, dtype=dtype)
    R_mat = dpt.eye(n, dtype=dtype)
    for _ in range(n + 1):
        R_mat = I_mat + dpt.matmul(N_mat, R_mat)

    assert dpt.allclose(dpt.matmul(I_mat - N_mat, R_mat), I_mat)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_nilpotent2(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 128
    u = dpt.ones((n, 1), dtype=dtype)
    v = dpt.ones((1, n), dtype=dtype)

    uv = dpt.matmul(u, v)
    uv_ref = u * v

    assert dpt.allclose(uv, uv_ref)


def test_matmul_null_axis():
    get_queue_or_skip()
    n = 3

    A_mat = dpt.ones((n, 0), dtype="f4")
    B_mat = dpt.ones((0, 1), dtype="f4")

    R_mat = dpt.matmul(A_mat, B_mat)
    assert R_mat.shape == (n, 1)

    R_mat = dpt.matmul(A_mat, B_mat[:, :0])
    assert R_mat.shape == (n, 0)


@pytest.mark.parametrize("dtype", ["i4", "f4"])
def test_matmul_dims(dtype):
    get_queue_or_skip()

    n, m, k, b = 4, 5, 7, 3
    v = dpt.ones(k, dtype=dtype)
    m1 = dpt.ones((n, k), dtype=dtype)
    m2 = dpt.ones((k, m), dtype=dtype)
    st1 = dpt.ones((b, n, k), dtype=dtype)
    st2 = dpt.ones((b, k, m), dtype=dtype)

    r = dpt.matmul(v, v)
    assert r.shape == ()
    assert dpt.round(r) == k

    r = dpt.matmul(m1, v)
    assert r.shape == (n,)
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(v, m2)
    assert r.shape == (m,)
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(m1, m2)
    assert r.shape == (
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(v, st2)
    assert r.shape == (
        b,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(st1, v)
    assert r.shape == (
        b,
        n,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(st1, m2)
    assert r.shape == (
        b,
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(m1, st2)
    assert r.shape == (
        b,
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)

    r = dpt.matmul(st1, st2)
    assert r.shape == (
        b,
        n,
        m,
    )
    assert dpt.all(dpt.round(r) == k)


def test_matmul_arg_validation():
    get_queue_or_skip()

    s1, s2 = dpt.ones(tuple()), dpt.zeros(tuple())
    v1, v2 = dpt.ones(16), dpt.zeros(16)

    with pytest.raises(ValueError):
        dpt.matmul(s1, v2)

    with pytest.raises(ValueError):
        dpt.matmul(v1, s2)

    with pytest.raises(TypeError):
        dpt.matmul(dict(), v2)

    with pytest.raises(TypeError):
        dpt.matmul(v2, None)


def test_matmul_dims_validation():
    get_queue_or_skip()

    m1 = dpt.ones((16, 16))
    m2 = dpt.ones((16, 16))

    # contraction dimensions mismatch
    with pytest.raises(ValueError):
        dpt.matmul(m1[:, :7], m2[:3, :])

    m1 = dpt.ones((3, 4, 5))
    m2 = dpt.ones((2, 5, 3))
    # broadcasting dimensions mismatch
    with pytest.raises(ValueError):
        dpt.matmul(m1, m2)


def test_matmul_broadcasting():
    get_queue_or_skip()

    for dt1, dt2 in [
        (dpt.int16, dpt.int32),
        (dpt.float32, dpt.int16),
        (dpt.int32, dpt.uint32),
    ]:
        m1 = dpt.ones((7, 11, 16), dtype=dt1)
        m2 = dpt.ones((16, 13), dtype=dt2)

        r = dpt.matmul(m1, m2[dpt.newaxis, ...])

        assert r.shape == (7, 11, 13)


@pytest.mark.parametrize("dtype", ["i4", "i8", "f4", "c8"])
def test_matmul_strided(dtype):
    get_queue_or_skip()

    m1_shape = (14, 22, 32)
    m1_size = 1
    for el in m1_shape:
        m1_size = m1_size * el

    m1 = dpt.remainder(dpt.arange(1, m1_size + 1, dtype="i8"), 13)
    m1_orig = dpt.reshape(dpt.astype(m1, dtype), m1_shape)
    m2_orig = dpt.ones((14, 16, 13), dtype=dtype)

    m1 = m1_orig[::2, ::-2, ::2]
    m2 = m2_orig[::2, :, :]
    r = dpt.matmul(m1, m2)

    assert r.shape == m1.shape[:2] + m2.shape[-1:]
    ref = np.matmul(dpt.asnumpy(m1), dpt.asnumpy(m2))
    assert np.allclose(dpt.asnumpy(r), ref)

    m1 = m1_orig[::2, ::2, ::-2]
    m2 = m2_orig[::2, :, :]
    r = dpt.matmul(m1, m2)

    assert r.shape == m1.shape[:2] + m2.shape[-1:]
    ref = np.matmul(dpt.asnumpy(m1), dpt.asnumpy(m2))
    assert np.allclose(dpt.asnumpy(r), ref)

    m1 = m1_orig[::-2, ::2, ::2]
    m2 = m2_orig[::-2, :, :]
    r = dpt.matmul(m1, m2)

    assert r.shape == m1.shape[:2] + m2.shape[-1:]
    ref = np.matmul(dpt.asnumpy(m1), dpt.asnumpy(m2))
    assert np.allclose(dpt.asnumpy(r), ref)


def test_matmul_out():
    get_queue_or_skip()

    m1 = (
        dpt.arange(14, dtype="f4")[:, dpt.newaxis, dpt.newaxis]
        + dpt.arange(17, dtype="f4")[dpt.newaxis, :, dpt.newaxis]
        + dpt.arange(128, dtype="f4")[dpt.newaxis, dpt.newaxis, :]
    )
    assert m1.shape == (14, 17, 128)
    m2 = dpt.tile(
        dpt.reshape(dpt.asarray([1, 2], dtype="f4"), (2, 1, 1)), (7, 128, 13)
    )
    assert m2.shape == (14, 128, 13)

    buf = dpt.zeros((2 * 14, 3 * 17, 13), dtype="f4")
    res = dpt.matmul(m1, m2, out=buf[::-2, 1::3, :])

    assert dpt.allclose(res, buf[::-2, 1::3, :])
    assert dpt.allclose(dpt.zeros_like(res), buf[::-2, 0::3, :])
    assert dpt.allclose(dpt.zeros_like(res), buf[::-2, 2::3, :])

    m1_np = dpt.asnumpy(m1)
    ref = np.matmul(m1_np, dpt.asnumpy(m2))
    assert np.allclose(ref, dpt.asnumpy(res))

    res = dpt.matmul(m1[:, :10, :10], m1[:, :10, :10].mT, out=m1[:, :10, :10])
    ref = np.matmul(
        m1_np[:, :10, :10], np.transpose(m1_np[:, :10, :10], (0, 2, 1))
    )
    assert np.allclose(ref, dpt.asnumpy(res))


def test_matmul_readonly_out():
    get_queue_or_skip()
    m = dpt.ones((10, 10), dtype=dpt.int32)
    r = dpt.empty_like(m)
    r.flags["W"] = False

    with pytest.raises(ValueError):
        dpt.matmul(m, m, out=r)


def test_matmul_dtype():
    get_queue_or_skip()

    for dt1, dt2 in [
        (dpt.int32, dpt.int16),
        (dpt.int16, dpt.int32),
        (dpt.float32, dpt.int16),
        (dpt.int32, dpt.float32),
    ]:
        m1 = dpt.ones((10, 10), dtype=dt1)
        m2 = dpt.ones((10, 10), dtype=dt2)

        for ord in ["C", "A", "F", "K"]:
            r = dpt.matmul(m1, m2, dtype=dpt.float32, order=ord)
            assert r.dtype == dpt.float32


@pytest.mark.parametrize("dt1", _numeric_types)
@pytest.mark.parametrize("dt2", _numeric_types)
@pytest.mark.parametrize("order", ["C", "K"])
def test_matmul_type_promotion(dt1, dt2, order):
    get_queue_or_skip()

    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    b, n, k, m = 8, 10, 17, 10
    m1 = dpt.ones((1, n, k), dtype=dt1)
    m2 = dpt.ones((b, k, m), dtype=dt2)
    expected_dt = dpt.result_type(m1, m2)

    r = dpt.matmul(m1, m2, order=order)
    assert r.shape == (b, n, m)
    assert r.dtype == expected_dt

    m1 = dpt.ones((b, n, k), dtype=dt1)
    m2 = dpt.ones((1, k, m), dtype=dt2)

    r = dpt.matmul(m1, m2, order=order)
    assert r.shape == (b, n, m)
    assert r.dtype == expected_dt

    m1 = dpt.ones((n, k), dtype=dt1)
    m2 = dpt.ones((k, m), dtype=dt2)

    r = dpt.matmul(m1, m2, order=order)
    assert r.shape == (n, m)
    assert r.dtype == expected_dt


def test_matmul_invalid_dtype():
    get_queue_or_skip()

    m1 = dpt.zeros((10, 10), dtype="f4")
    m2 = dpt.zeros((10, 10), dtype="f4")
    m3 = dpt.zeros((10, 10), dtype="i4")

    with pytest.raises(ValueError):
        dpt.matmul(m1, m2, dtype="i4")

    with pytest.raises(ValueError):
        dpt.matmul(m1, m3, dtype="i4")

    with pytest.raises(ValueError):
        dpt.matmul(m3, m1, dtype="i4")


def test_matmul_out_errors():
    q1 = get_queue_or_skip()
    q2 = dpctl.SyclQueue()

    sh = (10, 10)
    dt = "i4"
    m1 = dpt.zeros(sh, dtype=dt, sycl_queue=q1)
    m2 = dpt.zeros(sh, dtype=dt, sycl_queue=q1)

    with pytest.raises(TypeError):
        dpt.matmul(m1, m2, out=dict())

    with pytest.raises(ValueError):
        dpt.matmul(m1, m2, out=dpt.empty((10,), dtype=dt, sycl_queue=q1))

    with pytest.raises(ValueError):
        dpt.matmul(m1, m2, out=dpt.empty(sh, dtype="f4", sycl_queue=q1))

    with pytest.raises(ExecutionPlacementError):
        dpt.matmul(m1, m2, out=dpt.empty(sh, dtype=dt, sycl_queue=q2))


def test_matmul_order():
    get_queue_or_skip()

    sh = (
        10,
        10,
    )
    sh2 = tuple(2 * dim for dim in sh)
    n = sh[-1]

    for dt1, dt2 in zip(["i4", "i4", "f4"], ["i4", "f4", "i4"]):
        ar1 = dpt.ones(sh, dtype=dt1, order="C")
        ar2 = dpt.ones(sh, dtype=dt2, order="C")
        r1 = dpt.matmul(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.matmul(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.matmul(ar1, ar2, order="A")
        assert r3.flags.c_contiguous
        r4 = dpt.matmul(ar1, ar2, order="K")
        assert r4.flags.c_contiguous

        ar1 = dpt.ones(sh, dtype=dt1, order="F")
        ar2 = dpt.ones(sh, dtype=dt2, order="F")
        r1 = dpt.matmul(ar1, ar2, order="C")
        assert r1.flags.c_contiguous
        r2 = dpt.matmul(ar1, ar2, order="F")
        assert r2.flags.f_contiguous
        r3 = dpt.matmul(ar1, ar2, order="A")
        assert r3.flags.f_contiguous
        r4 = dpt.matmul(ar1, ar2, order="K")
        assert r4.flags.f_contiguous

        ar1 = dpt.ones(sh2, dtype=dt1, order="C")[:10, ::-2]
        ar2 = dpt.ones(sh2, dtype=dt2, order="C")[:10, ::-2]
        r4 = dpt.matmul(ar1, ar2, order="K")
        assert r4.strides == (n, -1)
        r5 = dpt.matmul(ar1, ar2, order="C")
        assert r5.strides == (n, 1)

        ar1 = dpt.ones(sh2, dtype=dt1, order="C")[:10, ::-2].mT
        ar2 = dpt.ones(sh2, dtype=dt2, order="C")[:10, ::-2].mT
        r4 = dpt.matmul(ar1, ar2, order="K")
        assert r4.strides == (-1, n)
        r5 = dpt.matmul(ar1, ar2, order="C")
        assert r5.strides == (n, 1)


def test_matmul_invalid_order():
    get_queue_or_skip()

    sh = (
        10,
        10,
    )
    dt = "i4"

    ar1 = dpt.ones(sh, dtype=dt, order="C")
    ar2 = dpt.ones(sh, dtype=dt, order="C")
    r = dpt.matmul(ar1, ar2, order="invalid")
    assert r.flags.c_contiguous

    ar1 = dpt.ones(sh, dtype=dt, order="F")
    ar2 = dpt.ones(sh, dtype=dt, order="F")
    r = dpt.matmul(ar1, ar2, order="invalid")
    assert r.flags.f_contiguous


def test_matmul_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = dpctl.SyclQueue()

    sh = (
        10,
        10,
    )
    dt = "i4"
    m1 = dpt.zeros(sh, dtype=dt, sycl_queue=q1)
    m2 = dpt.zeros(sh, dtype=dt, sycl_queue=q2)

    with pytest.raises(ExecutionPlacementError):
        dpt.matmul(m1, m2)


def test_matmul_inplace_broadcasting():
    get_queue_or_skip()

    sh = (3, 5, 5)
    dt = "i4"

    m1 = dpt.ones((3, 5, 5), dtype=dt)
    m2 = dpt.ones((1, 5, 5), dtype=dt)
    m1 @= m2
    assert dpt.all(m1 == dpt.full(sh, 5, dtype=dt))


def test_matmul_prepend_dims():
    get_queue_or_skip()

    n = 5
    for dt1, dt2 in [
        (dpt.int32, dpt.int32),
        (dpt.int32, dpt.int64),
        (dpt.int64, dpt.int32),
        (dpt.int32, dpt.uint32),
    ]:
        m = dpt.ones((n, 4), dtype=dt1)
        v = dpt.ones((4,), dtype=dt2)
        r = dpt.matmul(m, v)
        assert r.shape == (n,)

        r = dpt.matmul(v, m.mT)
        assert r.shape == (n,)


def test_matmul_inplace_same_tensors():
    get_queue_or_skip()

    n = 5
    sh = (
        n,
        n,
    )

    ar1 = dpt.ones(sh, dtype="i4")
    ar1 @= ar1
    assert dpt.all(ar1 == dpt.full(sh, n, dtype="i4"))

    ar1 = dpt.ones(sh, dtype="i8")
    ar2 = dpt.ones(sh, dtype="i4")
    dpt.matmul(ar1, ar2, out=ar1)
    assert dpt.all(ar1 == dpt.full(sh, n, dtype=ar1.dtype))

    ar1 = dpt.ones(sh, dtype="i4")
    ar2 = dpt.ones(sh, dtype="i8")
    dpt.matmul(ar1, ar2, out=ar2)
    assert dpt.all(ar2 == dpt.full(sh, n, dtype=ar2.dtype))


@pytest.fixture
def random_matrix():
    rs = np.random.RandomState(seed=123456)
    m_np = rs.randint(low=0, high=6, size=(400, 400))
    return m_np


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_largish_square(dtype, random_matrix):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m_np = random_matrix.astype(dtype)
    x_np = np.matmul(m_np.T, m_np)

    m = dpt.asarray(m_np)
    mT = dpt.asarray(m.mT, copy=True, order="C")
    x1 = dpt.matmul(m.mT, m)
    x2 = dpt.matmul(mT, m)

    tol = 0
    if dpt.isdtype(x2.dtype, ("real floating", "complex floating")):
        tol = 32 * dpt.finfo(x2.dtype).eps

    assert dpt.allclose(x1, x2, atol=tol, rtol=tol)
    assert dpt.allclose(x1, dpt.asarray(x_np), atol=tol, rtol=tol)

    # check stided input
    m_np = m_np[:-1, :-1]
    x_np = np.matmul(m_np.T, m_np)

    m = m[:-1, :-1]
    mT = dpt.asarray(m.mT, copy=True, order="C")
    x1 = dpt.matmul(m.mT, m)
    x2 = dpt.matmul(mT, m)

    assert dpt.allclose(x1, x2, atol=tol, rtol=tol)
    assert dpt.allclose(x1, dpt.asarray(x_np), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_matmul_largish_rect(dtype, random_matrix):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m_np = random_matrix.astype(dtype)[:, :-1]
    x_np = np.matmul(m_np.T[:-2, :], m_np)

    m = dpt.asarray(m_np)
    mmT = m.mT[:-2, :]
    mT = dpt.asarray(mmT, copy=True, order="C")
    x1 = dpt.matmul(mmT, m)
    x2 = dpt.matmul(mT, m)

    tol = 0
    if dpt.isdtype(x2.dtype, ("real floating", "complex floating")):
        tol = 32 * dpt.finfo(x2.dtype).eps

    assert dpt.allclose(x1, x2, atol=tol, rtol=tol)
    assert dpt.allclose(x1, dpt.asarray(x_np), atol=tol, rtol=tol)

    m_np = m_np[:-1, :-1]
    x_np = np.matmul(m_np.T[:-2, :], m_np)

    m = m[:-1, :-1]
    mmT = m.mT[:-2, :]
    mT = dpt.asarray(mmT, copy=True, order="C")
    x1 = dpt.matmul(mmT, m)
    x2 = dpt.matmul(mT, m)

    assert dpt.allclose(x1, x2, atol=tol, rtol=tol)
    assert dpt.allclose(x1, dpt.asarray(x_np), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_tensordot_outer(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    t1 = dpt.ones((3, 8), dtype=dtype)
    t2 = dpt.ones((4, 12), dtype=dtype)

    r = dpt.tensordot(t1, t2, axes=0)
    assert r.shape == t1.shape + t2.shape
    assert dpt.allclose(r, dpt.ones_like(r))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_tensordot_inner(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    t1 = dpt.ones((3, 8), dtype=dtype)
    t2 = dpt.ones((4, 8), dtype=dtype)

    r = dpt.tensordot(t1, t2.mT, axes=1)
    assert r.shape == t1.shape[:1] + t2.shape[:1]
    assert dpt.allclose(r, dpt.full_like(r, fill_value=t1.shape[1]))


@pytest.mark.parametrize("dtype", _numeric_types)
def test_tensordot_double(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    t1 = dpt.ones((2, 4, 8), dtype=dtype)
    t2 = dpt.ones((3, 4, 8), dtype=dtype)

    r = dpt.tensordot(t1, dpt.permute_dims(t2, (1, 2, 0)), axes=2)
    assert r.shape == t1.shape[:1] + t2.shape[:1]
    expected = dpt.prod(dpt.asarray(t1.shape[1:]))
    assert dpt.allclose(r, dpt.full_like(r, fill_value=expected))


@pytest.mark.parametrize("dtype", ["i4", "f4"])
def test_tensordot_axes_sequence(dtype):
    get_queue_or_skip()

    r = 4
    t1 = dpt.ones((2, 2, 4, 3), dtype=dtype)
    t2 = dpt.ones((3, 2, 4, 3), dtype=dtype)

    assert len(t1.shape) == r
    assert len(t2.shape) == r

    expected = dpt.prod(dpt.asarray(t1.shape[1:]))
    ps1 = itertools.permutations(range(r))
    ps2 = itertools.permutations(range(r))

    for p1 in ps1:
        assert len(p1) == r
        inv_p1 = sorted(range(r), key=p1.__getitem__)
        u1 = dpt.permute_dims(t1, p1)
        x1_axes = inv_p1[1:]
        for p2 in ps2:
            inv_p2 = sorted(range(r), key=p2.__getitem__)
            u2 = dpt.permute_dims(t2, p2)
            x2_axes = inv_p2[1:]

            tdr = dpt.tensordot(u1, u2, axes=(x1_axes, x2_axes))
            assert tdr.shape == t1.shape[:1] + t2.shape[:1]
            assert dpt.allclose(tdr, dpt.full_like(tdr, fill_value=expected))


def test_tensordot_validation():
    get_queue_or_skip()

    with pytest.raises(TypeError):
        dpt.tensordot(dict(), dict())

    t1 = dpt.empty((10, 10, 10))
    with pytest.raises(TypeError):
        dpt.tensordot(t1, dict())

    t2 = dpt.empty((10, 10, 10))
    q = dpctl.SyclQueue(t2.sycl_context, t2.sycl_device, property="in_order")
    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.tensordot(t1, t2.to_device(q))

    invalid_axes = (
        1,
        2,
        3,
    )
    with pytest.raises(ValueError):
        dpt.tensordot(t1, t2, axes=invalid_axes)

    invalid_axes = 5.2
    with pytest.raises(TypeError):
        dpt.tensordot(t1, t2, axes=invalid_axes)

    invalid_axes = (
        (1,),
        (
            0,
            2,
        ),
    )
    with pytest.raises(ValueError):
        dpt.tensordot(t1, t2, axes=invalid_axes)

    with pytest.raises(ValueError):
        dpt.tensordot(t1[..., :5], t2)


def test_tensordot_promotion():
    get_queue_or_skip()

    t1 = dpt.zeros((10, 10), dtype="i4")
    t2 = dpt.zeros((10, 10), dtype="i8")

    r1 = dpt.tensordot(t1, t2)
    assert r1.dtype == t2.dtype

    r2 = dpt.tensordot(t2, t1)
    assert r2.dtype == t2.dtype

    t3 = dpt.zeros((10, 10), dtype="u4")
    r3 = dpt.tensordot(t1, t3)
    assert r3.dtype == dpt.result_type(t1, t3)


def test_tensordot_axes_errors():
    get_queue_or_skip()

    m1 = dpt.zeros((10, 10), dtype="i4")
    m2 = dpt.zeros((10, 10), dtype="i4")

    with pytest.raises(ValueError):
        dpt.tensordot(m1, m2, axes=-1)


# tests for gh-1570
def test_tensordot_gemm_small_k_m():
    get_queue_or_skip()

    x1 = dpt.asarray(1, dtype="i2")
    x2 = dpt.asarray([0, 1, 0, 0], dtype="i2")

    res = dpt.tensordot(x1, x2, axes=0)
    assert dpt.all(x2 == res)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 511
    v1 = dpt.ones(n, dtype=dtype)

    v2 = dpt.ones(n, dtype=dtype)

    r = dpt.vecdot(v1, v2)
    expected_value = _map_int_to_type(n, r.dtype)
    assert r == expected_value


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_3d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m1, m2, n = 7, 3, 511
    v1 = dpt.ones((m1, m2, n), dtype=dtype)

    v2 = dpt.ones((m1, m2, n), dtype=dtype)

    r = dpt.vecdot(v1, v2)

    assert r.shape == (
        m1,
        m2,
    )
    expected_value = _map_int_to_type(n, r.dtype)
    assert dpt.all(r == expected_value)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_axis(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m1, m2, n = 7, 3, 511
    v1 = dpt.ones((m1, n, m2), dtype=dtype)

    v2 = dpt.ones((m1, n, m2), dtype=dtype)

    r = dpt.vecdot(v1, v2, axis=-2)

    assert r.shape == (
        m1,
        m2,
    )
    expected_value = _map_int_to_type(n, r.dtype)
    assert dpt.all(r == expected_value)


@pytest.mark.parametrize("dtype", _numeric_types)
def test_vecdot_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m1, m2, n = 7, 3, 511
    list1 = [1, 0, 2, 0]
    pattern1 = dpt.asarray(list1, dtype=dtype)
    n_padded1 = pattern1.size * (1 + ((n - 1) // pattern1.size))
    v1 = dpt.tile(dpt.reshape(pattern1, (1, -1, 1)), (m1, n_padded1, m2))[
        ::-1, :n, :
    ]

    list2 = [1, 2, 1, 2]
    pattern2 = dpt.asarray(list2, dtype=dtype)
    n_padded2 = pattern2.size * (1 + ((n - 1) // pattern2.size))
    v2 = dpt.tile(dpt.reshape(pattern2, (1, -1, 1)), (m1, n_padded2, m2))[
        :, :n, ::-1
    ]

    r = dpt.vecdot(v1, v2, axis=-2)

    ref = sum(
        el1 * el2
        for el1, el2 in zip((list1 * n_padded1)[:n], (list2 * n_padded1)[:n])
    )

    assert r.shape == (
        m1,
        m2,
    )
    ref = _map_int_to_type(ref, r.dtype)
    assert dpt.all(r == ref)


def test_vector_arg_validation():
    get_queue_or_skip()

    s1, s2 = dpt.ones(tuple()), dpt.zeros(tuple())
    v1, v2 = dpt.ones(16), dpt.zeros(16)

    with pytest.raises(ValueError):
        dpt.vecdot(s1, v2)

    with pytest.raises(ValueError):
        dpt.vecdot(v1, s2)

    with pytest.raises(TypeError):
        dpt.vecdot(dict(), v2)

    with pytest.raises(TypeError):
        dpt.vecdot(v2, None)

    with pytest.raises(ValueError):
        dpt.vecdot(v1[:5], v2[:4])

    with pytest.raises(ValueError):
        dpt.vecdot(v1, v2, axis=2)

    with pytest.raises(ValueError):
        dpt.vecdot(v1, v2, axis=-2)

    q = dpctl.SyclQueue(
        v2.sycl_context, v2.sycl_device, property="enable_profiling"
    )
    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.vecdot(v1, v2.to_device(q))

    m1 = dpt.empty((10, 5))
    m2 = dpt.empty((5, 5))
    with pytest.raises(ValueError):
        dpt.vecdot(m1, m2, axis=-1)


def test_vecdot_broadcast():
    get_queue_or_skip()

    for dt1, dt2 in [
        (dpt.int32, dpt.int32),
        (dpt.int32, dpt.int64),
        (dpt.int64, dpt.int32),
        (dpt.int32, dpt.uint32),
    ]:
        m1 = dpt.zeros((1, 5), dtype=dt1)
        m2 = dpt.zeros((5, 5), dtype=dt2)
        r1 = dpt.vecdot(m1, m2, axis=-1)
        r2 = dpt.vecdot(m2, m1, axis=-1)
        assert r1.shape == r2.shape


@pytest.mark.parametrize("dt1", _numeric_types)
@pytest.mark.parametrize("dt2", _numeric_types)
def test_vecdot_type_promotion(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    v1 = dpt.ones(128, dtype=dt1)
    v2 = dpt.ones(128, dtype=dt2)

    r = dpt.vecdot(v1, v2)
    mul = v1 * v2
    assert r.shape == ()
    assert r.dtype == mul.dtype
    assert dpt.allclose(r, dpt.sum(mul, dtype=mul.dtype))


def test_vecdot_broadcast_o1_buffer():
    get_queue_or_skip()

    v1 = dpt.arange(10, dtype="i2")
    v2 = dpt.ones((5, 10), dtype="i4")

    res1 = dpt.vecdot(v1, v2)
    assert res1.shape == (5,)

    res2 = dpt.vecdot(v2, v1)
    assert res2.shape == (5,)


def test_vecdot_contig_small():
    get_queue_or_skip()

    n = 1
    for dt in [dpt.int16, dpt.int32, dpt.complex64]:
        v1 = dpt.zeros((10, n), dtype=dt)
        v2 = dpt.ones_like(v1, dtype=dt)
        v1[-1] = 1
        res = dpt.vecdot(v1, v2)
        assert dpt.all(res[:-1] == 0)
        assert res[-1] == n


def test_matmul_out_appended_axes():
    get_queue_or_skip()

    n0, n1, n2 = 4, 10, 5
    # vm
    x1 = dpt.ones(n1, dtype="i4")
    x2 = dpt.ones((n0, n1, n2), dtype="i4")
    out = dpt.empty((n0, n2), dtype="i4")

    dpt.matmul(x1, x2, out=out)
    assert dpt.all(out == n1)

    # mv
    x2 = x2.mT
    x1, x2 = x2, x1
    dpt.matmul(x1, x2, out=out)
    assert dpt.all(out == n1)

    # vv
    x1 = dpt.ones(n1, dtype="i4")
    out = dpt.empty((), dtype="i4")
    dpt.matmul(x1, x2, out=out)
    assert out == n1
