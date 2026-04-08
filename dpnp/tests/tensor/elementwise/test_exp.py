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

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _map_to_device_dtype,
    _usm_types,
)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_exp_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.exp(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.exp(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_exp_real_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    Xnp = np.linspace(0.01, 88.1, num=n_seq, dtype=dtype)
    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt.exp(X)
    with np.errstate(all="ignore"):
        Ynp = np.exp(Xnp)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(Y), np.repeat(Ynp, n_rep), atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=dtype)
    dpt.exp(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), np.repeat(Ynp, n_rep), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_exp_complex_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    low = -88.0
    high = 88.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = np.array([complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt.exp(X)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np.exp(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt.exp(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np.exp(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("usm_type", _usm_types)
def test_exp_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 16.0
    X[..., 1::2] = 23.0

    Y = dpt.exp(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.exp(np.float32(16.0))
    expected_Y[..., 1::2] = np.exp(np.float32(23.0))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_exp_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 8.0
    X[..., 1::2] = 11.0

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.exp(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.exp(U, order=ord)
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_exp_analytical_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    log2_ = 0.69314718055994530943
    Xnp = np.array(x, dtype=dtype) * log2_
    X = dpt.asarray(Xnp, dtype=dtype)
    assert_allclose(dpt.asnumpy(dpt.exp(X)), np.exp(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_exp_real_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    x = [np.nan, np.inf, -np.inf, 0.0, -0.0]
    Xnp = np.array(x, dtype=dtype)
    X = dpt.asarray(x, dtype=dtype)

    Y = dpt.asnumpy(dpt.exp(X))
    Ynp = np.exp(Xnp)
    assert_allclose(Y, Ynp, atol=tol, rtol=tol)
    assert_array_equal(np.signbit(Y), np.signbit(Ynp))


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_exp_real_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    for ii in sizes:
        Xnp = np.random.uniform(low=0.01, high=88.1, size=ii)
        Xnp.astype(dtype)
        X = dpt.asarray(Xnp)
        Ynp = np.exp(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt.exp(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_exp_complex_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -88.0
    high = 88.0
    for ii in sizes:
        x1 = np.random.uniform(low=low, high=high, size=ii)
        x2 = np.random.uniform(low=low, high=high, size=ii)
        Xnp = np.array([complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype)
        X = dpt.asarray(Xnp)
        Ynp = np.exp(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt.exp(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_exp_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, +0.0, -0.0, +1.0, -1.0]
    xc = [complex(*val) for val in itertools.product(x, repeat=2)]

    Xc_np = np.array(xc, dtype=dtype)
    Xc = dpt.asarray(Xc_np, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Ynp = np.exp(Xc_np)
    Y = dpt.exp(Xc)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.real(Y)), np.real(Ynp), atol=tol, rtol=tol)
    assert_allclose(dpt.asnumpy(dpt.imag(Y)), np.imag(Ynp), atol=tol, rtol=tol)
