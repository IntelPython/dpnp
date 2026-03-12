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
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
from dpnp.tests.tensor.elementwise.utils import (
    _all_dtypes,
    _map_to_device_dtype,
    _usm_types,
)
from dpnp.tests.tensor.helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_complex_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.real(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.real(X).dtype == expected_dtype

    expected_dtype = np.imag(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.imag(X).dtype == expected_dtype

    expected_dtype = np.conj(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.conj(X).dtype == expected_dtype


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_complex_output(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100

    x1 = np.linspace(0, 10, num=n_seq, dtype=dtype)
    x2 = np.linspace(0, 20, num=n_seq, dtype=dtype)
    Xnp = x1 + 1j * x2
    X = dpt.asarray(Xnp, sycl_queue=q)

    Y = dpt_call(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np_call(Xnp), atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=Y.dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), np_call(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("usm_type", _usm_types)
def test_complex_usm_type(np_call, dpt_call, usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("c8")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = np.pi / 6 + 1j * np.pi / 3
    X[..., 1::2] = np.pi / 3 + 1j * np.pi / 6

    Y = dpt_call(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np_call(np.complex64(np.pi / 6 + 1j * np.pi / 3))
    expected_Y[..., 1::2] = np_call(np.complex64(np.pi / 3 + 1j * np.pi / 6))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_complex_order(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = np.pi / 6 + 1j * np.pi / 3
    X[..., 1::2] = np.pi / 3 + 1j * np.pi / 6

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np_call(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt_call(U, order=ord)
            assert_allclose(dpt.asnumpy(Y), expected_Y)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_projection_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = [
        complex(1, 2),
        complex(dpt.inf, -1),
        complex(0, -dpt.inf),
        complex(-dpt.inf, dpt.nan),
    ]
    Y = [
        complex(1, 2),
        complex(np.inf, -0.0),
        complex(np.inf, -0.0),
        complex(np.inf, 0.0),
    ]

    Xf = dpt.asarray(X, dtype=dtype, sycl_queue=q)
    Yf = np.array(Y, dtype=dtype)

    tol = 8 * dpt.finfo(Xf.dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.proj(Xf)), Yf, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_projection(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    Xf = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    out_dtype = dpt.proj(Xf).dtype
    Yf = np.array(complex(1, 0), dtype=out_dtype)

    tol = 8 * dpt.finfo(Yf.dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.proj(Xf)), Yf, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "np_call, dpt_call",
    [(np.real, dpt.real), (np.imag, dpt.imag), (np.conj, dpt.conj)],
)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_complex_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -1000.0
    high = 1000.0
    for ii in sizes:
        x1 = np.random.uniform(low=low, high=high, size=ii)
        x2 = np.random.uniform(low=low, high=high, size=ii)
        Xnp = np.array([complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype)
        X = dpt.asarray(Xnp)
        Ynp = np_call(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt_call(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_complex_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, -np.nan, np.inf, -np.inf, +0.0, -0.0]
    xc = [complex(*val) for val in itertools.product(x, repeat=2)]

    Xc_np = np.array(xc, dtype=dtype)
    Xc = dpt.asarray(Xc_np, dtype=dtype, sycl_queue=q)

    tol = 8 * dpt.finfo(dtype).resolution

    actual = dpt.real(Xc)
    expected = np.real(Xc_np)
    assert_allclose(dpt.asnumpy(actual), expected, atol=tol, rtol=tol)

    actual = dpt.imag(Xc)
    expected = np.imag(Xc_np)
    assert_allclose(dpt.asnumpy(actual), expected, atol=tol, rtol=tol)

    actual = dpt.conj(Xc)
    expected = np.conj(Xc_np)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_allclose(dpt.asnumpy(actual), expected, atol=tol, rtol=tol)
