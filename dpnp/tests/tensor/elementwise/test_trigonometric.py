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

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dpnp.tensor as dpt

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _map_to_device_dtype,
)

_trig_funcs = [(np.sin, dpt.sin), (np.cos, dpt.cos), (np.tan, dpt.tan)]
_inv_trig_funcs = [
    (np.arcsin, dpt.asin),
    (np.arccos, dpt.acos),
    (np.arctan, dpt.atan),
]
_all_funcs = _trig_funcs + _inv_trig_funcs


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_trig_out_type(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt_call(x).dtype == expected_dtype


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    if np_call in _trig_funcs:
        Xnp = np.linspace(
            -np.pi / 2 * 0.99, np.pi / 2 * 0.99, num=n_seq, dtype=dtype
        )
    if np_call == np.arctan:
        Xnp = np.linspace(-100.0, 100.0, num=n_seq, dtype=dtype)
    else:
        Xnp = np.linspace(-1.0, 1.0, num=n_seq, dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_trig_complex_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 256
    n_rep = 137
    low = -9.0
    high = 9.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = x1 + 1j * x2

    # stay away from poles and branch lines
    modulus = np.abs(Xnp)
    sel = np.logical_or(
        modulus < 0.9,
        np.logical_and(
            modulus > 1.2, np.minimum(np.abs(x2), np.abs(x1)) > 0.05
        ),
    )
    Xnp = Xnp[sel]

    X = dpt.repeat(dpt.asarray(Xnp, dtype=dtype, sycl_queue=q), n_rep)
    Y = dpt_call(X)

    expected = np.repeat(np_call(Xnp.astype(dtype)), n_rep)

    tol = 50 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(Y), expected, atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 3, 4, 6, 8, 9, 24, 50, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -100.0
    high = 100.0
    if np_call in [np.arccos, np.arcsin]:
        low = -1.0
        high = 1.0
    elif np_call in [np.tan]:
        low = -np.pi / 2 * (0.99)
        high = np.pi / 2 * (0.99)

    for ii in sizes:
        Xnp = np.random.uniform(low=low, high=high, size=ii)
        Xnp.astype(dtype)
        X = dpt.asarray(Xnp)
        Ynp = np_call(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt_call(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_trig_complex_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 50 * dpt.finfo(dtype).resolution

    low = -9.0
    high = 9.0
    while True:
        x1 = np.random.uniform(low=low, high=high, size=2 * sum(sizes))
        x2 = np.random.uniform(low=low, high=high, size=2 * sum(sizes))
        Xnp_all = np.array(
            [complex(v1, v2) for v1, v2 in zip(x1, x2)], dtype=dtype
        )

        # stay away from poles and branch lines
        modulus = np.abs(Xnp_all)
        sel = np.logical_or(
            modulus < 0.9,
            np.logical_and(
                modulus > 1.2, np.minimum(np.abs(x2), np.abs(x1)) > 0.05
            ),
        )
        Xnp_all = Xnp_all[sel]
        if Xnp_all.size > sum(sizes):
            break

    pos = 0
    for ii in sizes:
        pos = pos + ii
        Xnp = Xnp_all[:pos]
        Xnp = Xnp[-ii:]
        X = dpt.asarray(Xnp)
        Ynp = np_call(Xnp)
        for jj in strides:
            assert_allclose(
                dpt.asnumpy(dpt_call(X[::jj])),
                Ynp[::jj],
                atol=tol,
                rtol=tol,
            )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_trig_real_special_cases(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 2.0, -2.0, +0.0, -0.0, +1.0, -1.0]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Y_np = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    Y = dpt_call(yf)
    assert_allclose(dpt.asnumpy(Y), Y_np, atol=tol, rtol=tol)
