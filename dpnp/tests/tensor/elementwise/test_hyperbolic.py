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
    _complex_fp_dtypes,
    _map_to_device_dtype,
    _real_fp_dtypes,
)

_hyper_funcs = [(np.sinh, dpt.sinh), (np.cosh, dpt.cosh), (np.tanh, dpt.tanh)]
_inv_hyper_funcs = [
    (np.arcsinh, dpt.asinh),
    (np.arccosh, dpt.acosh),
    (np.arctanh, dpt.atanh),
]
_all_funcs = _hyper_funcs + _inv_hyper_funcs


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_hyper_out_type(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    a = 1 if np_call == np.arccosh else 0

    x = dpt.asarray(a, dtype=dtype, sycl_queue=q)
    expected_dtype = np_call(np.array(a, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt_call(x).dtype == expected_dtype


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_hyper_real_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    if np_call == np.arctanh:
        Xnp = np.linspace(-0.9, 0.9, num=n_seq, dtype=dtype)
    elif np_call == np.arccosh:
        Xnp = np.linspace(1.01, 10.0, num=n_seq, dtype=dtype)
    else:
        Xnp = np.linspace(-10.0, 10.0, num=n_seq, dtype=dtype)

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    tol = 8 * dpt.finfo(Y.dtype).resolution
    assert_allclose(
        dpt.asnumpy(Y), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(
        dpt.asnumpy(Z), np.repeat(np_call(Xnp), n_rep), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_hyper_complex_contig(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 100
    n_rep = 137
    low = -9.0
    high = 9.0
    x1 = np.random.uniform(low=low, high=high, size=n_seq)
    x2 = np.random.uniform(low=low, high=high, size=n_seq)
    Xnp = x1 + 1j * x2

    X = dpt.asarray(np.repeat(Xnp, n_rep), dtype=dtype, sycl_queue=q)
    Y = dpt_call(X)

    expected = np.repeat(np_call(Xnp), n_rep).astype(dtype)
    tol = 50 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(Y), expected, atol=tol, rtol=tol)

    Z = dpt.empty_like(X, dtype=dtype)
    dpt_call(X, out=Z)

    assert_allclose(dpt.asnumpy(Z), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_hyper_real_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 8 * dpt.finfo(dtype).resolution

    low = -10.0
    high = 10.0
    if np_call == np.arctanh:
        low = -0.9
        high = 0.9
    elif np_call == np.arccosh:
        low = 1.01
        high = 100.0

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
@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_hyper_complex_strided(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    np.random.seed(42)
    strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sizes = [2, 4, 6, 8, 9, 24, 72]
    tol = 50 * dpt.finfo(dtype).resolution

    low = -8.0
    high = 8.0
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


@pytest.mark.parametrize("np_call, dpt_call", _all_funcs)
@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_hyper_real_special_cases(np_call, dpt_call, dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [np.nan, np.inf, -np.inf, 2.0, -2.0, +0.0, -0.0, +1.0, -1.0]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    with np.errstate(all="ignore"):
        Y_np = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    assert_allclose(dpt.asnumpy(dpt_call(yf)), Y_np, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_acosh_zero_nan(dtype):
    # check acosh(±0 + NaN j) = NaN ± π/2 j (Array API spec)
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = [complex(+0.0, np.nan), complex(-0.0, np.nan)]

    xf = np.array(x, dtype=dtype)
    yf = dpt.asarray(xf, dtype=dtype, sycl_queue=q)

    Y_dpt = dpt.asnumpy(dpt.acosh(yf))

    assert np.isnan(Y_dpt.real).all()
    assert_allclose(np.abs(Y_dpt.imag), np.pi / 2, atol=1e-6, strict=False)


@pytest.mark.parametrize("np_call, dpt_call", [(np.arccosh, dpt.acosh)])
# @pytest.mark.parametrize("np_call, dpt_call", _inv_hyper_funcs)
@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_inv_hyper_large_negative_real(np_call, dpt_call, dtype):
    """
    Test inverse hyperbolic functions for large negative real parts.

    Regression acosh test for threshold bug where catastrophic cancellation in
    z + sqrt(z² - 1) caused log(0) = -infinity for |Re(z)| in [4e7, 9e7+]
    with negative real parts (gh-2924).
    """

    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # input values that previously returned infinity
    thr = np.sqrt(1 / dpt.finfo(dtype).eps) / 2
    x = [
        complex(-4e7, 1.0),  # Boundary of bug zone
        complex(-9e7, 1.0),  # Middle of bug zone
        complex(-1e8, 1.0),  # Upper range
        complex(-4.45712982e8, 1.0),  # Original reported value
        complex(thr, 1.0),  # Exact threshold value
        complex(np.nextafter(thr, np.inf), 1.0),  # Next after threshold
    ]

    xf = np.asarray(x, dtype=dtype)
    yf = dpt.asarray(x, dtype=dtype, sycl_queue=q)

    result = dpt_call(yf)
    expected = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    assert not dpt.any(dpt.isinf(result))
    assert_allclose(dpt.asnumpy(result), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("np_call, dpt_call", [(np.arccosh, dpt.acosh)])
# @pytest.mark.parametrize("np_call, dpt_call", _inv_hyper_funcs)
@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
@pytest.mark.parametrize("magnitude", [4e7, 9e7, 1e8, 4.45e8])
def test_inv_hyper_all_quadrants_large(np_call, dpt_call, dtype, magnitude):
    """
    Test inverse hyperbolic functions for large complex values in all quadrants.

    Ensures the threshold fix works correctly for all sign combinations
    of real and imaginary parts.
    """

    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # input values with four quadrants with large magnitude
    x = [
        complex(magnitude, 1.0),  # +Re, +Im
        complex(-magnitude, 1.0),  # -Re, +Im (bug zone)
        complex(magnitude, -1.0),  # +Re, -Im
        complex(-magnitude, -1.0),  # -Re, -Im (bug zone)
    ]

    xf = np.asarray(x, dtype=dtype)
    yf = dpt.asarray(x, dtype=dtype, sycl_queue=q)

    result = dpt_call(yf)
    expected = np_call(xf)

    tol = 8 * dpt.finfo(dtype).resolution
    assert not dpt.any(dpt.isinf(result))
    assert_allclose(dpt.asnumpy(result), expected, atol=tol, rtol=tol)
