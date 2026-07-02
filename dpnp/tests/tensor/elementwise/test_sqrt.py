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
from numpy.testing import assert_allclose, assert_equal

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


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_sqrt_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.sqrt(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.sqrt(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sqrt_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.sqrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.sqrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_sqrt_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    X = dpt.linspace(0, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.sqrt(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.sqrt(Xnp), atol=tol, rtol=tol)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
def test_sqrt_special_cases():
    q = get_queue_or_skip()

    X = dpt.asarray(
        [dpt.nan, -1.0, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4", sycl_queue=q
    )
    Xnp = dpt.asnumpy(X)

    assert_equal(dpt.asnumpy(dpt.sqrt(X)), np.sqrt(Xnp))


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_sqrt_real_fp_special_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_

    x = dpt.asarray(inps_, dtype=dtype)
    r = dpt.sqrt(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_np = np.sqrt(np.asarray(inps_, dtype=dtype))

    expected = dpt.asarray(expected_np, dtype=dtype)
    tol = dpt.finfo(r.dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_sqrt_complex_fp_special_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_
    c_ = [complex(*v) for v in itertools.product(inps_, repeat=2)]

    z = dpt.asarray(c_, dtype=dtype)
    r = dpt.sqrt(z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_np = np.sqrt(np.asarray(c_, dtype=dtype))

    expected = dpt.asarray(expected_np, dtype=dtype)
    tol = dpt.finfo(r.dtype).resolution

    if not dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True):
        for i in range(r.shape[0]):
            failure_data = []
            if not dpt.allclose(
                r[i], expected[i], atol=tol, rtol=tol, equal_nan=True
            ):
                msg = (
                    f"Test failed for input {z[i]}, i.e. {c_[i]} for index {i}"
                )
                msg += f", results were {r[i]} vs. {expected[i]}"
                failure_data.extend(msg)
        pytest.skip(reason=msg)
