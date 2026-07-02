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


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_log1p_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.log1p(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.log1p(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_log1p_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(0, 2, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.log1p(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.log1p(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_log1p_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2 * 1027

    X = dpt.linspace(0, 2, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.log1p(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.log1p(Xnp), atol=tol, rtol=tol)


def test_log1p_special_cases():
    q = get_queue_or_skip()

    X = dpt.asarray(
        [dpt.nan, -2.0, -1.0, -0.0, 0.0, dpt.inf],
        dtype="f4",
        sycl_queue=q,
    )
    res = np.asarray([np.nan, np.nan, -np.inf, -0.0, 0.0, np.inf], dtype="f4")

    tol = dpt.finfo(X.dtype).resolution
    with np.errstate(divide="ignore", invalid="ignore"):
        assert_allclose(dpt.asnumpy(dpt.log1p(X)), res, atol=tol, rtol=tol)

    # special cases for complex
    vals = [
        complex(-1.0, 0.0),
        complex(2.0, dpt.inf),
        complex(2.0, dpt.nan),
        complex(-dpt.inf, 1.0),
        complex(dpt.inf, 1.0),
        complex(-dpt.inf, dpt.inf),
        complex(dpt.inf, dpt.inf),
        complex(dpt.inf, dpt.nan),
        complex(dpt.nan, 1.0),
        complex(dpt.nan, dpt.inf),
        complex(dpt.nan, dpt.nan),
    ]
    X = dpt.asarray(vals, dtype=dpt.complex64)
    c_nan = complex(np.nan, np.nan)
    res = np.asarray(
        [
            complex(-np.inf, 0.0),
            complex(np.inf, np.pi / 2),
            c_nan,
            complex(np.inf, np.pi),
            complex(np.inf, 0.0),
            complex(np.inf, 3 * np.pi / 4),
            complex(np.inf, np.pi / 4),
            complex(np.inf, np.nan),
            c_nan,
            complex(np.inf, np.nan),
            c_nan,
        ],
        dtype=np.complex64,
    )

    tol = dpt.finfo(X.dtype).resolution
    with np.errstate(invalid="ignore"):
        dpt_res = dpt.asnumpy(dpt.log1p(X))
        assert_allclose(np.real(dpt_res), np.real(res), atol=tol, rtol=tol)
        assert_allclose(np.imag(dpt_res), np.imag(res), atol=tol, rtol=tol)
