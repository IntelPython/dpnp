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

import dpnp.tensor as dpt

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _compare_dtypes,
)


@pytest.mark.parametrize("op1_dtype", _all_dtypes)
@pytest.mark.parametrize("op2_dtype", _all_dtypes)
def test_not_equal_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype)

    r = dpt.not_equal(ar1, ar2)
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.not_equal(
        np.zeros(1, dtype=op1_dtype), np.zeros(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, False, dtype=r.dtype)).all()
    assert r.sycl_queue == ar1.sycl_queue

    ar3 = dpt.ones(sz, dtype=op1_dtype)
    ar4 = dpt.ones(2 * sz, dtype=op2_dtype)

    r = dpt.not_equal(ar3[::-1], ar4[::2])
    assert isinstance(r, dpt.usm_ndarray)
    expected_dtype = np.not_equal(
        np.ones(1, dtype=op1_dtype), np.ones(1, dtype=op2_dtype)
    ).dtype
    assert _compare_dtypes(r.dtype, expected_dtype, sycl_queue=q)
    assert r.shape == ar3.shape
    assert (dpt.asnumpy(r) == np.full(r.shape, False, dtype=r.dtype)).all()


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_not_equal_alignment(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n = 256
    s = dpt.concat((dpt.zeros(n, dtype=dtype), dpt.zeros(n, dtype=dtype)))

    mask = s[:-1] != s[1:]
    (pos,) = dpt.nonzero(mask)
    assert dpt.all(pos == n)

    out_arr = dpt.zeros(2 * n, dtype=mask.dtype)
    dpt.not_equal(s[:-1], s[1:], out=out_arr[1:])
    (pos,) = dpt.nonzero(mask)
    assert dpt.all(pos == (n + 1))
