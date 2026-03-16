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

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _map_to_device_dtype,
    _no_complex_dtypes,
    _real_fp_dtypes,
)


@pytest.mark.parametrize("dtype", _no_complex_dtypes)
def test_rsqrt_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    expected_dtype = np.reciprocal(np.sqrt(np.array(1, dtype=dtype))).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.rsqrt(x).dtype == expected_dtype


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_rsqrt_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    x = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)
    res = dpt.rsqrt(x)
    expected = np.reciprocal(np.sqrt(dpt.asnumpy(x), dtype=dtype))
    tol = 8 * dpt.finfo(res.dtype).resolution
    assert_allclose(dpt.asnumpy(res), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_rsqrt_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2054

    x = dpt.linspace(1, 13, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    res = dpt.rsqrt(x)
    expected = np.reciprocal(np.sqrt(dpt.asnumpy(x), dtype=dtype))
    tol = 8 * dpt.finfo(res.dtype).resolution
    assert_allclose(dpt.asnumpy(res), expected, atol=tol, rtol=tol)


def test_rsqrt_special_cases():
    get_queue_or_skip()

    x = dpt.asarray([dpt.nan, -1.0, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    res = dpt.rsqrt(x)
    expected = dpt.asarray(
        [dpt.nan, dpt.nan, dpt.inf, -dpt.inf, 0.0, dpt.nan], dtype="f4"
    )
    assert dpt.allclose(res, expected, equal_nan=True)
