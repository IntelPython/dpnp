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
from dpctl_ext.tensor._type_utils import _can_cast

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _complex_fp_dtypes,
    _no_complex_dtypes,
)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_angle_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(1, dtype=dtype, sycl_queue=q)
    dt = dpt.dtype(dtype)
    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(dt, dpt.complex64, _fp16, _fp64):
        assert dpt.angle(x).dtype == dpt.float32
    else:
        assert dpt.angle(x).dtype == dpt.float64


@pytest.mark.parametrize("dtype", _no_complex_dtypes[1:])
def test_angle_real(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.arange(10, dtype=dtype, sycl_queue=q)
    r = dpt.angle(x)

    assert dpt.all(r == 0)


@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_angle_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    tol = 8 * dpt.finfo(dtype).resolution
    vals = dpt.pi * dpt.arange(10, dtype=dpt.finfo(dtype).dtype, sycl_queue=q)

    x = dpt.zeros(10, dtype=dtype, sycl_queue=q)

    x.imag[...] = vals
    r = dpt.angle(x)
    expected = dpt.atan2(x.imag, x.real)
    assert dpt.allclose(r, expected, atol=tol, rtol=tol)

    x.real[...] += dpt.pi
    r = dpt.angle(x)
    expected = dpt.atan2(x.imag, x.real)
    assert dpt.allclose(r, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_angle_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    vals = [np.nan, -np.nan, np.inf, -np.inf, +0.0, -0.0]
    vals = [complex(*val) for val in itertools.product(vals, repeat=2)]

    x = dpt.asarray(vals, dtype=dtype, sycl_queue=q)

    r = dpt.angle(x)
    expected = dpt.atan2(x.imag, x.real)

    tol = 8 * dpt.finfo(dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)
