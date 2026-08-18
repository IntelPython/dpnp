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
from .utils import _all_dtypes


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_square_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    X = dpt.arange(5, dtype=arg_dt, sycl_queue=q)
    assert dpt.square(X).dtype == arg_dt

    r = dpt.empty_like(X, dtype=arg_dt)
    dpt.square(X, out=r)
    assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.square(X)))


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_square_special_cases(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    vals = [np.nan, np.inf, -np.inf, 0.0, -0.0]
    X = dpt.asarray(vals, dtype=dtype, sycl_queue=q)
    X_np = dpt.asnumpy(X)

    tol = 8 * dpt.finfo(dtype).resolution
    with np.errstate(all="ignore"):
        assert np.allclose(
            dpt.asnumpy(dpt.square(X)),
            np.square(X_np),
            atol=tol,
            rtol=tol,
            equal_nan=True,
        )
