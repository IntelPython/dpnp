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
from numpy.testing import assert_allclose

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
def test_expm1_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.asarray(0, dtype=dtype, sycl_queue=q)
    expected_dtype = np.expm1(np.array(0, dtype=dtype)).dtype
    expected_dtype = _map_to_device_dtype(expected_dtype, q.sycl_device)
    assert dpt.expm1(X).dtype == expected_dtype


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_expm1_output_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 1027

    X = dpt.linspace(-2, 2, num=n_seq, dtype=dtype, sycl_queue=q)
    Xnp = dpt.asnumpy(X)

    Y = dpt.expm1(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.expm1(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
def test_expm1_output_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n_seq = 2 * 1027

    X = dpt.linspace(-2, 2, num=n_seq, dtype=dtype, sycl_queue=q)[::-2]
    Xnp = dpt.asnumpy(X)

    Y = dpt.expm1(X)
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), np.expm1(Xnp), atol=tol, rtol=tol)


@pytest.mark.parametrize("usm_type", _usm_types)
def test_expm1_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("f4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 1 / 50
    X[..., 1::2] = 1 / 25

    Y = dpt.expm1(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = np.empty(input_shape, dtype=arg_dt)
    expected_Y[..., 0::2] = np.expm1(np.float32(1 / 50))
    expected_Y[..., 1::2] = np.expm1(np.float32(1 / 25))
    tol = 8 * dpt.finfo(Y.dtype).resolution

    assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_expm1_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 1 / 50
    X[..., 1::2] = 1 / 25

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.expm1(dpt.asnumpy(U))
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.expm1(U, order=ord)
            tol = 8 * max(
                dpt.finfo(Y.dtype).resolution,
                np.finfo(expected_Y.dtype).resolution,
            )
            assert_allclose(dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol)


def test_expm1_special_cases():
    get_queue_or_skip()

    X = dpt.asarray([dpt.nan, 0.0, -0.0, dpt.inf, -dpt.inf], dtype="f4")
    res = np.asarray([np.nan, 0.0, -0.0, np.inf, -1.0], dtype="f4")

    tol = dpt.finfo(X.dtype).resolution
    assert_allclose(dpt.asnumpy(dpt.expm1(X)), res, atol=tol, rtol=tol)

    # special cases for complex variant
    num_finite = 1.0
    vals = [
        complex(0.0, 0.0),
        complex(num_finite, dpt.inf),
        complex(num_finite, dpt.nan),
        complex(dpt.inf, 0.0),
        complex(-dpt.inf, num_finite),
        complex(dpt.inf, num_finite),
        complex(-dpt.inf, dpt.inf),
        complex(dpt.inf, dpt.inf),
        complex(-dpt.inf, dpt.nan),
        complex(dpt.inf, dpt.nan),
        complex(dpt.nan, 0.0),
        complex(dpt.nan, num_finite),
        complex(dpt.nan, dpt.nan),
    ]
    X = dpt.asarray(vals, dtype=dpt.complex64)
    cis_1 = complex(np.cos(num_finite), np.sin(num_finite))
    c_nan = complex(np.nan, np.nan)
    res = np.asarray(
        [
            complex(0.0, 0.0),
            c_nan,
            c_nan,
            complex(np.inf, 0.0),
            0.0 * cis_1 - 1.0,
            np.inf * cis_1 - 1.0,
            complex(-1.0, 0.0),
            complex(np.inf, np.nan),
            complex(-1.0, 0.0),
            complex(np.inf, np.nan),
            complex(np.nan, 0.0),
            c_nan,
            c_nan,
        ],
        dtype=np.complex64,
    )

    tol = dpt.finfo(X.dtype).resolution
    with np.errstate(invalid="ignore"):
        assert_allclose(dpt.asnumpy(dpt.expm1(X)), res, atol=tol, rtol=tol)
