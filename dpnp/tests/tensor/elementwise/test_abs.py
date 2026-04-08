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

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import numpy as np
import pytest

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _all_dtypes,
    _complex_fp_dtypes,
    _real_fp_dtypes,
    _usm_types,
)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_abs_out_type(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    X = dpt.asarray(0, dtype=arg_dt, sycl_queue=q)
    if np.issubdtype(arg_dt, np.complexfloating):
        type_map = {
            np.dtype("c8"): np.dtype("f4"),
            np.dtype("c16"): np.dtype("f8"),
        }
        assert dpt.abs(X).dtype == type_map[arg_dt]

        r = dpt.empty_like(X, dtype=type_map[arg_dt])
        dpt.abs(X, out=r)
        assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.abs(X)))
    else:
        assert dpt.abs(X).dtype == arg_dt

        r = dpt.empty_like(X, dtype=arg_dt)
        dpt.abs(X, out=r)
        assert np.allclose(dpt.asnumpy(r), dpt.asnumpy(dpt.abs(X)))


@pytest.mark.parametrize("usm_type", _usm_types)
def test_abs_usm_type(usm_type):
    q = get_queue_or_skip()

    arg_dt = np.dtype("i4")
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, usm_type=usm_type, sycl_queue=q)
    X[..., 0::2] = 1
    X[..., 1::2] = 0

    Y = dpt.abs(X)
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == X.sycl_queue
    assert Y.flags.c_contiguous

    expected_Y = dpt.asnumpy(X)
    assert np.allclose(dpt.asnumpy(Y), expected_Y)


def test_abs_types_property():
    get_queue_or_skip()
    types = dpt.abs.types
    assert isinstance(types, list)
    assert len(types) > 0
    assert types == dpt.abs.types_


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_abs_order(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    exp_dt = np.abs(np.ones(tuple(), dtype=arg_dt)).dtype
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    X[..., 0::2] = 1
    X[..., 1::2] = 0

    for perms in itertools.permutations(range(4)):
        U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
        expected_Y = np.ones(U.shape, dtype=exp_dt)
        expected_Y[..., 1::2] = 0
        expected_Y = np.transpose(expected_Y, perms)
        for ord in ["C", "F", "A", "K"]:
            Y = dpt.abs(U, order=ord)
            assert np.allclose(dpt.asnumpy(Y), expected_Y)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_abs_complex(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    input_shape = (10, 10, 10, 10)
    X = dpt.empty(input_shape, dtype=arg_dt, sycl_queue=q)
    Xnp = np.random.standard_normal(
        size=input_shape
    ) + 1j * np.random.standard_normal(size=input_shape)
    Xnp = Xnp.astype(arg_dt)
    X[...] = Xnp

    for ord in ["C", "F", "A", "K"]:
        for perms in itertools.permutations(range(4)):
            U = dpt.permute_dims(X[:, ::-1, ::-1, :], perms)
            Y = dpt.abs(U, order=ord)
            expected_Y = np.abs(np.transpose(Xnp[:, ::-1, ::-1, :], perms))
            tol = dpt.finfo(Y.dtype).resolution
            np.testing.assert_allclose(
                dpt.asnumpy(Y), expected_Y, atol=tol, rtol=tol
            )


def test_abs_out_overlap():
    get_queue_or_skip()

    X = dpt.arange(-3, 3, 1, dtype="i4")
    expected = dpt.asarray([3, 2, 1, 0, 1, 2], dtype="i4")
    Y = dpt.abs(X, out=X)

    assert Y is X
    assert dpt.all(expected == X)

    X = dpt.arange(-3, 3, 1, dtype="i4")
    expected = expected[::-1]
    Y = dpt.abs(X, out=X[::-1])
    assert Y is not X
    assert dpt.all(expected == X)


@pytest.mark.parametrize("dtype", _real_fp_dtypes)
def test_abs_real_fp_special_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_

    x = dpt.asarray(inps_, dtype=dtype)
    r = dpt.abs(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_np = np.abs(np.asarray(inps_, dtype=dtype))

    expected = dpt.asarray(expected_np, dtype=dtype)
    tol = dpt.finfo(r.dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", _complex_fp_dtypes)
def test_abs_complex_fp_special_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    nans_ = [dpt.nan, -dpt.nan]
    infs_ = [dpt.inf, -dpt.inf]
    finites_ = [-1.0, -0.0, 0.0, 1.0]
    inps_ = nans_ + infs_ + finites_
    c_ = [complex(*v) for v in itertools.product(inps_, repeat=2)]

    z = dpt.asarray(c_, dtype=dtype)
    r = dpt.abs(z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_np = np.abs(np.asarray(c_, dtype=dtype))

    expected = dpt.asarray(expected_np, dtype=dtype)
    tol = dpt.finfo(r.dtype).resolution

    assert dpt.allclose(r, expected, atol=tol, rtol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_abs_alignment(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.ones(512, dtype=dtype)
    r = dpt.abs(x)

    r2 = dpt.abs(x[1:])
    assert np.allclose(dpt.asnumpy(r[1:]), dpt.asnumpy(r2))

    dpt.abs(x[:-1], out=r[1:])
    assert np.allclose(dpt.asnumpy(r[1:]), dpt.asnumpy(r2))
