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
from .utils import _integral_dtypes


@pytest.mark.parametrize("op_dtype", _integral_dtypes)
def test_bitwise_or_dtype_matrix_contig(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 7
    n = 2 * sz
    dt1 = dpt.dtype(op_dtype)
    dt2 = dpt.dtype(op_dtype)

    x1_range_begin = -sz if dpt.iinfo(dt1).min < 0 else 0
    x1 = dpt.arange(x1_range_begin, x1_range_begin + n, dtype=dt1)

    x2_range_begin = -sz if dpt.iinfo(dt2).min < 0 else 0
    x2 = dpt.arange(x2_range_begin, x2_range_begin + n, dtype=dt1)

    r = dpt.bitwise_or(x1, x2)
    assert isinstance(r, dpt.usm_ndarray)

    x1_np = np.arange(x1_range_begin, x1_range_begin + n, dtype=op_dtype)
    x2_np = np.arange(x2_range_begin, x2_range_begin + n, dtype=op_dtype)
    r_np = np.bitwise_or(x1_np, x2_np)

    assert (r_np == dpt.asnumpy(r)).all()


@pytest.mark.parametrize("op_dtype", _integral_dtypes)
def test_bitwise_or_dtype_matrix_strided(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 11
    n = 2 * sz
    dt1 = dpt.dtype(op_dtype)
    dt2 = dpt.dtype(op_dtype)

    x1_range_begin = -sz if dpt.iinfo(dt1).min < 0 else 0
    x1 = dpt.arange(x1_range_begin, x1_range_begin + n, dtype=dt1)[::2]

    x2_range_begin = -(sz // 2) if dpt.iinfo(dt2).min < 0 else 0
    x2 = dpt.arange(x2_range_begin, x2_range_begin + n, dtype=dt1)[::-2]

    r = dpt.bitwise_or(x1, x2)
    assert isinstance(r, dpt.usm_ndarray)

    x1_np = np.arange(x1_range_begin, x1_range_begin + n, dtype=op_dtype)[::2]
    x2_np = np.arange(x2_range_begin, x2_range_begin + n, dtype=op_dtype)[::-2]
    r_np = np.bitwise_or(x1_np, x2_np)

    assert (r_np == dpt.asnumpy(r)).all()


def test_bitwise_or_bool():
    get_queue_or_skip()

    x1 = dpt.asarray([True, False])
    x2 = dpt.asarray([False, True])

    r_bw = dpt.bitwise_or(x1[:, dpt.newaxis], x2[dpt.newaxis])
    r_lo = dpt.logical_or(x1[:, dpt.newaxis], x2[dpt.newaxis])

    assert dpt.all(dpt.equal(r_bw, r_lo))


@pytest.mark.parametrize("dtype", ["?"] + _integral_dtypes)
def test_bitwise_or_inplace_python_scalar(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.zeros((10, 10), dtype=dtype, sycl_queue=q)
    dt_kind = X.dtype.kind
    if dt_kind == "b":
        X |= False
    else:
        X |= int(0)


@pytest.mark.parametrize("op1_dtype", ["?"] + _integral_dtypes)
@pytest.mark.parametrize("op2_dtype", ["?"] + _integral_dtypes)
def test_bitwise_or_inplace_dtype_matrix(op1_dtype, op2_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op1_dtype, q)
    skip_if_dtype_not_supported(op2_dtype, q)

    sz = 127
    ar1 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype, sycl_queue=q)

    dev = q.sycl_device
    _fp16 = dev.has_aspect_fp16
    _fp64 = dev.has_aspect_fp64
    if _can_cast(ar2.dtype, ar1.dtype, _fp16, _fp64, casting="same_kind"):
        ar1 |= ar2
        assert dpt.all(ar1 == 1)

        ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)[::2]
        ar3 |= ar4
        assert dpt.all(ar3 == 1)
    else:
        with pytest.raises(ValueError):
            ar1 |= ar2
            dpt.bitwise_or(ar1, ar2, out=ar1)

    # out is second arg
    ar1 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)
    ar2 = dpt.ones_like(ar1, dtype=op2_dtype, sycl_queue=q)
    if _can_cast(ar1.dtype, ar2.dtype, _fp16, _fp64):
        dpt.bitwise_or(ar1, ar2, out=ar2)
        assert dpt.all(ar2 == 1)

        ar3 = dpt.ones(sz, dtype=op1_dtype, sycl_queue=q)[::-1]
        ar4 = dpt.ones(2 * sz, dtype=op2_dtype, sycl_queue=q)[::2]
        dpt.bitwise_or(ar3, ar4, out=ar4)
        dpt.all(ar4 == 1)
    else:
        with pytest.raises(ValueError):
            dpt.bitwise_or(ar1, ar2, out=ar2)
