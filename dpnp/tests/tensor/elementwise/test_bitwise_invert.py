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

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)
from .utils import (
    _compare_dtypes,
    _integral_dtypes,
    _usm_types,
)


@pytest.mark.parametrize(
    "op_dtype",
    [
        "b1",
    ]
    + _integral_dtypes,
)
def test_bitwise_invert_dtype_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 7
    ar1 = dpt.asarray(np.random.randint(0, 2, sz), dtype=op_dtype)

    r = dpt.bitwise_invert(ar1)
    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == ar1.dtype

    expected = np.bitwise_not(dpt.asnumpy(ar1))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected).all()
    assert r.sycl_queue == ar1.sycl_queue

    r2 = dpt.empty_like(r, dtype=r.dtype)
    dpt.bitwise_invert(ar1, out=r2)
    assert dpt.all(dpt.equal(r, r2))

    ar2 = dpt.zeros(sz, dtype=op_dtype)
    r = dpt.bitwise_invert(ar2[::-1])
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.bitwise_not(np.zeros(ar2.shape, dtype=op_dtype))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar2.shape
    assert (dpt.asnumpy(r) == expected).all()

    ar3 = dpt.ones(sz, dtype=op_dtype)
    r2 = dpt.bitwise_invert(ar3[::2])
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.bitwise_not(np.ones(ar3.shape, dtype=op_dtype)[::2])
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert (dpt.asnumpy(r2) == expected).all()

    r3 = dpt.empty_like(r, dtype=r.dtype)
    dpt.bitwise_invert(ar2[::-1], out=r3)
    assert dpt.all(dpt.equal(r, r3))


@pytest.mark.parametrize("op_usm_type", _usm_types)
def test_bitwise_invert_usm_type_matrix(op_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.asarray(
        np.random.randint(0, 2, sz), dtype="i4", usm_type=op_usm_type
    )

    r = dpt.bitwise_invert(ar1)
    assert isinstance(r, dpt.usm_ndarray)
    assert r.usm_type == op_usm_type


def test_bitwise_invert_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.bitwise_invert(ar1, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.bitwise_invert(ar1, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.bitwise_invert(ar1, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.bitwise_invert(ar1, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.zeros((20, 20), dtype="i4", order="F")
    r1 = dpt.bitwise_invert(ar1, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.bitwise_invert(ar1, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.bitwise_invert(ar1, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.bitwise_invert(ar1, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.bitwise_invert(ar1, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.zeros((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.bitwise_invert(ar1, order="K")
    assert r4.strides == (-1, 20)


def test_bitwise_invert_large_boolean():
    get_queue_or_skip()

    x = dpt.tril(dpt.ones((32, 32), dtype="?"), k=-1)
    res = dpt.astype(dpt.bitwise_invert(x), "i4")

    assert dpt.all(res >= 0)
    assert dpt.all(res <= 1)
