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
    _all_dtypes,
    _compare_dtypes,
    _usm_types,
)


@pytest.mark.parametrize("op_dtype", _all_dtypes)
def test_logical_not_dtype_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 7
    ar1_np = np.random.randint(0, 2, sz)
    ar1 = dpt.asarray(ar1_np, dtype=op_dtype)

    r = dpt.logical_not(ar1)
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.logical_not(ar1_np)
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar1.shape
    assert (dpt.asnumpy(r) == expected).all()
    assert r.sycl_queue == ar1.sycl_queue

    r2 = dpt.empty_like(r, dtype=r.dtype)
    dpt.logical_not(ar1, out=r2)
    assert (dpt.asnumpy(r) == dpt.asnumpy(r2)).all()

    ar2 = dpt.zeros(sz, dtype=op_dtype)
    r = dpt.logical_not(ar2[::-1])
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.logical_not(np.zeros(ar2.shape, dtype=op_dtype))
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == ar2.shape
    assert (dpt.asnumpy(r) == expected).all()

    ar3 = dpt.ones(sz, dtype=op_dtype)
    r2 = dpt.logical_not(ar3[::2])
    assert isinstance(r, dpt.usm_ndarray)

    expected = np.logical_not(np.ones(ar3.shape, dtype=op_dtype)[::2])
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert (dpt.asnumpy(r2) == expected).all()

    r3 = dpt.empty_like(r, dtype=r.dtype)
    dpt.logical_not(ar2[::-1], out=r3)
    assert (dpt.asnumpy(r) == dpt.asnumpy(r3)).all()


@pytest.mark.parametrize("op_dtype", ["c8", "c16"])
def test_logical_not_complex_matrix(op_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(op_dtype, q)

    sz = 127
    ar1_np_real = np.random.randint(0, 2, sz)
    ar1_np_imag = np.random.randint(0, 2, sz)
    ar1_np = ar1_np_real + 1j * ar1_np_imag
    ar1 = dpt.asarray(ar1_np, dtype=op_dtype)

    r = dpt.logical_not(ar1)
    expected = np.logical_not(ar1_np)
    assert _compare_dtypes(r.dtype, expected.dtype, sycl_queue=q)
    assert r.shape == expected.shape
    assert (dpt.asnumpy(r) == expected).all()

    r1 = dpt.logical_not(ar1[::-2])
    expected1 = np.logical_not(ar1_np[::-2])
    assert _compare_dtypes(r.dtype, expected1.dtype, sycl_queue=q)
    assert r1.shape == expected1.shape
    assert (dpt.asnumpy(r1) == expected1).all()

    ar2 = dpt.asarray(
        [
            2.0 + 0j,
            dpt.nan,
            dpt.nan * 1j,
            dpt.inf,
            dpt.inf * 1j,
            -dpt.inf,
            -dpt.inf * 1j,
        ],
        dtype=op_dtype,
    )
    ar2_np = dpt.asnumpy(ar2)
    r2 = dpt.logical_not(ar2)
    with np.errstate(invalid="ignore"):
        expected2 = np.logical_not(ar2_np)
    assert (dpt.asnumpy(r2) == expected2).all()


def test_logical_not_complex_float():
    get_queue_or_skip()

    ar1 = dpt.asarray([1j, 1.0 + 9j, 2.0 + 0j, 2.0 + 1j], dtype="c8")

    r = dpt.logical_not(ar1)
    expected = np.logical_not(dpt.asnumpy(ar1))
    assert (dpt.asnumpy(r) == expected).all()

    with np.errstate(invalid="ignore"):
        for tp in [
            dpt.nan,
            dpt.nan * 1j,
            dpt.inf,
            dpt.inf * 1j,
            -dpt.inf,
            -dpt.inf * 1j,
        ]:
            ar2 = dpt.full(ar1.shape, tp)
            r2 = dpt.logical_not(ar2)
            expected2 = np.logical_not(dpt.asnumpy(ar2))
            assert (dpt.asnumpy(r2) == expected2).all()


@pytest.mark.parametrize("op_usm_type", _usm_types)
def test_logical_not_usm_type_matrix(op_usm_type):
    get_queue_or_skip()

    sz = 128
    ar1 = dpt.asarray(
        np.random.randint(0, 2, sz), dtype="i4", usm_type=op_usm_type
    )

    r = dpt.logical_not(ar1)
    assert isinstance(r, dpt.usm_ndarray)
    assert r.usm_type == op_usm_type


def test_logical_not_order():
    get_queue_or_skip()

    ar1 = dpt.ones((20, 20), dtype="i4", order="C")
    r1 = dpt.logical_not(ar1, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.logical_not(ar1, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.logical_not(ar1, order="A")
    assert r3.flags.c_contiguous
    r4 = dpt.logical_not(ar1, order="K")
    assert r4.flags.c_contiguous

    ar1 = dpt.zeros((20, 20), dtype="i4", order="F")
    r1 = dpt.logical_not(ar1, order="C")
    assert r1.flags.c_contiguous
    r2 = dpt.logical_not(ar1, order="F")
    assert r2.flags.f_contiguous
    r3 = dpt.logical_not(ar1, order="A")
    assert r3.flags.f_contiguous
    r4 = dpt.logical_not(ar1, order="K")
    assert r4.flags.f_contiguous

    ar1 = dpt.ones((40, 40), dtype="i4", order="C")[:20, ::-2]
    r4 = dpt.logical_not(ar1, order="K")
    assert r4.strides == (20, -1)

    ar1 = dpt.zeros((40, 40), dtype="i4", order="C")[:20, ::-2].mT
    r4 = dpt.logical_not(ar1, order="K")
    assert r4.strides == (-1, 20)
