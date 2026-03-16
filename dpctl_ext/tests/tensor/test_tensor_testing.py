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

import pytest

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

_all_dtypes = [
    "?",
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_allclose(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    a1 = dpt.ones(10, dtype=dtype)
    a2 = dpt.ones(10, dtype=dtype)

    assert dpt.allclose(a1, a2)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_allclose_real_fp(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    v = [dpt.nan, -dpt.nan, dpt.inf, -dpt.inf, -0.0, 0.0, 1.0, -1.0]
    a1 = dpt.asarray(v[2:], dtype=dtype)
    a2 = dpt.asarray(v[2:], dtype=dtype)

    tol = dpt.finfo(a1.dtype).resolution
    assert dpt.allclose(a1, a2, atol=tol, rtol=tol)

    a1 = dpt.asarray(v, dtype=dtype)
    a2 = dpt.asarray(v, dtype=dtype)

    assert not dpt.allclose(a1, a2, atol=tol, rtol=tol)
    assert dpt.allclose(a1, a2, atol=tol, rtol=tol, equal_nan=True)


@pytest.mark.parametrize("dtype", ["c8", "c16"])
def test_allclose_complex_fp(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    v = [dpt.nan, -dpt.nan, dpt.inf, -dpt.inf, -0.0, 0.0, 1.0, -1.0]

    not_nans = [complex(*xy) for xy in itertools.product(v[2:], repeat=2)]
    z1 = dpt.asarray(not_nans, dtype=dtype)
    z2 = dpt.asarray(not_nans, dtype=dtype)

    tol = dpt.finfo(z1.dtype).resolution
    assert dpt.allclose(z1, z2, atol=tol, rtol=tol)

    both = [complex(*xy) for xy in itertools.product(v, repeat=2)]
    z1 = dpt.asarray(both, dtype=dtype)
    z2 = dpt.asarray(both, dtype=dtype)

    tol = dpt.finfo(z1.dtype).resolution
    assert not dpt.allclose(z1, z2, atol=tol, rtol=tol)
    assert dpt.allclose(z1, z2, atol=tol, rtol=tol, equal_nan=True)


def test_allclose_validation():
    with pytest.raises(TypeError):
        dpt.allclose(True, False)

    get_queue_or_skip()
    x = dpt.asarray(True)
    with pytest.raises(TypeError):
        dpt.allclose(x, False)


def test_allclose_type_promotion():
    get_queue_or_skip()

    x1 = dpt.ones(10, dtype="i4")
    x2 = dpt.ones(10, dtype="i8")

    assert dpt.allclose(x1, x2)


def test_allclose_tolerance():
    get_queue_or_skip()

    x = dpt.zeros(10, dtype="f4")
    atol = 1e-5
    y = dpt.full_like(x, atol)
    assert dpt.allclose(x, y, atol=atol, rtol=0)

    # about 8e-6
    tol = float.fromhex("0x1.0p-17")
    x = dpt.ones(10, dtype="f4")
    y = x - tol
    assert dpt.allclose(x, y, atol=0, rtol=tol)


def test_allclose_real_fp_early_exists():
    get_queue_or_skip()

    x1 = dpt.asarray([0.0, dpt.inf, -dpt.inf], dtype="f4")
    x2 = dpt.asarray([dpt.inf, 0.0, -dpt.inf], dtype="f4")

    # early exists, inf positions are different
    assert not dpt.allclose(x1, x2)

    x2 = dpt.asarray([0.0, -dpt.inf, dpt.inf], dtype="f4")

    # early exists, inf positions are the same, but signs differ
    assert not dpt.allclose(x1, x2)


def test_allclose_complex_fp_early_exists():
    get_queue_or_skip()

    x1 = dpt.asarray([0.0, dpt.inf, -dpt.inf], dtype="c8")
    x2 = dpt.asarray([dpt.inf, 0.0, -dpt.inf], dtype="c8")

    # early exists, inf positions of real parts are different
    assert not dpt.allclose(x1, x2)

    x2 = dpt.asarray([0.0, -dpt.inf, dpt.inf], dtype="c8")

    # early exists, inf positions of real parts are the same, but signs differ
    assert not dpt.allclose(x1, x2)

    x1 = dpt.asarray([0.0, dpt.inf * 1j, -dpt.inf * 1j], dtype="c8")
    x2 = dpt.asarray([dpt.inf * 1j, 0.0, -dpt.inf * 1j], dtype="c8")

    # early exists, inf positions of imag parts are different
    assert not dpt.allclose(x1, x2)

    x2 = dpt.asarray([0.0, -dpt.inf * 1j, dpt.inf * 1j], dtype="c8")
    assert not dpt.allclose(x1, x2)
