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

import pytest

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


def _expected_largest_inds(inp, n, shift, k):
    "Computed expected top_k indices for mode='largest'"
    assert k < n
    ones_start_id = shift % (2 * n)

    alloc_dev = inp.device

    if ones_start_id < n:
        expected_inds = dpt.arange(
            ones_start_id, ones_start_id + k, dtype="i8", device=alloc_dev
        )
    else:
        # wrap-around
        ones_end_id = (ones_start_id + n) % (2 * n)
        if ones_end_id >= k:
            expected_inds = dpt.arange(k, dtype="i8", device=alloc_dev)
        else:
            expected_inds = dpt.concat(
                (
                    dpt.arange(ones_end_id, dtype="i8", device=alloc_dev),
                    dpt.arange(
                        ones_start_id,
                        ones_start_id + k - ones_end_id,
                        dtype="i8",
                        device=alloc_dev,
                    ),
                )
            )

    return expected_inds


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
@pytest.mark.parametrize("n", [33, 43, 255, 511, 1021, 8193])
def test_top_k_1d_largest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shift, k = 734, 5
    o = dpt.ones(n, dtype=dtype)
    z = dpt.zeros(n, dtype=dtype)
    oz = dpt.concat((o, z))
    inp = dpt.roll(oz, shift)

    expected_inds = _expected_largest_inds(oz, n, shift, k)

    s = dpt.top_k(inp, k, mode="largest")
    assert s.values.shape == (k,)
    assert s.values.dtype == inp.dtype
    assert s.indices.shape == (k,)
    assert dpt.all(s.values == dpt.ones(k, dtype=dtype)), s.values
    assert dpt.all(s.values == inp[s.indices]), s.indices
    assert dpt.all(s.indices == expected_inds), (s.indices, expected_inds)


def _expected_smallest_inds(inp, n, shift, k):
    "Computed expected top_k indices for mode='smallest'"
    assert k < n
    zeros_start_id = (n + shift) % (2 * n)
    zeros_end_id = (shift) % (2 * n)

    alloc_dev = inp.device

    if zeros_start_id < zeros_end_id:
        expected_inds = dpt.arange(
            zeros_start_id, zeros_start_id + k, dtype="i8", device=alloc_dev
        )
    else:
        if zeros_end_id >= k:
            expected_inds = dpt.arange(k, dtype="i8", device=alloc_dev)
        else:
            expected_inds = dpt.concat(
                (
                    dpt.arange(zeros_end_id, dtype="i8", device=alloc_dev),
                    dpt.arange(
                        zeros_start_id,
                        zeros_start_id + k - zeros_end_id,
                        dtype="i8",
                        device=alloc_dev,
                    ),
                )
            )

    return expected_inds


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
@pytest.mark.parametrize("n", [37, 39, 61, 255, 257, 513, 1021, 8193])
def test_top_k_1d_smallest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shift, k = 734, 5
    o = dpt.ones(n, dtype=dtype)
    z = dpt.zeros(n, dtype=dtype)
    oz = dpt.concat((o, z))
    inp = dpt.roll(oz, shift)

    expected_inds = _expected_smallest_inds(oz, n, shift, k)

    s = dpt.top_k(inp, k, mode="smallest")
    assert s.values.shape == (k,)
    assert s.values.dtype == inp.dtype
    assert s.indices.shape == (k,)
    assert dpt.all(s.values == dpt.zeros(k, dtype=dtype)), s.values
    assert dpt.all(s.values == inp[s.indices]), s.indices
    assert dpt.all(s.indices == expected_inds), (s.indices, expected_inds)


@pytest.mark.parametrize(
    "dtype",
    [
        # skip short types to ensure that m*n can be represented
        # in the type
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
@pytest.mark.parametrize("n", [37, 39, 61, 255, 257, 513, 1021, 8193])
def test_top_k_2d_largest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m, k = 8, 3
    if dtype == "f2" and m * n > 2000:
        pytest.skip(
            "f2 can not distinguish between large integers used in this test"
        )

    x = dpt.reshape(dpt.arange(m * n, dtype=dtype), (m, n))

    r = dpt.top_k(x, k, axis=1)

    assert r.values.shape == (m, k)
    assert r.indices.shape == (m, k)
    expected_inds = dpt.reshape(dpt.arange(n, dtype=r.indices.dtype), (1, n))[
        :, -k:
    ]
    assert expected_inds.shape == (1, k)
    assert dpt.all(
        dpt.sort(r.indices, axis=1) == dpt.sort(expected_inds, axis=1)
    ), (r.indices, expected_inds)
    expected_vals = x[:, -k:]
    assert dpt.all(
        dpt.sort(r.values, axis=1) == dpt.sort(expected_vals, axis=1)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        # skip short types to ensure that m*n can be represented
        # in the type
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
@pytest.mark.parametrize("n", [37, 39, 61, 255, 257, 513, 1021, 8193])
def test_top_k_2d_smallest(dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    m, k = 8, 3
    if dtype == "f2" and m * n > 2000:
        pytest.skip(
            "f2 can not distinguish between large integers used in this test"
        )

    x = dpt.reshape(dpt.arange(m * n, dtype=dtype), (m, n))

    r = dpt.top_k(x, k, axis=1, mode="smallest")

    assert r.values.shape == (m, k)
    assert r.indices.shape == (m, k)
    expected_inds = dpt.reshape(dpt.arange(n, dtype=r.indices.dtype), (1, n))[
        :, :k
    ]
    assert dpt.all(
        dpt.sort(r.indices, axis=1) == dpt.sort(expected_inds, axis=1)
    )
    assert dpt.all(dpt.sort(r.values, axis=1) == dpt.sort(x[:, :k], axis=1))


def test_top_k_0d():
    get_queue_or_skip()

    a = dpt.ones(tuple(), dtype="i4")
    assert a.ndim == 0
    assert a.size == 1

    r = dpt.top_k(a, 1)
    assert r.values == a
    assert r.indices == dpt.zeros_like(a, dtype=r.indices.dtype)


def test_top_k_noncontig():
    get_queue_or_skip()

    a = dpt.arange(256, dtype=dpt.int32)[::2]
    r = dpt.top_k(a, 3)

    assert dpt.all(dpt.sort(r.values) == dpt.asarray([250, 252, 254])), r.values
    assert dpt.all(
        dpt.sort(r.indices) == dpt.asarray([125, 126, 127])
    ), r.indices


def test_top_k_axis0():
    get_queue_or_skip()

    m, n, k = 128, 8, 3
    x = dpt.reshape(dpt.arange(m * n, dtype=dpt.int32), (m, n))

    r = dpt.top_k(x, k, axis=0, mode="smallest")
    assert r.values.shape == (k, n)
    assert r.indices.shape == (k, n)
    expected_inds = dpt.reshape(dpt.arange(m, dtype=r.indices.dtype), (m, 1))[
        :k, :
    ]
    assert dpt.all(
        dpt.sort(r.indices, axis=0) == dpt.sort(expected_inds, axis=0)
    )
    assert dpt.all(dpt.sort(r.values, axis=0) == dpt.sort(x[:k, :], axis=0))


def test_top_k_validation():
    get_queue_or_skip()
    x = dpt.ones(10, dtype=dpt.int64)
    with pytest.raises(ValueError):
        # k must be positive
        dpt.top_k(x, -1)
    with pytest.raises(TypeError):
        # argument should be usm_ndarray
        dpt.top_k(list(), 2)
    x2 = dpt.reshape(x, (2, 5))
    with pytest.raises(ValueError):
        # k must not exceed array dimension
        # along specified axis
        dpt.top_k(x2, 100, axis=1)
    with pytest.raises(ValueError):
        # for 0d arrays, k must be 1
        dpt.top_k(x[0], 2)
    with pytest.raises(ValueError):
        # mode must be "largest", or "smallest"
        dpt.top_k(x, 2, mode="invalid")
