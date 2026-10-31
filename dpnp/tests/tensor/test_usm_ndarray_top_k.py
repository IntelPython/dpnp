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
from numpy.testing import assert_array_equal

import dpnp.tensor as dpt

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
    assert dpt.all(dpt.sort(s.indices) == dpt.sort(expected_inds)), (
        s.indices,
        expected_inds,
    )


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
    assert dpt.all(dpt.sort(s.indices) == dpt.sort(expected_inds)), (
        s.indices,
        expected_inds,
    )


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_nan(dtype, mode):
    # NaNs (and complex values with a NaN component) are ordered to the end
    # for both modes, so top_k excludes them until k reaches the NaN region
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    is_complex = dtype in ("c8", "c16")
    nan = complex(dpt.nan, dpt.nan) if is_complex else dpt.nan

    def isnan(a):
        if is_complex:
            return dpt.isnan(dpt.real(a)) | dpt.isnan(dpt.imag(a))
        return dpt.isnan(a)

    # 5 distinct finite values followed by 2 NaNs, then rolled to interleave
    x = dpt.roll(dpt.asarray([3, 1, 5, 2, 4, nan, nan], dtype=dtype), 3)

    # k within the finite region: NaNs are excluded from the result
    r = dpt.top_k(x, 3, mode=mode)
    assert not dpt.any(isnan(r.values))
    assert dpt.all(r.values == x[r.indices])
    extreme = [5, 4, 3] if mode == "largest" else [1, 2, 3]
    expected = dpt.asarray(extreme, dtype=dtype)
    assert dpt.all(dpt.sort(r.values) == dpt.sort(expected))

    # k reaching into the NaN region: all finite values are selected, NaNs
    # fill the rest (the elements may come in any order)
    for k, n_nans in [(6, 1), (7, 2)]:
        r = dpt.top_k(x, k, mode=mode)
        assert int(dpt.sum(isnan(r.values))) == n_nans


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


def _reference_top_k_inds(x_np, k, mode):
    "indices of the top k along the last axis, equal elements by lower index"
    if mode == "largest":
        n = x_np.shape[-1]
        inds = np.argsort(x_np[..., ::-1], axis=-1, kind="stable")[..., ::-1]
        inds = n - 1 - inds
        # NaNs order last for both modes, as with `descending=True` of
        # numpy>=2.5
        nans = np.isnan(np.take_along_axis(x_np, inds, axis=-1))
        inds = np.take_along_axis(
            inds, np.argsort(nans, axis=-1, kind="stable"), axis=-1
        )
    else:
        inds = np.argsort(x_np, axis=-1, kind="stable")
    return inds[..., :k]


def _check_top_k(r, x_np, k, mode, axis=-1):
    "the result is exact, but its elements may come in any order"
    x_np = np.moveaxis(x_np, axis, -1)
    inds = np.moveaxis(dpt.asnumpy(r.indices), axis, -1)
    vals = np.moveaxis(dpt.asnumpy(r.values), axis, -1)
    expected_inds = np.sort(_reference_top_k_inds(x_np, k, mode), axis=-1)
    order = np.argsort(inds, axis=-1)
    assert_array_equal(np.take_along_axis(inds, order, axis=-1), expected_inds)
    # the values are compared bitwise with the input, so that the sign of
    # zeros and the bits of NaNs are checked to be kept
    vals = np.take_along_axis(vals, order, axis=-1)
    expected_vals = np.take_along_axis(x_np, expected_inds, axis=-1)
    bits_dt = f"u{x_np.dtype.itemsize}"
    assert_array_equal(vals.view(bits_dt), expected_vals.view(bits_dt))


def _random_bits(dtype, shape, rng):
    "values of `dtype` with uniformly random bits, over its full range"
    dt = np.dtype(dtype)
    bits_info = np.iinfo(f"u{dt.itemsize}")
    bits = rng.integers(
        0, bits_info.max, size=shape, dtype=bits_info.dtype, endpoint=True
    )
    return bits.view(dt)


_all_real_dtypes = [
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
]


@pytest.mark.parametrize(
    "dtype",
    ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8"],
)
@pytest.mark.parametrize(
    "shape", [(3000, 5), (2000, 15), (1000, 50), (4, 3001), (2, 100003)]
)
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_ties(dtype, shape, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # few distinct values, so that the k-th value is repeated
    rng = np.random.default_rng(42)
    x_np = rng.integers(0, 4, size=shape).astype(dtype)
    x = dpt.asarray(x_np, sycl_queue=q)

    n = shape[-1]
    for k in sorted({1, min(n, 7), min(n, 300), max(1, n // 3), n}):
        r = dpt.top_k(x, k, axis=-1, mode=mode)
        _check_top_k(r, x_np, k, mode)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
@pytest.mark.parametrize("n", [11, 257, 100003])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_nans_and_signed_zeros(dtype, n, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # NaNs compare equal and order after other values, -0.0 == 0.0
    special = np.array(
        [np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf, 1.0, -1.0], dtype=dtype
    )
    rng = np.random.default_rng(7)
    x_np = rng.choice(special, size=n)
    x = dpt.asarray(x_np, sycl_queue=q)

    for k in sorted({1, min(n, 5), min(n, 200), n // 2, n}):
        r = dpt.top_k(x, k, mode=mode)
        _check_top_k(r, x_np, k, mode)


@pytest.mark.parametrize("n", [5, 30, 64, 128, 3001, 70000])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_nonstandard_bool_bytes(n, mode):
    # NumPy treats any non-zero byte of a bool as True, see gh-2121
    q = get_queue_or_skip()

    rng = np.random.default_rng(19)
    m = 3 if n > 1000 else 40
    raw_np = rng.choice(np.array([0, 1, 2, 127, 128, 255], dtype="u1"), (m, n))
    raw = dpt.asarray(raw_np, sycl_queue=q)
    x = dpt.usm_ndarray(raw.shape, dtype="?", buffer=raw.usm_data)
    x_np = raw_np != 0

    for k in sorted({1, min(n, 4), n // 2, n}):
        r = dpt.top_k(x, k, axis=-1, mode=mode)
        inds = dpt.asnumpy(r.indices)
        expected_inds = np.sort(_reference_top_k_inds(x_np, k, mode), axis=-1)
        assert_array_equal(np.sort(inds, axis=-1), expected_inds)
        vals = dpt.asnumpy(r.values).view("u1") != 0
        assert_array_equal(vals, np.take_along_axis(x_np, inds, axis=-1))


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


@pytest.mark.parametrize("dtype", ["i4", "u4", "i8", "f4", "f8"])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_long_rows(dtype, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # long rows of many distinct values, with repeats of some of them
    rng = np.random.default_rng(3)
    shape = (3, 300001)
    if np.dtype(dtype).kind == "f":
        x_np = rng.standard_normal(size=shape).astype(dtype)
    else:
        info = np.iinfo(dtype)
        x_np = rng.integers(info.min, info.max, size=shape, dtype=dtype)
    x_np[:, ::7] = x_np[:, 1:2]
    x_np[:, 5::11] = x_np[:, 2:3]
    x = dpt.asarray(x_np, sycl_queue=q)

    for k in [1, 10, 2000, 60000]:
        r = dpt.top_k(x, k, axis=-1, mode=mode)
        _check_top_k(r, x_np, k, mode)


# the row sizes at which the kernel changes depend on the sub-group sizes of
# the device: rows of up to 4 or 8 times the smallest one are ranked by a
# sub-group each, longer ones by a work-group, and from 65536 elements on by
# several work-groups when there are few rows
@pytest.mark.parametrize("dtype", _all_real_dtypes)
@pytest.mark.parametrize(
    "n", [8, 9, 16, 17, 32, 33, 64, 65, 128, 513, 2049, 65535, 65536, 65537]
)
@pytest.mark.parametrize("data", ["full_range", "ties"])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_dispatch_thresholds(dtype, n, data, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    rng = np.random.default_rng(11)
    m = 2 if n >= 65535 else max(2, 20000 // n)
    if data == "full_range":
        x_np = _random_bits(dtype, (m, n), rng)
    else:
        # few distinct values from over the full range, extremes included
        values = _random_bits(dtype, 4, rng)
        if np.dtype(dtype).kind == "f":
            values[:2] = [-np.inf, np.inf]
        else:
            info = np.iinfo(dtype)
            values[:2] = [info.min, info.max]
        x_np = rng.choice(values, size=(m, n))
    x = dpt.asarray(x_np, sycl_queue=q)

    for k in sorted({1, 7, n // 2, n - 1, n}):
        r = dpt.top_k(x, k, axis=-1, mode=mode)
        _check_top_k(r, x_np, k, mode)


def _degenerate_rows(dtype, n, rng):
    "rows of equal values, of the extremes only and of few finite values"
    dt = np.dtype(dtype)
    if dt.kind == "f":
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = np.iinfo(dt).min, np.iinfo(dt).max
    rows = [
        np.full(n, lo, dtype=dt),
        np.full(n, hi, dtype=dt),
        np.zeros(n, dtype=dt),
        rng.choice(np.array([lo, hi], dtype=dt), size=n),
    ]
    if dt.kind == "f":
        few_finite = np.full(n, np.nan, dtype=dt)
        few_finite[rng.choice(n, size=min(n, 3), replace=False)] = [1, -2, 0][
            : min(n, 3)
        ]
        rows += [
            np.full(n, np.nan, dtype=dt),
            rng.choice(np.array([0.0, -0.0], dtype=dt), size=n),
            few_finite,
        ]
    return np.stack(rows)


@pytest.mark.parametrize("dtype", _all_real_dtypes)
@pytest.mark.parametrize("n", [5, 30, 64, 128, 3001, 70000])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_degenerate_rows(dtype, n, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    rng = np.random.default_rng(5)
    rows = _degenerate_rows(dtype, n, rng)
    # many rows of each kind, so that their work-groups run concurrently
    reps = max(1, 50000 // rows.size)
    x_np = np.tile(rows, (reps, 1))
    x = dpt.asarray(x_np, sycl_queue=q)

    for k in sorted({1, 2, min(n, 4), n - 1, n}):
        r = dpt.top_k(x, k, axis=-1, mode=mode)
        _check_top_k(r, x_np, k, mode)


@pytest.mark.parametrize("dtype", ["i4", "f8"])
@pytest.mark.parametrize("n", [5, 50, 3001, 70000])
@pytest.mark.parametrize("layout", ["step", "reversed", "transposed", "axis0"])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_strided(dtype, n, layout, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    rng = np.random.default_rng(13)
    m = 3 if n > 1000 else 40
    if layout == "step":
        base_np = _random_bits(dtype, (m, 2 * n), rng)
        x_np, axis = base_np[:, ::2], -1
    elif layout == "reversed":
        base_np = _random_bits(dtype, (m, n), rng)
        x_np, axis = base_np[::-1, ::-1], -1
    elif layout == "transposed":
        base_np = _random_bits(dtype, (n, m), rng)
        x_np, axis = base_np.T, -1
    else:
        base_np = _random_bits(dtype, (n, m), rng)
        x_np, axis = base_np, 0

    base = dpt.asarray(base_np, sycl_queue=q)
    if layout == "step":
        x = base[:, ::2]
    elif layout == "reversed":
        x = base[::-1, ::-1]
    elif layout == "transposed":
        x = base.T
    else:
        x = base

    for k in sorted({1, min(n, 3), n // 2, n}):
        r = dpt.top_k(x, k, axis=axis, mode=mode)
        _check_top_k(r, x_np, k, mode, axis=axis)


@pytest.mark.parametrize("dtype", ["i4", "u4", "i8", "u8", "f4", "f8"])
@pytest.mark.parametrize("run_fraction", [0.01, 0.1])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_top_k_long_rows_tie_run(dtype, run_fraction, mode):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # few long rows are split over several work-groups; when the elements
    # sharing the leading digit of the k-th one are few, as here with a run
    # of 1% equal values containing it, they are listed for the later passes
    # to go over, and with a run of 10% they are not
    rng = np.random.default_rng(17)
    n = (1 << 18) + 3
    row = _random_bits(dtype, n, rng)
    if np.dtype(dtype).kind == "f":
        row[np.isnan(row)] = 0
    v = row[0]
    run_len = int(n * run_fraction)
    row[1000 : 1000 + run_len] = v
    # the rows are permutations of each other, so that their k-th elements
    # are the same
    x_np = np.stack([row, rng.permutation(row)])
    x = dpt.asarray(x_np, sycl_queue=q)

    n_better = int(((row > v) if mode == "largest" else (row < v)).sum())
    for k in sorted({n_better + run_len // 2, n // 3}):
        r = dpt.top_k(x, k, axis=-1, mode=mode)
        _check_top_k(r, x_np, k, mode)
