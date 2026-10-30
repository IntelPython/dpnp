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

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import dpnp.tensor as dpt
from dpnp.tests.helper import numpy_version

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


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
def test_sort_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    inp = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )

    s = dpt.sort(inp, descending=False)
    assert dpt.all(s[:-1] <= s[1:])

    s1 = dpt.sort(inp, descending=True)
    assert dpt.all(s1[:-1] >= s1[1:])


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
def test_sort_2d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    fl = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )
    inp = dpt.reshape(fl, (20, -1))

    s = dpt.sort(inp, axis=1, descending=False)
    assert dpt.all(s[:, :-1] <= s[:, 1:])

    s1 = dpt.sort(inp, axis=1, descending=True)
    assert dpt.all(s1[:, :-1] >= s1[:, 1:])


def test_sort_strides():
    get_queue_or_skip()

    fl = dpt.roll(
        dpt.concat((dpt.ones(10000, dtype="i4"), dpt.zeros(10000, dtype="i4"))),
        734,
    )
    inp = dpt.reshape(fl, (-1, 20))

    s = dpt.sort(inp, axis=0, descending=False)
    assert dpt.all(s[:-1, :] <= s[1:, :])

    s1 = dpt.sort(inp, axis=0, descending=True)
    assert dpt.all(s1[:-1, :] >= s1[1:, :])


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
def test_argsort_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    inp = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )

    s_idx = dpt.argsort(inp, descending=False)
    assert dpt.all(inp[s_idx[:-1]] <= inp[s_idx[1:]])

    s1_idx = dpt.argsort(inp, descending=True)
    assert dpt.all(inp[s1_idx[:-1]] >= inp[s1_idx[1:]])


def test_sort_validation():
    with pytest.raises(TypeError):
        dpt.sort(dict())


def test_sort_validation_kind():
    get_queue_or_skip()

    x = dpt.ones(128, dtype="u1")

    with pytest.raises(ValueError):
        dpt.sort(x, kind=Ellipsis)

    with pytest.raises(ValueError):
        dpt.sort(x, kind="invalid")


def test_argsort_validation():
    with pytest.raises(TypeError):
        dpt.argsort(dict())


def test_argsort_validation_kind():
    get_queue_or_skip()

    x = dpt.arange(127, stop=0, step=-1, dtype="i1")

    with pytest.raises(ValueError):
        dpt.argsort(x, kind=Ellipsis)

    with pytest.raises(ValueError):
        dpt.argsort(x, kind="invalid")


_all_kinds = ["stable", "mergesort", "radixsort", "bitonicsort"]


@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_axis0(kind):
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    s = dpt.sort(x, axis=0, kind=kind)

    assert dpt.all(s[:-1, :] <= s[1:, :])


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_axis0(kind):
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    idx = dpt.argsort(x, axis=0, kind=kind)

    s = dpt.take_along_axis(x, idx, axis=0)

    assert dpt.all(s[:-1, :] <= s[1:, :])


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_axis1(kind):
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    idx = dpt.argsort(x, axis=1, kind=kind)

    s = dpt.take_along_axis(x, idx, axis=1)

    assert dpt.all(s[:, :-1] <= s[:, 1:])


@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_strided(kind):
    get_queue_or_skip()

    x_orig = dpt.arange(100, dtype="i4")
    x_flipped = dpt.flip(x_orig, axis=0)
    s = dpt.sort(x_flipped, kind=kind)

    assert dpt.all(s == x_orig)


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_strided(kind):
    get_queue_or_skip()

    x_orig = dpt.arange(100, dtype="i4")
    x_flipped = dpt.flip(x_orig, axis=0)
    idx = dpt.argsort(x_flipped, kind=kind)
    s = dpt.take_along_axis(x_flipped, idx, axis=0)

    assert dpt.all(s == x_orig)


@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_0d_array(kind):
    get_queue_or_skip()

    x = dpt.asarray(1, dtype="i4")
    expected = dpt.asarray(1, dtype="i4")
    assert dpt.sort(x, kind=kind) == expected


@pytest.mark.parametrize("kind", _all_kinds)
def test_argsort_0d_array(kind):
    get_queue_or_skip()

    x = dpt.asarray(1, dtype="i4")
    expected = dpt.asarray(0, dtype="i4")
    assert dpt.argsort(x, kind=kind) == expected


@pytest.mark.parametrize(
    "dtype",
    [
        "f2",
        "f4",
        "f8",
    ],
)
@pytest.mark.parametrize("kind", _all_kinds)
def test_sort_real_fp_nan(dtype, kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.asarray(
        [-0.0, 0.1, dpt.nan, 0.0, -0.1, dpt.nan, 0.2, -0.3], dtype=dtype
    )
    s = dpt.sort(x, kind=kind)

    expected = dpt.asarray(
        [-0.3, -0.1, -0.0, 0.0, 0.1, 0.2, dpt.nan, dpt.nan], dtype=dtype
    )

    assert dpt.allclose(s, expected, equal_nan=True)

    s = dpt.sort(x, descending=True, kind=kind)

    # NaNs sort to the end for descending order too matching NumPy
    expected = dpt.asarray(
        [0.2, 0.1, -0.0, 0.0, -0.1, -0.3, dpt.nan, dpt.nan], dtype=dtype
    )

    assert dpt.allclose(s, expected, equal_nan=True)


@pytest.mark.parametrize(
    "dtype",
    [
        "c8",
        "c16",
    ],
)
def test_sort_complex_fp_nan(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    rvs = [-0.0, 0.1, 0.0, 0.2, -0.3, dpt.nan]
    ivs = [-0.0, 0.1, 0.0, 0.2, -0.3, dpt.nan]

    cv = []
    for rv in rvs:
        for iv in ivs:
            cv.append(complex(rv, iv))

    inp = dpt.asarray(cv, dtype=dtype)
    s = dpt.sort(inp)

    expected = np.sort(dpt.asnumpy(inp))

    assert np.allclose(dpt.asnumpy(s), expected, equal_nan=True)

    pairs = []
    for i, j in itertools.permutations(range(inp.shape[0]), 2):
        pairs.append([i, j])
    sub_arrs = inp[dpt.asarray(pairs)]
    m1 = dpt.asnumpy(dpt.sort(sub_arrs, axis=1))
    m2 = np.sort(dpt.asnumpy(sub_arrs), axis=1)
    for k in range(len(pairs)):
        i, j = pairs[k]
        r1 = m1[k]
        r2 = m2[k]
        assert np.array_equal(
            r1.view(np.int64), r2.view(np.int64)
        ), f"Failed for {i} and {j}"

    # complex values with a NaN component sort to the end for descending
    # order too, matching NumPy (`descending` requires numpy>=2.5)
    if numpy_version() >= "2.5.0":
        s = dpt.sort(inp, descending=True)
        expected = np.sort(dpt.asnumpy(inp), descending=True)
        assert np.allclose(dpt.asnumpy(s), expected, equal_nan=True)

        m1 = dpt.asnumpy(dpt.sort(sub_arrs, axis=1, descending=True))
        m2 = np.sort(dpt.asnumpy(sub_arrs), axis=1, descending=True)
        for k in range(len(pairs)):
            i, j = pairs[k]
            assert np.array_equal(
                m1[k].view(np.int64), m2[k].view(np.int64)
            ), f"Failed for {i} and {j}"


def test_radix_sort_size_1_axis():
    get_queue_or_skip()

    x1 = dpt.ones((), dtype="i1")
    r1 = dpt.sort(x1, kind="radixsort")
    assert_array_equal(dpt.asnumpy(r1), dpt.asnumpy(x1))

    x2 = dpt.ones([1], dtype="i1")
    r2 = dpt.sort(x2, kind="radixsort")
    assert_array_equal(dpt.asnumpy(r2), dpt.asnumpy(x2))

    x3 = dpt.reshape(dpt.arange(10, dtype="i1"), (10, 1))
    r3 = dpt.sort(x3, kind="radixsort")
    assert dpt.asnumpy(r3 == x3).all()

    x4 = dpt.reshape(dpt.arange(10, dtype="i1"), (1, 10))
    r4 = dpt.sort(x4, axis=0, kind="radixsort")
    assert dpt.asnumpy(r4 == x4).all()


def test_radix_argsort_size_1_axis():
    get_queue_or_skip()

    x1 = dpt.ones((), dtype="i1")
    r1 = dpt.argsort(x1, kind="radixsort")
    assert r1 == 0

    x2 = dpt.ones([1], dtype="i1")
    r2 = dpt.argsort(x2, kind="radixsort")
    assert dpt.asnumpy(r2 == 0).all()

    x3 = dpt.reshape(dpt.arange(10, dtype="i1"), (10, 1))
    r3 = dpt.argsort(x3, kind="radixsort")
    assert dpt.asnumpy(r3 == 0).all()

    x4 = dpt.reshape(dpt.arange(10, dtype="i1"), (1, 10))
    r4 = dpt.argsort(x4, axis=0, kind="radixsort")
    assert dpt.asnumpy(r4 == 0).all()


def test_bitonic_sort_long_rows():
    get_queue_or_skip()

    # a row too long to be held in the local memory of the device exercises
    # the global memory path of the bitonic network, and a length which is not
    # a power of two exercises the padding of the network
    n = 70001
    x = dpt.asarray(np.random.default_rng(42).random((2, n), dtype=np.float32))

    s = dpt.sort(x, kind="bitonicsort")
    assert_array_equal(dpt.asnumpy(s), np.sort(dpt.asnumpy(x), axis=-1))

    idx = dpt.argsort(x, kind="bitonicsort")
    assert_array_equal(
        dpt.asnumpy(dpt.take_along_axis(x, idx, axis=-1)),
        np.sort(dpt.asnumpy(x), axis=-1),
    )


@pytest.mark.parametrize("dtype", ["i1", "u2", "i4", "f4", "f8"])
@pytest.mark.parametrize("descending", [False, True])
def test_bitonic_sort_extremal_values(dtype, descending):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    # rows are padded up to a power of two length with a value the comparator
    # of the network orders last, so inputs which contain that value must be
    # sorted correctly all the same
    dt = dpt.dtype(dtype)
    if dt.kind == "f":
        vals = [np.nan, np.inf, -np.inf, 0.0, 1.0, np.nan, np.inf]
    else:
        info = dpt.iinfo(dt)
        vals = [info.max, info.min, 0, 1, info.max, info.min, 3]
    x = dpt.asarray(np.asarray(vals, dtype=dt), sycl_queue=q)

    s = dpt.sort(x, descending=descending, kind="bitonicsort")
    expected = np.sort(dpt.asnumpy(x))
    if descending:
        # NaNs are sorted to the end for descending order too
        nans = np.isnan(expected)
        expected = np.concatenate((np.flip(expected[~nans]), expected[nans]))
    assert np.array_equal(dpt.asnumpy(s), expected, equal_nan=True)

    idx = dpt.argsort(x, descending=descending, kind="bitonicsort")
    assert np.array_equal(
        dpt.asnumpy(dpt.take(x, idx)), expected, equal_nan=True
    )
