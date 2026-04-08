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
import pytest

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


@pytest.mark.parametrize("arg_dtype", _all_dtypes)
def test_sum_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    # test reduction for C-contiguous input
    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.sum(m)

    assert isinstance(r, dpt.usm_ndarray)
    if m.dtype.kind == "i":
        assert r.dtype.kind == "i"
    elif m.dtype.kind == "u":
        assert r.dtype.kind == "u"
    elif m.dtype.kind == "f":
        assert r.dtype.kind == "f"
    elif m.dtype.kind == "c":
        assert r.dtype.kind == "c"

    assert dpt.all(r == 100)

    # test reduction for strided input
    m = dpt.ones(200, dtype=arg_dtype)[:1:-2]
    r = dpt.sum(m)
    assert dpt.all(r == 99)

    # test reduction for strided input which can be simplified
    # to contiguous computation
    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.sum(dpt.flip(m))
    assert dpt.all(r == 100)


@pytest.mark.parametrize("arg_dtype", _all_dtypes)
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_sum_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.sum(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
    assert dpt.all(r == 100)


def test_sum_empty():
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="u1")
    y = dpt.sum(x)
    assert y.shape == ()
    assert int(y) == 0


def test_sum_axis():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.sum(m, axis=(1, 2, -1))

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 6)
    assert dpt.all(s == dpt.asarray(4 * 5 * 7, dtype="i4"))


def test_sum_keepdims():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.sum(m, axis=(1, 2, -1), keepdims=True)

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 1, 1, 6, 1)
    assert dpt.all(s == dpt.asarray(4 * 5 * 7, dtype=s.dtype))


def test_sum_scalar():
    get_queue_or_skip()

    m = dpt.ones(())
    s = dpt.sum(m)

    assert isinstance(s, dpt.usm_ndarray)
    assert m.sycl_queue == s.sycl_queue
    assert s.shape == ()
    assert s == dpt.full((), 1)


@pytest.mark.parametrize("arg_dtype", _all_dtypes)
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_sum_arg_out_dtype_scalar(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    m = dpt.ones((), dtype=arg_dtype)
    r = dpt.sum(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
    assert r == 1


def test_sum_keepdims_zero_size():
    """See gh-1293"""
    get_queue_or_skip()
    n = 10
    a = dpt.ones((n, 0, n))

    s1 = dpt.sum(a, keepdims=True)
    assert s1.shape == (1, 1, 1)

    s2 = dpt.sum(a, axis=(0, 1), keepdims=True)
    assert s2.shape == (1, 1, n)

    s3 = dpt.sum(a, axis=(1, 2), keepdims=True)
    assert s3.shape == (n, 1, 1)

    s4 = dpt.sum(a, axis=(0, 2), keepdims=True)
    assert s4.shape == (1, 0, 1)

    a0 = a[0]
    s5 = dpt.sum(a0, keepdims=True)
    assert s5.shape == (1, 1)


@pytest.mark.parametrize("arg_dtype", ["i8", "f4", "c8"])
@pytest.mark.parametrize("n", [1023, 1024, 1025])
def test_largish_reduction(arg_dtype, n):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    m = 5
    x = dpt.ones((m, n, m), dtype=arg_dtype)

    y1 = dpt.sum(x, axis=(0, 1))
    y2 = dpt.sum(x, axis=(1, 2))

    assert dpt.all(dpt.equal(y1, y2))
    assert dpt.all(dpt.equal(y1, n * m))


@pytest.mark.parametrize("n", [1023, 1024, 1025])
def test_largish_reduction_axis1_axis0(n):
    get_queue_or_skip()

    m = 25
    x1 = dpt.ones((m, n), dtype="f4")
    x2 = dpt.ones((n, m), dtype="f4")

    y1 = dpt.sum(x1, axis=1)
    y2 = dpt.sum(x2, axis=0)

    assert dpt.all(y1 == n)
    assert dpt.all(y2 == n)


def test_axis0_bug():
    "gh-1391"
    get_queue_or_skip()

    sh = (1, 2, 3)
    a = dpt.arange(sh[0] * sh[1] * sh[2], dtype="i4")
    a = dpt.reshape(a, sh)
    aT = dpt.permute_dims(a, (2, 1, 0))

    s = dpt.sum(aT, axis=2)
    expected = dpt.asarray([[0, 3], [1, 4], [2, 5]])

    assert dpt.all(s == expected)


def test_sum_axis1_axis0():
    """See gh-1455"""
    get_queue_or_skip()

    # The atomic case is checked in `test_usm_ndarray_reductions`
    # This test checks the tree reduction path for correctness
    x = dpt.reshape(dpt.arange(3 * 4 * 5, dtype="f4"), (3, 4, 5))

    m = dpt.sum(x, axis=0)
    expected = dpt.asarray(
        [
            [60, 63, 66, 69, 72],
            [75, 78, 81, 84, 87],
            [90, 93, 96, 99, 102],
            [105, 108, 111, 114, 117],
        ],
        dtype="f4",
    )
    tol = dpt.finfo(m.dtype).resolution
    assert dpt.allclose(m, expected, atol=tol, rtol=tol)

    x = dpt.flip(x, axis=2)
    m = dpt.sum(x, axis=2)
    expected = dpt.asarray(
        [[10, 35, 60, 85], [110, 135, 160, 185], [210, 235, 260, 285]],
        dtype="f4",
    )
    assert dpt.allclose(m, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("arg_dtype", _all_dtypes[1:])
def test_prod_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    arg_dtype = dpt.dtype(arg_dtype)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.prod(m)

    assert isinstance(r, dpt.usm_ndarray)
    if m.dtype.kind == "i":
        assert r.dtype.kind == "i"
    elif m.dtype.kind == "u":
        assert r.dtype.kind == "u"
    elif m.dtype.kind == "f":
        assert r.dtype.kind == "f"
    elif m.dtype.kind == "c":
        assert r.dtype.kind == "c"
    assert dpt.all(r == 1)

    if dpt.isdtype(m.dtype, "unsigned integer"):
        m = dpt.tile(dpt.arange(1, 3, dtype=arg_dtype), 10)[:1:-2]
        r = dpt.prod(m)
        assert dpt.all(r == dpt.asarray(512, dtype=r.dtype))
    else:
        m = dpt.full(200, -1, dtype=arg_dtype)[:1:-2]
        r = dpt.prod(m)
        assert dpt.all(r == dpt.asarray(-1, dtype=r.dtype))


def test_prod_empty():
    get_queue_or_skip()
    x = dpt.empty((0,), dtype="u1")
    y = dpt.prod(x)
    assert y.shape == ()
    assert int(y) == 1


def test_prod_axis():
    get_queue_or_skip()

    m = dpt.ones((3, 4, 5, 6, 7), dtype="i4")
    s = dpt.prod(m, axis=(1, 2, -1))

    assert isinstance(s, dpt.usm_ndarray)
    assert s.shape == (3, 6)
    assert dpt.all(s == dpt.asarray(1, dtype="i4"))


@pytest.mark.parametrize("arg_dtype", _all_dtypes)
@pytest.mark.parametrize("out_dtype", _all_dtypes[1:])
def test_prod_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    out_dtype = dpt.dtype(out_dtype)
    arg_dtype = dpt.dtype(arg_dtype)

    m = dpt.ones(100, dtype=arg_dtype)
    r = dpt.prod(m, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
    assert dpt.all(r == 1)


def test_gh_1468():
    "See https://github.com/IntelPython/dpctl/issues/1468"
    get_queue_or_skip()

    a = dpt.full((2, 3, 4), 123456789, dtype=dpt.int32)
    t = dpt.sum(a, dtype="f4")
    assert t > 0


@pytest.mark.parametrize(
    "dt", ["i1", "i2", "i4", "i8", "f2", "f4", "f8", "c8", "c16"]
)
def test_gh_1944(dt):
    "See https://github.com/IntelPython/dpctl/issues/1944"
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)
    x = dpt.asarray([-1, 1], dtype=dpt.dtype(dt), sycl_queue=q)
    r = dpt.sum(x, dtype="?")
    # reduction must be performed in the requested dtype
    # if performed in the input type, result is False
    assert r
