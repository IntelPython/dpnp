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

from random import randrange

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import pytest
from dpctl.utils import ExecutionPlacementError

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

sint_types = [
    dpt.int8,
    dpt.int16,
    dpt.int32,
    dpt.int64,
]
uint_types = [
    dpt.uint8,
    dpt.uint16,
    dpt.uint32,
    dpt.uint64,
]
rfp_types = [
    dpt.float16,
    dpt.float32,
    dpt.float64,
]
cfp_types = [
    dpt.complex64,
    dpt.complex128,
]

no_complex_types = [dpt.bool] + sint_types + uint_types + rfp_types

all_types = [dpt.bool] + sint_types + uint_types + rfp_types + cfp_types


@pytest.mark.parametrize("dt", sint_types)
def test_contig_cumsum_sint(dt):
    get_queue_or_skip()
    n = 10000
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), n)

    res = dpt.cumulative_sum(x, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)


@pytest.mark.parametrize("dt", sint_types)
def test_strided_cumsum_sint(dt):
    get_queue_or_skip()
    n = 10000
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), 2 * n)[1::2]

    res = dpt.cumulative_sum(x, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)

    x2 = dpt.repeat(dpt.asarray([-1, 1], dtype=dt), 2 * n)[-1::-2]

    res = dpt.cumulative_sum(x2, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)


@pytest.mark.parametrize("dt", sint_types)
def test_contig_cumsum_axis_sint(dt):
    get_queue_or_skip()
    n0, n1 = 1000, 173
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), n0)
    m = dpt.tile(dpt.expand_dims(x, axis=1), (1, n1))

    res = dpt.cumulative_sum(m, dtype=dt, axis=0)

    ar = dpt.arange(n0, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == dpt.expand_dims(expected, axis=1))


@pytest.mark.parametrize("dt", sint_types)
def test_strided_cumsum_axis_sint(dt):
    get_queue_or_skip()
    n0, n1 = 1000, 173
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), 2 * n0)
    m = dpt.tile(dpt.expand_dims(x, axis=1), (1, n1))[1::2, ::-1]

    res = dpt.cumulative_sum(m, dtype=dt, axis=0)

    ar = dpt.arange(n0, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == dpt.expand_dims(expected, axis=1))


def test_accumulate_scalar():
    get_queue_or_skip()

    s = dpt.asarray(1, dtype="i8")
    r = dpt.cumulative_sum(s)
    assert r == s
    assert r.ndim == s.ndim

    r = dpt.cumulative_sum(s, include_initial=True)
    r_expected = dpt.asarray([0, 1], dtype="i8")
    assert dpt.all(r == r_expected)


def test_cumulative_sum_include_initial():
    get_queue_or_skip()

    n0, n1 = 3, 5
    x = dpt.ones((n0, n1), dtype="i4")
    r = dpt.cumulative_sum(x, axis=0, include_initial=True)
    assert dpt.all(r[0, :] == 0)

    r = dpt.cumulative_sum(x, axis=1, include_initial=True)
    assert dpt.all(r[:, 0] == 0)

    x = dpt.ones(n1, dtype="i4")
    r = dpt.cumulative_sum(x, include_initial=True)
    assert r.shape == (n1 + 1,)
    assert r[0] == 0


def test_cumulative_prod_identity():
    get_queue_or_skip()

    x = dpt.zeros(5, dtype="i4")
    r = dpt.cumulative_prod(x, include_initial=True)
    assert r[0] == 1


def test_cumulative_logsumexp_identity():
    get_queue_or_skip()

    x = dpt.ones(5, dtype="f4")
    r = dpt.cumulative_logsumexp(x, include_initial=True)
    assert r[0] == -dpt.inf


def test_accumulate_zero_size_dims():
    get_queue_or_skip()

    n0, n1, n2 = 3, 0, 5
    x = dpt.ones((n0, n1, n2), dtype="i8")
    r = dpt.cumulative_sum(x, axis=1)
    assert r.shape == x.shape
    assert r.size == 0

    r = dpt.cumulative_sum(x, axis=0)
    assert r.shape == x.shape
    assert r.size == 0

    r = dpt.cumulative_sum(x, axis=1, include_initial=True)
    assert r.shape == (n0, n1 + 1, n2)
    assert r.size == (n0 * n2)

    r = dpt.cumulative_sum(x, axis=0, include_initial=True)
    assert r.shape == (n0 + 1, n1, n2)
    assert r.size == 0


@pytest.mark.parametrize("arg_dtype", all_types)
def test_cumsum_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    n = 100
    x = dpt.ones(n, dtype=arg_dtype)
    r = dpt.cumulative_sum(x)

    assert isinstance(r, dpt.usm_ndarray)
    if x.dtype.kind == "i":
        assert r.dtype.kind == "i"
    elif x.dtype.kind == "u":
        assert r.dtype.kind == "u"
    elif x.dtype.kind == "fc":
        assert r.dtype == arg_dtype

    r_expected = dpt.arange(1, n + 1, dtype=r.dtype)

    assert dpt.all(r == r_expected)


@pytest.mark.parametrize("arg_dtype", all_types)
@pytest.mark.parametrize("out_dtype", all_types)
def test_cumsum_arg_out_dtype_matrix(arg_dtype, out_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)
    skip_if_dtype_not_supported(out_dtype, q)

    n = 100
    x = dpt.ones(n, dtype=arg_dtype)
    r = dpt.cumulative_sum(x, dtype=out_dtype)

    assert isinstance(r, dpt.usm_ndarray)
    assert r.dtype == dpt.dtype(out_dtype)
    if out_dtype == dpt.bool:
        assert dpt.all(r)
    else:
        r_expected = dpt.arange(1, n + 1, dtype=out_dtype)
        assert dpt.all(r == r_expected)


def test_accumulator_out_kwarg():
    q = get_queue_or_skip()

    n = 100

    expected = dpt.arange(1, n + 1, dtype="i4", sycl_queue=q)
    x = dpt.ones(n, dtype="i4", sycl_queue=q)
    out = dpt.empty_like(x, dtype="i4")
    dpt.cumulative_sum(x, dtype="i4", out=out)
    assert dpt.all(expected == out)

    # overlap
    x = dpt.ones(n, dtype="i4", sycl_queue=q)
    dpt.cumulative_sum(x, dtype="i4", out=x)
    assert dpt.all(x == expected)

    # axis before final axis
    expected = dpt.broadcast_to(
        dpt.arange(1, n + 1, dtype="i4", sycl_queue=q), (n, n)
    ).mT
    x = dpt.ones((n, n), dtype="i4", sycl_queue=q)
    out = dpt.empty_like(x, dtype="i4")
    dpt.cumulative_sum(x, axis=0, dtype="i4", out=out)
    assert dpt.all(expected == out)

    # scalar
    x = dpt.asarray(3, dtype="i4")
    out = dpt.empty((), dtype="i4")
    expected = 3
    dpt.cumulative_sum(x, dtype="i4", out=out)
    assert expected == out


def test_accumulator_arg_validation():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    n = 5
    x1 = dpt.ones((n, n), dtype="f4", sycl_queue=q1)
    x2 = dpt.ones(n, dtype="f4", sycl_queue=q1)

    # must be usm_ndarray
    with pytest.raises(TypeError):
        dpt.cumulative_sum(dict())

    # axis must be specified when input not 1D
    with pytest.raises(ValueError):
        dpt.cumulative_sum(x1)

    # out must be usm_ndarray
    with pytest.raises(TypeError):
        dpt.cumulative_sum(x2, out=dict())

    # out must be writable
    out_not_writable = dpt.empty_like(x2)
    out_not_writable.flags.writable = False
    with pytest.raises(ValueError):
        dpt.cumulative_sum(x2, out=out_not_writable)

    # out must be expected shape
    out_wrong_shape = dpt.ones(n + 1, dtype=x2.dtype, sycl_queue=q1)
    with pytest.raises(ValueError):
        dpt.cumulative_sum(x2, out=out_wrong_shape)

    # out must be expected dtype
    out_wrong_dtype = dpt.empty_like(x2, dtype="i4")
    with pytest.raises(ValueError):
        dpt.cumulative_sum(x2, out=out_wrong_dtype)

    # compute follows data
    out_wrong_queue = dpt.empty_like(x2, sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.cumulative_sum(x2, out=out_wrong_queue)


def test_cumsum_nan_propagation():
    get_queue_or_skip()

    n = 100
    x = dpt.ones(n, dtype="f4")
    i = randrange(n)
    x[i] = dpt.nan

    r = dpt.cumulative_sum(x)
    assert dpt.all(dpt.isnan(r[i:]))


def test_cumprod_nan_propagation():
    get_queue_or_skip()

    n = 100
    x = dpt.ones(n, dtype="f4")
    i = randrange(n)
    x[i] = dpt.nan

    r = dpt.cumulative_prod(x)
    assert dpt.all(dpt.isnan(r[i:]))


def test_logcumsumexp_nan_propagation():
    get_queue_or_skip()

    n = 100
    x = dpt.ones(n, dtype="f4")
    i = randrange(n)
    x[i] = dpt.nan

    r = dpt.cumulative_logsumexp(x)
    assert dpt.all(dpt.isnan(r[i:]))


@pytest.mark.parametrize("arg_dtype", no_complex_types)
def test_logcumsumexp_arg_dtype_default_output_dtype_matrix(arg_dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(arg_dtype, q)

    x = dpt.ones(10, dtype=arg_dtype, sycl_queue=q)
    r = dpt.cumulative_logsumexp(x)

    if arg_dtype.kind in "biu":
        assert r.dtype.kind == "f"
    else:
        assert r.dtype == arg_dtype


def test_logcumsumexp_complex_error():
    get_queue_or_skip()

    x = dpt.ones(10, dtype="c8")
    with pytest.raises(ValueError):
        dpt.cumulative_logsumexp(x)


def test_cumprod_basic():
    get_queue_or_skip()

    n = 50
    val = 2
    x = dpt.full(n, val, dtype="i8")
    r = dpt.cumulative_prod(x)
    expected = dpt.pow(val, dpt.arange(1, n + 1, dtype="i8"))

    assert dpt.all(r == expected)

    x = dpt.tile(dpt.asarray([2, 0.5], dtype="f4"), 10000)
    expected = dpt.tile(dpt.asarray([2, 1], dtype="f4"), 10000)
    r = dpt.cumulative_prod(x)
    assert dpt.all(r == expected)


def test_logcumsumexp_basic():
    get_queue_or_skip()

    dt = dpt.float32
    x = dpt.ones(1000, dtype=dt)
    r = dpt.cumulative_logsumexp(x)

    expected = 1 + dpt.log(dpt.arange(1, 1001, dtype=dt))

    tol = 4 * dpt.finfo(dt).resolution
    assert dpt.allclose(r, expected, atol=tol, rtol=tol)


def geometric_series_closed_form(n, dtype=None, device=None):
    """Closed form for cumulative_logsumexp(dpt.arange(-n, 0))

    :math:`r[k] == -n + k + log(1 - exp(-k-1)) - log(1-exp(-1))`
    """
    x = dpt.arange(-n, 0, dtype=dtype, device=device)
    y = dpt.arange(-1, -n - 1, step=-1, dtype=dtype, device=device)
    y = dpt.exp(y, out=y)
    y = dpt.negative(y, out=y)
    y = dpt.log1p(y, out=y)
    y -= y[0]
    return x + y


@pytest.mark.parametrize("fpdt", rfp_types)
def test_cumulative_logsumexp_closed_form(fpdt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(fpdt, q)

    n = 128
    r = dpt.cumulative_logsumexp(dpt.arange(-n, 0, dtype=fpdt, device=q))
    expected = geometric_series_closed_form(n, dtype=fpdt, device=q)

    tol = 4 * dpt.finfo(fpdt).eps
    assert dpt.allclose(r, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("p", [257, 260, 273, 280, 509, 512])
def test_cumulative_sum_gh_1901(p):
    get_queue_or_skip()

    n = p * p
    dt = dpt.int32
    inp = dpt.ones(n, dtype=dt)
    r = dpt.cumulative_sum(inp, dtype=dt)
    assert dpt.all(r == dpt.arange(1, n + 1, dtype=dt))


@pytest.mark.parametrize(
    "dt", ["i1", "i2", "i4", "i8", "f2", "f4", "f8", "c8", "c16"]
)
def test_gh_2017(dt):
    "See https://github.com/IntelPython/dpctl/issues/2017"
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)
    x = dpt.asarray([-1, 1], dtype=dpt.dtype(dt), sycl_queue=q)
    r = dpt.cumulative_sum(x, dtype="?")
    assert dpt.all(r)
